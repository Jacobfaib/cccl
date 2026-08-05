// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_reduce.cuh>

#include <cuda/__event/event.h>
#include <cuda/__runtime/api_wrapper.h>
#include <cuda/__runtime/ensure_current_context.h>
#include <cuda/buffer>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/stream>

#include <cstddef>
#include <future>
#include <memory>
#include <stdexcept>
#include <vector>

#include <nccl_device.h>

#include "common.hpp"
#include "locality_domain.hpp"
#include "locality_domain_resource.hpp"
#include "nccl_support.hpp"
#include <nvbench/nvbench.cuh>

namespace
{
//! Per-rank scalar windows registered as NCCL symmetric memory plus the device communicator.
struct rank_windows
{
  ncclDevComm devcomm{};
  ncclWindow_t source{};
  ncclWindow_t destination{};
};

//! Terminal-epilogue hook for the device NCCL path: every rank publishes its aggregate into its
//! symmetric window, rank 0 sums them and publishes the total, then every rank stores that total to
//! its own output, matching the broadcast semantics of `cudax::reduce(cudax::broadcasted, ...)`.
struct device_nccl_epilogue
{
  ncclDevComm devcomm{};
  ncclWindow_t window{};

  template <typename OutputIteratorT>
  _CCCL_DEVICE_API void operator()(float value, OutputIteratorT) const noexcept
  {
    const auto cooperative = ncclCoopCta{};
    ncclLsaBarrierSession<ncclCoopCta> barrier{cooperative, devcomm, ncclTeamTagLsa(), blockIdx.x};
    const int rank   = devcomm.rank;
    const int nRanks = devcomm.nRanks;

    barrier.sync(cooperative, cuda::memory_order_acquire);

    auto* const local_pointer = static_cast<float*>(ncclGetLocalPointer(window, 0));

    for (int peer = 0; peer < nRanks; ++peer)
    {
      if (rank == peer)
      {
        continue;
      }
      value += *static_cast<const float*>(ncclGetLsaPointer(window, 0, rank));
    }
    *local_pointer = value;

    barrier.sync(cooperative, cuda::memory_order_release);
  }
};

void cudax_device_nccl(nvbench::state& state)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));
  const auto device   = mgmn::state_device(state);

  cudaSetDevice(device.get());
  cudaDeviceSynchronize();
  device.init();

  // One rank per locality domain, so each rank's SMs and its data sit in the same partition.
  const auto rank_count = static_cast<int>(mgmn::locality::domain_count(device));
  if (rank_count < 2)
  {
    state.skip("the GPU does not expose multiple locality domains");
    return;
  }
  const auto per_rank = elements / rank_count;

  // Execution locality: one green context per locality domain, each with its own non-blocking
  // stream created directly against it via `cuGreenCtxStreamCreate`.
  const auto contexts = mgmn::make_domain_contexts(device, rank_count);

  std::vector<cuda::stream> streams;
  // Data locality: one memory-pool-backed resource per domain. The owning resource is non-movable
  // (it has sole responsibility for its pool), hence the indirection.
  std::vector<std::unique_ptr<mgmn::locality_domain_resource>> resources;

  streams.reserve(rank_count);
  resources.reserve(rank_count);
  for (int rank = 0; rank != rank_count; ++rank)
  {
    streams.emplace_back(cuda::stream::from_native_handle(mgmn::create_green_ctx_stream(contexts[rank].__green_ctx)));
    resources.push_back(std::make_unique<mgmn::locality_domain_resource>(device, static_cast<unsigned int>(rank)));
  }

  // Built once: creating events inside the measured region would charge the measurement for that
  // host work.
  cuda::event fork{device};
  std::vector<cuda::event> join;
  join.reserve(rank_count);
  for (int rank = 0; rank != rank_count; ++rank)
  {
    join.emplace_back(device);
  }

  std::vector<cuda::device_buffer<float>> inputs;
  std::vector<cuda::device_buffer<float>> outputs;
  // Registered windows must come from `ncclMemAlloc`: `ncclCommWindowRegister` resolves the backing
  // allocation with `cuMemGetAddressRange`, which rejects stream-ordered pool allocations.
  std::vector<mgmn::nccl_buffer<float>> aggregates;

  inputs.reserve(rank_count);
  outputs.reserve(rank_count);
  aggregates.reserve(rank_count);
  for (int rank = 0; rank != rank_count; ++rank)
  {
    cuda::__ensure_current_context guard{contexts[rank].__transformed};
    const auto resource = resources[rank]->ref();
    inputs.emplace_back(cuda::make_buffer<float>(streams[rank], resource, per_rank, 1.0F));
    outputs.emplace_back(cuda::make_buffer<float>(streams[rank], resource, 1, cuda::no_init));
    aggregates.emplace_back(1);
  }
  for (auto&& s : streams)
  {
    s.sync();
  }

  // Confirm the pools honored the request before timing anything; a silent fallback to
  // non-localized memory would make the measurement meaningless.
  for (int rank = 0; rank != rank_count; ++rank)
  {
    if (mgmn::locality::pointer_domain(inputs[rank].data()) != static_cast<unsigned int>(rank))
    {
      state.skip("an input buffer did not land in its requested locality domain");
      return;
    }
  }

  // Initialize host and device communicators per green context, registering the rank-local
  // aggregate and destination scalars as symmetric windows. ncclCommInitRank and
  // ncclDevCommCreate are collective, so each rank owns its host thread with its context current.
  ncclUniqueId unique_id{};
  mgmn::check_nccl(ncclGetUniqueId(&unique_id), "ncclGetUniqueId");
  std::vector<ncclComm_t> host_communicators(rank_count);
  std::vector<rank_windows> windows(rank_count);

  {
    const auto setup_rank = [&](int rank) {
      cuda::__ensure_current_context guard{contexts[rank].__transformed};
      mgmn::check_nccl(ncclCommInitRank(&host_communicators[rank], rank_count, unique_id, rank), "ncclCommInitRank");

      ncclCommProperties_t properties = NCCL_COMM_PROPERTIES_INITIALIZER;
      mgmn::check_nccl(ncclCommQueryProperties(host_communicators[rank], &properties), "ncclCommQueryProperties");
      if (!properties.deviceApiSupport)
      {
        throw std::runtime_error("NCCL device API is unavailable on this build");
      }

      mgmn::check_nccl(
        ncclCommWindowRegister(
          host_communicators[rank],
          aggregates[rank].data(),
          aggregates[rank].size_bytes(),
          &windows[rank].source,
          NCCL_WIN_COLL_SYMMETRIC),
        "ncclCommWindowRegister(source)");

      ncclDevCommRequirements_t requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
      requirements.lsaBarrierCount           = 1;
      mgmn::check_nccl(ncclDevCommCreate(host_communicators[rank], &requirements, &windows[rank].devcomm),
                       "ncclDevCommCreate");
    };

    std::vector<std::future<void>> initialization(rank_count);
    for (int rank = 0; rank < rank_count; ++rank)
    {
      initialization[rank] = std::async(std::launch::async, setup_rank, rank);
    }
    for (auto& task : initialization)
    {
      task.get();
    }
  }

  for (auto&& s : streams)
  {
    s.sync();
  }

  // The env carries the domain's memory resource alongside its stream, so the temporary storage CUB
  // allocates for its two-pass reduction is drawn from that domain's localized pool. Without it the
  // dispatch falls back to the device default pool, which is not localized, and the second pass then
  // reads the partial aggregates across domains on the critical path.
  //
  // Built once rather than per iteration, which would charge the measurement for host-side work.
  using env_type = decltype(cuda::std::execution::env{
    cuda::stream_ref{streams[0]}, resources[0]->ref(), cub::terminal_epilogue(device_nccl_epilogue{})});

  std::vector<env_type> envs;
  envs.reserve(rank_count);
  for (int rank = 0; rank != rank_count; ++rank)
  {
    envs.emplace_back(cuda::std::execution::env{
      cuda::stream_ref{streams[rank]},
      resources[rank]->ref(),
      cub::terminal_epilogue(device_nccl_epilogue{windows[rank].devcomm, windows[rank].source})});
  }

  mgmn::add_common_throughput(state, elements, rank_count);
  mgmn::add_domain_count(state, rank_count);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    mgmn::run_forked_iteration(cuda::stream_ref{launch.get_stream().get_stream()}, streams, fork, join, [&] {
      for (int rank = 0; rank < rank_count; ++rank)
      {
        _CCCL_TRY_CUDA_API(
          cub::DeviceReduce::Reduce,
          "Device-NCCL terminal-epilogue reduction failed",
          inputs[rank].data(),
          outputs[rank].data(),
          per_rank,
          cuda::std::plus<>{},
          0.0F,
          envs[rank]);
      }
    });
  });

  const auto teardown_rank = [&](int rank) {
    cuda::__ensure_current_context guard{contexts[rank].__transformed};
    mgmn::check_nccl(ncclDevCommDestroy(host_communicators[rank], &windows[rank].devcomm), "ncclDevCommDestroy");
    mgmn::check_nccl(ncclCommWindowDeregister(host_communicators[rank], windows[rank].source),
                     "ncclCommWindowDeregister(source)");
    mgmn::check_nccl(ncclCommDestroy(host_communicators[rank]), "ncclCommDestroy");
  };
  std::vector<std::future<void>> teardown(rank_count);
  for (int rank = 0; rank != rank_count; ++rank)
  {
    teardown[rank] = std::async(std::launch::async, teardown_rank, rank);
  }
  for (auto& task : teardown)
  {
    task.get();
  }
}
} // namespace

NVBENCH_BENCH(cudax_device_nccl)
  .set_name("cudax_device_nccl")
  .add_int64_power_of_two_axis("Elements",
                               nvbench::range(mgmn::min_elements_pow2, mgmn::max_elements_pow2, mgmn::elements_stride));
