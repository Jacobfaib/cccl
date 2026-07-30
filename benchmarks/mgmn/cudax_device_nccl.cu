// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_reduce.cuh>

#include <cuda/__driver/driver_api.h>
#include <cuda/__event/event.h>
#include <cuda/__event/timed_event.h>
#include <cuda/__runtime/ensure_current_context.h>
#include <cuda/buffer>
#include <cuda/std/execution>
#include <cuda/stream>

#include <array>
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

namespace
{
//! Per-rank scalar windows registered as NCCL symmetric memory plus the device communicator.
struct rank_windows
{
  ncclDevComm devcomm{};
  ncclWindow_t source{};
  ncclWindow_t destination{};
};

//! Terminal-epilogue hook for the device NCCL path. Each domain's final CUB reduction publishes its
//! local aggregate into its registered source window, then every rank fuses the rank-local
//! aggregates with a single-element device-side reduce/copy over the LSA team, bracketed by LSA
//! barriers. No host-launched collective occurs in the timed interval.
//!
//! CUB invokes the terminal epilogue from a single thread of a single block (see the
//! `threadIdx.x == 0` guard in `kernel_reduce.cuh`), so `ncclCoopThread` is the cooperation level
//! that matches the caller and barrier index 0 is uncontended.
struct device_nccl_epilogue
{
  float* local_aggregate{};
  ncclDevComm devcomm{};
  ncclWindow_t source{};
  ncclWindow_t destination{};

  _CCCL_DEVICE_API void operator()(float value) const noexcept
  {
    *local_aggregate = value;

    // `ncclLsaReduceSumCopy` is not rank-collective in the sense of one rank doing the work for
    // all: every rank issues the call for its own region. Guarding it to a single rank leaves the
    // others idling in the trailing barrier while that rank performs every remote read serially.
    const ncclCoopThread cooperative = ncclCoopThread{};
    ncclLsaBarrierSession<ncclCoopThread> barrier{cooperative, devcomm, ncclTeamTagLsa{}, 0};
    barrier.sync(cooperative, cuda::memory_order_acquire);
    ncclLsaReduceSumCopy<float>(cooperative, source, 0, destination, 0, 1, ncclTeamLsa(devcomm));
    barrier.sync(cooperative, cuda::memory_order_release);
  }
};

void benchmark_cudax_device_nccl(benchmark::State& state)
{
  const auto elements = static_cast<std::size_t>(state.range(0));
  const auto device   = cuda::devices[0];

  cudaSetDevice(device.get());
  cudaDeviceSynchronize();
  device.init();

  // One rank per locality domain, so each rank's SMs and its data sit in the same partition.
  // `ncclLsaReduceSumCopy` reduces across the whole LSA team, so any rank count works.
  const auto rank_count = static_cast<int>(mgmn::locality::domain_count(device));
  if (rank_count < 2)
  {
    state.SkipWithError("the GPU does not expose multiple locality domains");
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

  cuda::timed_event start{device};
  cuda::timed_event stop{device};
  std::vector<cuda::event> completed;
  completed.reserve(rank_count);
  for (int rank = 0; rank != rank_count; ++rank)
  {
    completed.emplace_back(device);
  }

  // Each domain owns its input share and CUB destination, drawn from that domain's localized pool.
  // The green context is made current so the fill kernel that writes the initial values also runs
  // on that domain's SMs.
  std::vector<cuda::device_buffer<float>> inputs;
  std::vector<cuda::device_buffer<float>> outputs;
  // The scalars registered as NCCL symmetric windows must come from `ncclMemAlloc` rather than from
  // any memory pool: `ncclCommWindowRegister` resolves the backing allocation with
  // `cuMemGetAddressRange`, which rejects stream-ordered pool allocations - both the localized pool
  // and the device default pool - with `invalid argument`. Their placement is otherwise irrelevant
  // here, being single floats touched once per reduction by the epilogue.
  std::vector<mgmn::nccl_buffer<float>> aggregates;
  std::vector<mgmn::nccl_buffer<float>> destinations;

  inputs.reserve(rank_count);
  outputs.reserve(rank_count);
  aggregates.reserve(rank_count);
  destinations.reserve(rank_count);
  for (int rank = 0; rank != rank_count; ++rank)
  {
    cuda::__ensure_current_context guard{contexts[rank].__transformed};
    const auto resource = resources[rank]->ref();
    inputs.emplace_back(cuda::make_buffer<float>(streams[rank], resource, per_rank, 1.0F));
    outputs.emplace_back(cuda::make_buffer<float>(streams[rank], resource, 1, cuda::no_init));
    aggregates.emplace_back(1);
    destinations.emplace_back(1);
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
      state.SkipWithError("an input buffer did not land in its requested locality domain");
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
      mgmn::check_nccl(
        ncclCommWindowRegister(
          host_communicators[rank],
          destinations[rank].data(),
          destinations[rank].size_bytes(),
          &windows[rank].destination,
          NCCL_WIN_COLL_SYMMETRIC),
        "ncclCommWindowRegister(destination)");

      ncclDevCommRequirements_t requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
      requirements.lsaBarrierCount           = 1;
      mgmn::check_nccl(ncclDevCommCreate(host_communicators[rank], &requirements, &windows[rank].devcomm),
                       "ncclDevCommCreate");
    };

    std::vector<std::future<void>> initialization(rank_count);
    for (int rank = 0; rank != rank_count; ++rank)
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

  // The env pairs the domain's stream with its terminal epilogue. Both are fixed for the whole run,
  // so they are built once here rather than per iteration: constructing them inside the timed loop
  // would charge the measurement for host-side work that is not part of the reduction.
  using env_type =
    decltype(cuda::std::execution::env{cuda::stream_ref{streams[0]}, cub::terminal_epilogue(device_nccl_epilogue{})});

  std::vector<env_type> envs;
  envs.reserve(rank_count);
  for (int rank = 0; rank != rank_count; ++rank)
  {
    envs.emplace_back(cuda::std::execution::env{
      cuda::stream_ref{streams[rank]},
      cub::terminal_epilogue(device_nccl_epilogue{
        aggregates[rank].data(), windows[rank].devcomm, windows[rank].source, windows[rank].destination})});
  }

  for (auto _ : state)
  {
    static_cast<void>(_);
    start.record(streams.front());
    // Launch one reduction per locality domain, each over its own domain-local input. The stream was
    // created against the domain's green context, so the kernel is confined to that domain's SMs.
    for (int rank = 0; rank != rank_count; ++rank)
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
    // Record every rank's completion before waiting on any of them. Interleaving the record and
    // the wait makes each wait a barrier against the host issuing the next record, which shows up
    // directly in the measurement at these timescales.
    for (int rank = 1; rank != rank_count; ++rank)
    {
      completed[rank].record(streams[rank]);
    }
    for (int rank = 1; rank != rank_count; ++rank)
    {
      streams.front().wait(completed[rank]);
    }
    stop.record(streams.front());
    stop.sync();
    state.SetIterationTime(static_cast<double>((stop - start).count()) / 1'000'000'000.0);
  }

  const auto teardown_rank = [&](int rank) {
    cuda::__ensure_current_context guard{contexts[rank].__transformed};
    mgmn::check_nccl(ncclDevCommDestroy(host_communicators[rank], &windows[rank].devcomm), "ncclDevCommDestroy");
    mgmn::check_nccl(ncclCommWindowDeregister(host_communicators[rank], windows[rank].source),
                     "ncclCommWindowDeregister(source)");
    mgmn::check_nccl(ncclCommWindowDeregister(host_communicators[rank], windows[rank].destination),
                     "ncclCommWindowDeregister(destination)");
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

  const auto sm_count = ::cuda::__driver::__deviceGetAttribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device.get());
  mgmn::set_common_counters(state, static_cast<std::size_t>(elements), static_cast<unsigned int>(sm_count));
  state.counters["locality_domains"] = static_cast<double>(rank_count);
}
} // namespace

int main(int argc, char** argv)
{
  return mgmn::run_benchmark(argc, argv, "cudax_device_nccl", benchmark_cudax_device_nccl);
}
