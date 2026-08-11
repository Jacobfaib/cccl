// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cuda/__event/event.h>
#include <cuda/buffer>
#include <cuda/std/execution>
#include <cuda/std/ranges>
#include <cuda/stream>

#include <cuda/experimental/__device/logical_device.cuh>
#include <cuda/experimental/__multi_gpu/algorithm/reduce/reduce.h>
#include <cuda/experimental/__multi_gpu/nccl_communicator.h>
#include <cuda/experimental/stream.cuh>

#include <cstddef>
#include <memory>
#include <vector>

#include "common.hpp"
#include "locality_domain.hpp"
#include "locality_domain_resource.hpp"
#include "nccl_support.hpp"
#include <nvbench/nvbench.cuh>

namespace
{
//! Split the input across `rank_count` green contexts and reduce with `cudax::reduce`, which
//! performs a local CUB reduction per rank followed by a host-launched NCCL collective.
void cudax_host_nccl(nvbench::state& state)
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

  // Split the SM resource by locality domain and give each partition its own non-blocking
  // stream created directly against the green context via `cuGreenCtxStreamCreate`.
  const auto contexts = mgmn::make_domain_contexts(device, rank_count);

  std::vector<cudax::nccl_communicator> communicators;
  {
    std::vector<ncclComm_t> raw_comms(rank_count);
    std::vector<int> devs(rank_count);

    mgmn::check_nccl(ncclCommInitAll(raw_comms.data(), devs.size(), devs.data()), "ncclCommInitAll");
    for (int rank = 0; rank < rank_count; ++rank)
    {
      communicators.emplace_back(
        cudax::nccl_communicator::from_native_handle(raw_comms[rank], cudax::logical_device{contexts[rank]}));
    }
  }

  std::vector<cuda::stream> streams;
  // Data locality: one memory-pool-backed resource per domain. The owning resource is non-movable
  // (it has sole responsibility for its pool), hence the indirection.
  std::vector<std::unique_ptr<mgmn::locality_domain_resource>> resources;
  // The env carries the domain's memory resource alongside its stream, so the temporary storage
  // `cudax::reduce` allocates internally is drawn from that domain's localized pool as well. Without
  // it the algorithm falls back to the device default pool, which is not localized.
  using env_type = decltype(cuda::std::execution::env{cuda::stream_ref{streams[0]}, resources[0]->ref()});

  std::vector<env_type> environments;
  // Each green context owns its input share and its output scalar, both drawn from that domain's
  // localized pool. The green context is made current so the fill kernel that writes the initial
  // values also runs on that domain's SMs.
  std::vector<cuda::device_buffer<float>> inputs_buf;
  std::vector<cuda::device_buffer<float>> outputs;
  std::vector<cuda::device_buffer<float>::iterator> output_its;

  streams.reserve(rank_count);
  resources.reserve(rank_count);
  environments.reserve(rank_count);
  inputs_buf.reserve(rank_count);
  outputs.reserve(rank_count);
  output_its.reserve(rank_count);
  for (int rank = 0; rank < rank_count; ++rank)
  {
    cuda::stream_ref s =
      streams.emplace_back(cuda::stream::from_native_handle(mgmn::create_green_ctx_stream(contexts[rank].__green_ctx)));
    auto res =
      resources.emplace_back(std::make_unique<mgmn::locality_domain_resource>(device, static_cast<unsigned int>(rank)))
        ->ref();
    environments.emplace_back(cuda::std::execution::env{s, res});

    inputs_buf.emplace_back(cuda::make_buffer<float>(s, res, per_rank, 1.0F));
    auto& o = outputs.emplace_back(cuda::make_buffer<float>(s, res, 1, cuda::no_init));
    output_its.emplace_back(o.begin());
  }

  // Confirm the pools honored the request before timing anything; a silent fallback to
  // non-localized memory would make the measurement meaningless.
  for (int rank = 0; rank < rank_count; ++rank)
  {
    if (mgmn::locality::pointer_domain(inputs_buf[rank].data()) != static_cast<unsigned int>(rank))
    {
      state.skip("an input buffer did not land in its requested locality domain");
      return;
    }
  }

  // Built once: creating events inside the measured region would charge the measurement for that
  // host work.
  cuda::event fork{device};
  std::vector<cuda::event> join;
  join.reserve(rank_count);
  for (int rank = 0; rank < rank_count; ++rank)
  {
    join.emplace_back(device);
  }

  mgmn::add_common_throughput(state, elements, rank_count);
  mgmn::add_domain_count(state, rank_count);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    mgmn::run_forked_iteration(cuda::stream_ref{launch.get_stream().get_stream()}, streams, fork, join, [&] {
      cudax::reduce(
        cudax::broadcasted,
        communicators,
        environments,
        inputs_buf | cuda::std::views::transform(cuda::std::ranges::begin),
        inputs_buf | cuda::std::views::transform(cuda::std::ranges::size),
        output_its);
    });
  });
}
} // namespace

NVBENCH_BENCH(cudax_host_nccl)
  .set_name("cudax_host_nccl")
  .add_int64_power_of_two_axis("Elements",
                               nvbench::range(mgmn::min_elements_pow2, mgmn::max_elements_pow2, mgmn::elements_stride));
