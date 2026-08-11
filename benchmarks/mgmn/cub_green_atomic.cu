// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_reduce.cuh>

#include <cuda/__event/event.h>
#include <cuda/__runtime/api_wrapper.h>
#include <cuda/__runtime/ensure_current_context.h>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/stream>

#include <cuda/experimental/stream.cuh>

#include <cstddef>
#include <memory>
#include <vector>

#include <cuda_runtime_api.h>

#include "common.hpp"
#include "locality_domain.hpp"
#include "locality_domain_resource.hpp"
#include <nvbench/nvbench.cuh>

namespace
{
//! Terminal-epilogue hook: each green context's final CUB reduction adds its local aggregate
//! into one device-scope location with a single relaxed floating-point atomic.
struct atomic_epilogue
{
  float* aggregate{};

  template <typename OutputIteratorT>
  _CCCL_DEVICE_API void operator()(float value, OutputIteratorT&& d_out) const
  {
    ::cuda::atomic_ref<float, ::cuda::thread_scope_device>{*aggregate}.fetch_add(value, ::cuda::memory_order_relaxed);
    *d_out = value;
  }
};

void cub_green_atomic(nvbench::state& state)
{
  using T             = float;
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

  // Execution locality: split the SM resource *by locality domain id* (not by SM count), so green
  // context `rank` owns exactly the SMs of domain `rank`. Each gets a non-blocking stream created
  // directly against it via `cuGreenCtxStreamCreate`.
  const auto contexts = mgmn::make_domain_contexts(device, rank_count);

  std::vector<cuda::stream> streams;
  // Data locality: one memory-pool-backed resource per domain. The owning resource is non-movable
  // (it has sole responsibility for its pool), hence the indirection.
  std::vector<std::unique_ptr<mgmn::locality_domain_resource>> resources;

  streams.reserve(rank_count);
  resources.reserve(rank_count);
  for (int rank = 0; rank < rank_count; ++rank)
  {
    streams.emplace_back(cuda::stream::from_native_handle(mgmn::create_green_ctx_stream(contexts[rank].__green_ctx)));
    resources.push_back(std::make_unique<mgmn::locality_domain_resource>(device, static_cast<unsigned int>(rank)));
  }

  // The aggregate is touched by every domain's atomic, so it has no natural home; leave it in
  // ordinary device memory rather than biasing it toward one domain.
  auto aggregate = cuda::make_device_buffer<T>(cuda::stream_ref{cudaStream_t{}}, device, 1, cuda::no_init);

  // The env carries the domain's memory resource alongside its stream, so the temporary storage CUB
  // allocates for its two-pass reduction is drawn from that domain's localized pool rather than the
  // non-localized device default pool.
  using env_type = decltype(cuda::std::execution::env{
    cuda::stream_ref{streams[0]},
    resources[0]->ref(),
    cub::terminal_epilogue(atomic_epilogue{aggregate.data()}),
    cuda::execution::require(cuda::execution::determinism::not_guaranteed)});

  std::vector<cuda::device_buffer<T>> inputs;
  std::vector<cuda::device_buffer<T>> local_outputs;
  std::vector<env_type> envs;

  inputs.reserve(rank_count);
  local_outputs.reserve(rank_count);
  envs.reserve(rank_count);

  for (int rank = 0; rank < rank_count; ++rank)
  {
    // Allocated from the domain-local pool, with the domain's green context current so the fill
    // kernel that writes the initial values also runs on that domain's SMs.
    cuda::__ensure_current_context guard{contexts[rank].__transformed};
    inputs.emplace_back(cuda::make_buffer<T>(streams[rank], resources[rank]->ref(), elements / rank_count, T{1}));
    local_outputs.emplace_back(cuda::make_buffer<T>(streams[rank], resources[rank]->ref(), 1, cuda::no_init));
    envs.emplace_back(cuda::std::execution::env{
      cuda::stream_ref{streams[rank]},
      resources[rank]->ref(),
      cub::terminal_epilogue(atomic_epilogue{aggregate.data()}),
      cuda::execution::require(cuda::execution::determinism::not_guaranteed)});
  }

  for (auto&& s : streams)
  {
    s.sync();
  }

  // Confirm the pools honored the request before timing anything; a silent fallback to
  // non-localized memory would make the measurement meaningless.
  for (int rank = 0; rank < rank_count; ++rank)
  {
    if (mgmn::locality::pointer_domain(inputs[rank].data()) != static_cast<unsigned int>(rank))
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

  mgmn::add_common_throughput<T>(state, elements, rank_count);
  mgmn::add_domain_count(state, rank_count);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    mgmn::run_forked_iteration(cuda::stream_ref{launch.get_stream().get_stream()}, streams, fork, join, [&] {
      for (int rank = 0; rank < rank_count; ++rank)
      {
        _CCCL_TRY_CUDA_API(
          cub::DeviceReduce::Reduce,
          "Terminal-epilogue CUB reduction failed",
          inputs[rank].data(),
          local_outputs[rank].data(),
          inputs[rank].size(),
          cuda::std::plus<>{},
          T{},
          envs[rank]);
      }
    });
  });
}
} // namespace

NVBENCH_BENCH(cub_green_atomic)
  .set_name("cub_green_atomic")
  .add_int64_power_of_two_axis("Elements",
                               nvbench::range(mgmn::min_elements_pow2, mgmn::max_elements_pow2, mgmn::elements_stride));
