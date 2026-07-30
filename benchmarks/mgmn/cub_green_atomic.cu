// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_reduce.cuh>

#include <cuda/__driver/driver_api.h>
#include <cuda/__event/event.h>
#include <cuda/__event/timed_event.h>
#include <cuda/__runtime/ensure_current_context.h>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/std/execution>
#include <cuda/stream>

#include <cuda/experimental/stream.cuh>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "common.hpp"
#include "locality_domain.hpp"
#include "locality_domain_resource.hpp"

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

void benchmark_cub_green_atomic(benchmark::State& state)
{
  const auto elements = static_cast<std::size_t>(state.range(0));
  const auto device   = cuda::devices[0];

  cudaSetDevice(device.get());
  cudaDeviceSynchronize();
  device.init();

  // One rank per locality domain, so each rank's SMs and its data sit in the same partition.
  const auto rank_count = mgmn::locality::domain_count(device);

  if (rank_count < 2)
  {
    state.SkipWithError("the GPU does not expose multiple locality domains");
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
  auto aggregate = cuda::make_device_buffer<float>(cuda::stream_ref{cudaStream_t{}}, device, 1, cuda::no_init);

  // The env carries the domain's memory resource alongside its stream, so the temporary storage CUB
  // allocates for its two-pass reduction is drawn from that domain's localized pool rather than the
  // non-localized device default pool.
  using env_type = decltype(cuda::std::execution::env{
    cuda::stream_ref{streams[0]}, resources[0]->ref(), cub::terminal_epilogue(atomic_epilogue{aggregate.data()})});

  std::vector<cuda::device_buffer<float>> inputs;
  std::vector<cuda::device_buffer<float>> local_outputs;
  std::vector<env_type> envs;

  inputs.reserve(rank_count);
  local_outputs.reserve(rank_count);
  envs.reserve(rank_count);

  for (int rank = 0; rank < rank_count; ++rank)
  {
    // Allocated from the domain-local pool, with the domain's green context current so the fill
    // kernel that writes the initial values also runs on that domain's SMs.
    cuda::__ensure_current_context guard{contexts[rank].__transformed};
    inputs.emplace_back(cuda::make_buffer<float>(streams[rank], resources[rank]->ref(), elements / rank_count, 1.0F));
    local_outputs.emplace_back(cuda::make_buffer<float>(streams[rank], resources[rank]->ref(), 1, cuda::no_init));
    envs.emplace_back(cuda::std::execution::env{
      cuda::stream_ref{streams[rank]},
      resources[rank]->ref(),
      cub::terminal_epilogue(atomic_epilogue{aggregate.data()})});
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
      state.SkipWithError("an input buffer did not land in its requested locality domain");
      return;
    }
  }

  cuda::timed_event start{device};
  cuda::timed_event stop{device};
  std::vector<cuda::event> completed;
  completed.reserve(rank_count);
  for (int rank = 0; rank < rank_count; ++rank)
  {
    completed.emplace_back(device);
  }

  for (auto _ : state)
  {
    static_cast<void>(_);
    start.record(streams.front());
    for (int rank = 0; rank < rank_count; ++rank)
    {
      // cuda::__ensure_current_context guard{contexts[rank].__transformed};
      _CCCL_TRY_CUDA_API(
        cub::DeviceReduce::Reduce,
        "Terminal-epilogue CUB reduction failed",
        inputs[rank].data(),
        local_outputs[rank].data(),
        inputs[rank].size(),
        cuda::std::plus<>{},
        0.0F,
        envs[rank]);
    }
    // Record every rank's completion before waiting on any of them. Interleaving the record and
    // the wait makes each wait a barrier against the host issuing the next record, which shows up
    // directly in the measurement at these timescales.
    for (int rank = 1; rank < rank_count; ++rank)
    {
      completed[rank].record(streams[rank]);
    }
    for (int rank = 1; rank < rank_count; ++rank)
    {
      streams.front().wait(completed[rank]);
    }
    stop.record(streams.front());
    stop.sync();
    state.SetIterationTime(static_cast<double>((stop - start).count()) / 1'000'000'000.0);
  }

  const auto sm_count = ::cuda::__driver::__deviceGetAttribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device.get());
  mgmn::set_common_counters(state, elements, static_cast<unsigned int>(sm_count));
  state.counters["locality_domains"] = static_cast<double>(rank_count);
}
} // namespace

int main(int argc, char** argv)
{
  return mgmn::run_benchmark(argc, argv, "cub_green_atomic", benchmark_cub_green_atomic);
}
