// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_reduce.cuh>

#include <cuda/__driver/driver_api.h>
#include <cuda/__event/event.h>
#include <cuda/__event/timed_event.h>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/std/execution>
#include <cuda/stream>

#include <cuda/experimental/stream.cuh>

#include <cstddef>
#include <vector>

#include "common.hpp"
#include "green_ctx_partition.hpp"

namespace
{
//! Terminal-epilogue hook: each green context's final CUB reduction adds its local aggregate
//! into one device-scope location with a single relaxed floating-point atomic.
struct atomic_epilogue
{
  float* aggregate{};

  _CCCL_DEVICE_API void operator()(float value) const
  {
    ::cuda::atomic_ref<float, ::cuda::thread_scope_device>{*aggregate}.fetch_add(value, ::cuda::memory_order_relaxed);
  }
};

void benchmark_cub_green_atomic(benchmark::State& state)
{
  constexpr int rank_count = 2;
  const auto elements      = static_cast<std::size_t>(state.range(0));
  const auto device        = cuda::devices[0];

  // Split the device into equal green-context halves and give each its own non-blocking stream
  // created directly against the green context via `cuGreenCtxStreamCreate`.
  const auto contexts = mgmn::make_green_halves(device, rank_count);

  std::vector<cuda::stream> streams;

  streams.reserve(rank_count);
  for (int rank = 0; rank < rank_count; ++rank)
  {
    streams.emplace_back(cuda::stream::from_native_handle(mgmn::create_green_ctx_stream(contexts[rank].__green_ctx)));
  }

  std::vector<cuda::device_buffer<float>> inputs;
  std::vector<cuda::device_buffer<float>> local_outputs;

  auto aggregate = cuda::make_device_buffer<float>(cuda::stream_ref{cudaStream_t{}}, device, 1, cuda::no_init);

  using env_type = decltype(cuda::std::execution::env{
    cuda::stream_ref{streams[0]}, cub::terminal_epilogue(atomic_epilogue{aggregate.data()})});

  std::vector<env_type> envs;

  inputs.reserve(rank_count);
  local_outputs.reserve(rank_count);
  envs.reserve(rank_count);

  for (int rank = 0; rank < rank_count; ++rank)
  {
    inputs.emplace_back(cuda::make_device_buffer<float>(streams[rank], device, elements / rank_count, 1.0F));
    local_outputs.emplace_back(cuda::make_device_buffer<float>(streams[rank], device, 1, cuda::no_init));
    envs.emplace_back(cuda::std::execution::env{
      cuda::stream_ref{streams[rank]}, cub::terminal_epilogue(atomic_epilogue{aggregate.data()})});
  }

  cuda::timed_event start{device};
  cuda::timed_event stop{device};
  cuda::event completed{device};

  for (auto&& s : streams)
  {
    s.sync();
  }

  for (auto _ : state)
  {
    static_cast<void>(_);
    // Establish a common start boundary: record `start` on stream 0 and make stream 1 wait on it.
    start.record(streams.front());
    for (int rank = 1; rank < rank_count; ++rank)
    {
      streams[rank].wait(start);
    }
    for (int rank = 0; rank != rank_count; ++rank)
    {
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
    // Join both halves onto stream 0, then record and time the stop boundary.
    completed.record(streams.back());
    for (int rank = 0; rank < rank_count - 1; ++rank)
    {
      streams[rank].wait(completed);
    }
    stop.record(streams.front());
    stop.sync();
    state.SetIterationTime(static_cast<double>((stop - start).count()) / 1'000'000'000.0);
  }
  const auto sm_count = ::cuda::__driver::__deviceGetAttribute(
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, ::cuda::__driver::__deviceGet(device.get()));
  mgmn::set_common_counters(state, elements, static_cast<unsigned int>(sm_count));
}
} // namespace

int main(int argc, char** argv)
{
  return mgmn::run_benchmark(argc, argv, "cub_green_atomic", benchmark_cub_green_atomic);
}
