// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_reduce.cuh>

#include <cuda/__event/event.h>
#include <cuda/__event/timed_event.h>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/std/execution>
#include <cuda/stream>

#include <array>
#include <cstddef>
#include <vector>

#include "common.hpp"
#include "green_context_support.hpp"

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
  const auto elements = static_cast<std::size_t>(state.range(0));
  const auto half     = elements / 2;

  mgmn::green_partition partition{cuda::devices[0]};
  const auto device = partition.device();

  cuda::stream coordinator{device};
  cuda::timed_event start{device};
  cuda::timed_event stop{device};
  std::array<cuda::event, mgmn::green_partition::rank_count> completed{cuda::event{device}, cuda::event{device}};

  // Each green context owns its input half and local output; only the atomic target is shared.
  const std::vector<float> half_values(half, 1.0F);
  std::array inputs{cuda::make_device_buffer<float>(partition.stream(0), device, half_values),
                    cuda::make_device_buffer<float>(partition.stream(1), device, half_values)};

  // TODO: this needs to allocate per uGPU, need to figure out how to do that.
  std::array local_outputs{cuda::make_device_buffer<float>(partition.stream(0), device, 1, cuda::no_init),
                           cuda::make_device_buffer<float>(partition.stream(1), device, 1, cuda::no_init)};
  auto aggregate = cuda::make_device_buffer<float>(coordinator, device, 1, cuda::no_init);
  partition.stream(0).sync();
  partition.stream(1).sync();
  coordinator.sync();

  for (auto _ : state)
  {
    static_cast<void>(_);
    mgmn::begin_partition_timing(partition, coordinator, start);
    for (int rank = 0; rank != mgmn::green_partition::rank_count; ++rank)
    {
      const auto environment = cuda::std::execution::env{
        cuda::stream_ref{partition.stream(rank)}, cub::terminal_epilogue(atomic_epilogue{aggregate.data()})};
      _CCCL_TRY_CUDA_API(
        cub::DeviceReduce::Reduce,
        "Terminal-epilogue CUB reduction failed",
        inputs[rank].data(),
        local_outputs[rank].data(),
        half,
        cuda::std::plus<>{},
        0.0F,
        environment);
    }
    mgmn::end_partition_timing(partition, coordinator, start, stop, completed, state);
  }
  mgmn::set_common_counters(state, elements, partition.sm_count());
}
} // namespace

int main(int argc, char** argv)
{
  return mgmn::run_benchmark(argc, argv, "cub_green_atomic", benchmark_cub_green_atomic);
}
