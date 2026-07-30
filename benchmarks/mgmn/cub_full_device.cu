// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_reduce.cuh>

#include <cuda/__event/timed_event.h>
#include <cuda/__runtime/api_wrapper.h>
#include <cuda/buffer>
#include <cuda/stream>

#include <cstdint>
#include <vector>

#include "common.hpp"

namespace
{
void benchmark_cub_full_device(benchmark::State& state)
{
  const auto elements = static_cast<std::size_t>(state.range(0));
  const auto device   = cuda::devices[0];
  cuda::stream stream{device};
  const auto input = cuda::make_device_buffer<float>(stream, device, elements, 1.0F);
  auto output      = cuda::make_device_buffer<float>(stream, device, 1, cuda::no_init);

  const auto env = cuda::std::execution::env{
    cuda::stream_ref{stream}, input.memory_resource(),
    cuda::execution::require(cuda::execution::determinism::not_guaranteed)};

  cuda::timed_event start{device};
  cuda::timed_event stop{device};

  for (auto _ : state)
  {
    static_cast<void>(_);
    start.record(stream);
    _CCCL_TRY_CUDA_API(
		       cub::DeviceReduce::Reduce, "cub::DeviceReduce::Reduce failed", input.begin(), output.begin(), elements, ::cuda::std::plus<>{}, float{}, env);
    stop.record(stream);
    stop.sync();
    state.SetIterationTime(static_cast<double>((stop - start).count()) / 1'000'000'000.0);
  }
  mgmn::set_common_counters(state, elements);
}
} // namespace

int main(int argc, char** argv)
{
  return mgmn::run_benchmark(argc, argv, "cub_full_device", benchmark_cub_full_device);
}
