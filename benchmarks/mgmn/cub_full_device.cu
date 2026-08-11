// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_reduce.cuh>

#include <cuda/__runtime/api_wrapper.h>
#include <cuda/buffer>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/stream>

#include <cstddef>

#include "common.hpp"
#include <nvbench/nvbench.cuh>

namespace
{
//! Baseline: one CUB reduction over the whole device, with no locality partitioning. Every other
//! scenario is ranked against this one.
void cub_full_device(nvbench::state& state)
{
  using T             = float;
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));
  const auto device   = mgmn::state_device(state);
  cuda::stream stream{device};

  const auto input = cuda::make_device_buffer<T>(stream, device, elements, 1.0F);
  auto output      = cuda::make_device_buffer<T>(stream, device, 1, cuda::no_init);
  stream.sync();

  const auto env = cuda::std::execution::env{
    input.memory_resource(), cuda::execution::require(cuda::execution::determinism::not_guaranteed)};

  mgmn::add_common_throughput<T>(state, elements, /*rank_count*/ 1);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    const auto env_with_stream = cuda::std::execution::env{cuda::stream_ref{launch.get_stream().get_stream()}, env};

    _CCCL_TRY_CUDA_API(
      cub::DeviceReduce::Reduce,
      "cub::DeviceReduce::Reduce failed",
      input.begin(),
      output.begin(),
      elements,
      cuda::std::plus<>{},
      T{},
      env_with_stream);
  });
}
} // namespace

NVBENCH_BENCH(cub_full_device)
  .set_name("cub_full_device")
  .add_int64_power_of_two_axis("Elements",
                               nvbench::range(mgmn::min_elements_pow2, mgmn::max_elements_pow2, mgmn::elements_stride));
