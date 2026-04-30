//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/device_vector.h>

#include <cuda/experimental/__multi_gpu/sharded_container.h>
#include <cuda/experimental/__multi_gpu/thread_group.h>
#include <cuda/experimental/__multi_gpu/transform.h>

#include <testing.cuh>

[[nodiscard]] cudax::thread_group make_group() {}

static void foo() {}

TEST_CASE("basic", "[multi_gpu][transform][basic]")
{
  constexpr auto size = 10;
  thrust::device_vector<int> d_input(size, 1);
  thrust::device_vector<int> d_output(size, 0);

  auto input_shard  = cudax::shard{d_input, cuda::devices[0]};
  auto output_shard = cudax::shard{d_output, cuda::devices[0]};
  auto in_buf       = cudax::sharded_buffer{input_shard};
  auto out_buf      = cudax::sharded_buffer{output_shard};

  cudax::transform(cuda::devices, in_buf, out_buf, [](int x) {
    return x + 1;
  });

  thrust::host_vector<int> h = d_output;

  for (auto&& v : h)
  {
    REQUIRE(v == 2);
  }
}
