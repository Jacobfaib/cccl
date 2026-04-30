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

static void foo()
{
  thrust::device_vector<int> d(10);

  auto shard = cudax::shard{d, cuda::devices[0]};

  cudax::transform(cuda::devices, g);
}

TEST_CASE("basic", "[multi_gpu][transform][basic]")
{
  thrust::device_vector<int> d(10);

  cudax::thread_group g;

  cudax::transform(g);
}
