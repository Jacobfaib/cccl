//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/buffer>
#include <cuda/memory_resource>
#include <cuda/std/cstdint>
#include <cuda/std/execution>
#include <cuda/std/type_traits>

#include <cuda/experimental/__multi_gpu/algorithm/reduce/reduce.h>

#include <numeric>
#include <vector>

#include <testing.cuh>

#include "../../communicators/nccl/nccl_test_helpers.cuh"

namespace
{
template <class T>
[[nodiscard]] std::vector<typename cuda::device_buffer<T>::iterator>
make_output_iterators(const std::vector<cuda::device_buffer<T>>& out)
{
  std::vector<typename cuda::device_buffer<T>::iterator> outputs;

  outputs.reserve(out.size());
  for (auto& buf : out)
  {
    outputs.push_back(buf.begin());
  }
  return outputs;
}
} // namespace

MGMN_TEST("reduce, range overloads default values", )
{
  using T = cuda::std::int32_t;

  constexpr auto init = T{};

  auto comms   = this->communicators();
  auto streams = nccl_test_util::make_streams();

  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<decltype(::cuda::std::execution::env{::cuda::stream_ref{streams[0]}})> envs;

  in.reserve(comms.size());
  out.reserve(comms.size());
  envs.reserve(comms.size());
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto values = {static_cast<T>(comms[i].rank())};

    in.emplace_back(cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), values));
    out.emplace_back(
      cuda::make_device_buffer<T>(streams[i], comms[i].logical_device().underlying_device(), 1, cuda::no_init));
    envs.emplace_back(::cuda::std::execution::env{::cuda::stream_ref{streams[i]}});
  }

  auto outputs = make_output_iterators(out);

  const auto expected = [&] {
    std::vector<T> reference;

    for (int r = 0; r < comms.front().size(); ++r)
    {
      reference.push_back(r);
    }

    return std::accumulate(reference.begin(), reference.end(), init);
  }();

  SECTION("Default init, op, ident (all)")
  {
    cudax::reduce(comms, envs, in, outputs);

    for (const auto& buf : out)
    {
      const auto exp = cuda::make_buffer(buf.stream(), cuda::mr::legacy_pinned_memory_resource{}, 1, expected);

      REQUIRE_THAT(buf, Equals(exp));
    }
  }

  SECTION("Default op, ident")
  {
    cudax::reduce(comms, envs, in, outputs, init);

    for (const auto& buf : out)
    {
      const auto exp = cuda::make_buffer(buf.stream(), cuda::mr::legacy_pinned_memory_resource{}, 1, expected);

      REQUIRE_THAT(buf, Equals(exp));
    }
  }

  SECTION("Default ident")
  {
    cudax::reduce(comms, envs, in, outputs, init, ::cuda::std::plus<>{});

    for (const auto& buf : out)
    {
      const auto exp = cuda::make_buffer(buf.stream(), cuda::mr::legacy_pinned_memory_resource{}, 1, expected);

      REQUIRE_THAT(buf, Equals(exp));
    }
  }

  SECTION("Default none")
  {
    // Identity same as init
    cudax::reduce(comms, envs, in, outputs, init, ::cuda::std::plus<>{}, /*ident=*/init);

    for (const auto& buf : out)
    {
      const auto exp = cuda::make_buffer(buf.stream(), cuda::mr::legacy_pinned_memory_resource{}, 1, expected);

      REQUIRE_THAT(buf, Equals(exp));
    }
  }
}
