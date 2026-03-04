//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___SHARDED_ARRAY_CUH
#define _CUDAX___SHARDED_ARRAY_CUH

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/std/mdspan>
#include <cuda/std/tuple>
#include <cuda/std/utility>

namespace cuda::experimental
{
template <size_t I, typename... Ts>
using type_at_index_t = std::tuple_element_t<I, std::tuple<Ts...>>;

template <typename... MDSpans>
class basic_sharded_mdarray
{
public:
  static_assert(std::conjunction_v<
                std::is_same<typename type_at_index_t<0, MDSpans...>::value_type, typename MDSpans::value_type>...>);

  template <typename MDSpan>
  class shard_type
  {
  public:
    int device{};
    MDSpan mdspan{};
  };

  template <typename... MDSpan, typename = std::enable_if_t<(sizeof...(MDSpan) > 0)>>
  basic_sharded_mdarray(const ::cuda::std::pair<int, MDSpan>&... rest)
      : shards_{{rest.first, rest.second}...}
  {}

  [[nodiscard]] const ::cuda::std::tuple<shard_type<MDSpans>...>& shards() const
  {
    return shards_;
  }

  [[nodiscard]] ::cuda::std::tuple<shard_type<MDSpans>...>& shards()
  {
    return shards_;
  }

private:
  ::cuda::std::tuple<shard_type<MDSpans>...> shards_{};
};

template <typename... MDSpans, typename F, size_t... Is>
inline void transform(const basic_sharded_mdarray<MDSpans...>& mdarray, F&& functor, std::index_sequence<Is...>)
{
  auto&& shards = mdarray.shards();

  (([&] {
     auto&& shard = std::get<Is>(shards);

     cuDeviceSet(shard.device);
     functor<<<...>>>(shard.mdspan);
   }()),
   ...);
}

template <typename... MDSpans, typename F>
inline void transform(const basic_sharded_mdarray<MDSpans...>& mdarray, F&& functor)
{
  transform(mdarray, std::forward<F>(functor), std::index_sequence_for<MDSpans...>{});
}

inline void foo()
{
  auto v1 = thrust::device_vector<int>{};
  auto v2 = thrust::device_vector<int>{};

  auto v1_span = ::cuda::std::mdspan{thrust::raw_pointer_cast(v1.data()), 1, 2};
  auto v2_span = ::cuda::std::mdspan{thrust::raw_pointer_cast(v2.data()), 4, 5};
  auto sharded = basic_sharded_mdarray{std::make_pair(0, v1_span), std::make_pair(1, v2_span)};

  transform(sharded)
}
} // namespace cuda::experimental
#endif
