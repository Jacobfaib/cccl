/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_adjacent_difference.cuh>

#include <thrust/iterator/discard_iterator.h>

#include <algorithm>
#include <numeric>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/custom_type.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceAdjacentDifference::SubtractRight, adjacent_difference_subtract_right);
DECLARE_LAUNCH_WRAPPER(cub::DeviceAdjacentDifference::SubtractRightCopy, adjacent_difference_subtract_right_copy);

// %PARAM% TEST_LAUNCH lid 0:1:2

using all_types =
  c2h::type_list<std::uint8_t,
                 std::uint64_t,
                 std::int8_t,
                 std::int64_t,
                 ulonglong2,
                 c2h::custom_type_t<c2h::equal_comparable_t, c2h::subtractable_t>>;

using types = c2h::type_list<std::uint8_t, std::int32_t>;

C2H_TEST("DeviceAdjacentDifference::SubtractRight can run with empty input", "[device][adjacent_difference]", types)
{
  using type = typename c2h::get<0, TestType>;

  constexpr int num_items = 0;
  c2h::device_vector<type> in(num_items);

  adjacent_difference_subtract_right(in.begin(), num_items, ::cuda::std::minus<>{});
}

C2H_TEST("DeviceAdjacentDifference::SubtractRightCopy can run with empty input", "[device][adjacent_difference]", types)
{
  using type = typename c2h::get<0, TestType>;

  constexpr int num_items = 0;
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);

  adjacent_difference_subtract_right_copy(in.begin(), out.begin(), num_items, ::cuda::std::minus<>{});
}

C2H_TEST("DeviceAdjacentDifference::SubtractRightCopy does not change the input", "[device][adjacent_difference]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::gen(C2H_SEED(2), in);

  c2h::device_vector<type> reference = in;
  adjacent_difference_subtract_right_copy(in.begin(), thrust::discard_iterator<>(), num_items, ::cuda::std::minus<>{});

  REQUIRE(reference == in);
}

_CCCL_SUPPRESS_DEPRECATED_PUSH
template <class T>
struct ref_diff
{
  template <class T2, cuda::std::enable_if_t<cuda::std::is_same_v<T, T2>, int> = 0>
  __host__ __device__ constexpr T2 operator()(const T2& lhs, const T2& rhs) const noexcept
  {
    return rhs - lhs;
  }

  __host__ __device__ constexpr ulonglong2 operator()(const ulonglong2& lhs, const ulonglong2& rhs) const noexcept
  {
    return ulonglong2{rhs.x - lhs.x, rhs.y - lhs.y};
  }
  __host__ __device__ constexpr ulonglong4 operator()(const ulonglong4& lhs, const ulonglong4& rhs) const noexcept
  {
    return ulonglong4{rhs.x - lhs.x, rhs.y - lhs.y, rhs.z - lhs.z, rhs.w - lhs.w};
  }

  __host__ __device__ constexpr long2 operator()(const long2& lhs, const long2& rhs) const noexcept
  {
    return long2{rhs.x - lhs.x, rhs.y - lhs.y};
  }
};
_CCCL_SUPPRESS_DEPRECATED_POP

C2H_TEST("DeviceAdjacentDifference::SubtractRight works with iterators", "[device][adjacent_difference]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(C2H_SEED(2), in);

  c2h::host_vector<type> h_in = in;
  c2h::host_vector<type> reference(num_items);
  std::adjacent_difference(h_in.begin(), h_in.end(), reference.begin(), ref_diff<type>{});
  std::rotate(reference.begin(), reference.begin() + 1, reference.end());
  reference.back() = h_in.back();

  adjacent_difference_subtract_right(in.begin(), num_items, ::cuda::std::minus<>{});

  REQUIRE(reference == in);
}

C2H_TEST("DeviceAdjacentDifference::SubtractRightCopy works with iterators", "[device][adjacent_difference]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(C2H_SEED(2), in);

  c2h::host_vector<type> h_in = in;
  c2h::host_vector<type> reference(num_items);
  std::adjacent_difference(h_in.begin(), h_in.end(), reference.begin(), ref_diff<type>{});
  std::rotate(reference.begin(), reference.begin() + 1, reference.end());
  reference.back() = h_in.back();

  adjacent_difference_subtract_right_copy(in.begin(), out.begin(), num_items, ::cuda::std::minus<>{});

  REQUIRE(reference == out);
}

C2H_TEST("DeviceAdjacentDifference::SubtractRight works with pointers", "[device][adjacent_difference]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::gen(C2H_SEED(2), in);

  c2h::host_vector<type> h_in = in;
  c2h::host_vector<type> reference(num_items);
  std::adjacent_difference(h_in.begin(), h_in.end(), reference.begin(), ref_diff<type>{});
  std::rotate(reference.begin(), reference.begin() + 1, reference.end());
  reference.back() = h_in.back();

  adjacent_difference_subtract_right(thrust::raw_pointer_cast(in.data()), num_items, ::cuda::std::minus<>{});

  REQUIRE(reference == in);
}

C2H_TEST("DeviceAdjacentDifference::SubtractRightCopy works with pointers", "[device][adjacent_difference]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(C2H_SEED(2), in);

  c2h::host_vector<type> h_in = in;
  c2h::host_vector<type> reference(num_items);
  std::adjacent_difference(h_in.begin(), h_in.end(), reference.begin(), ref_diff<type>{});
  std::rotate(reference.begin(), reference.begin() + 1, reference.end());
  reference.back() = h_in.back();

  adjacent_difference_subtract_right_copy(
    thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), num_items, ::cuda::std::minus<>{});

  REQUIRE(reference == out);
}

_CCCL_SUPPRESS_DEPRECATED_PUSH
struct cust_diff
{
  template <class T>
  __host__ __device__ constexpr T operator()(const T& lhs, const T& rhs) const noexcept
  {
    return lhs - rhs;
  }

  __host__ __device__ constexpr ulonglong2 operator()(const ulonglong2& lhs, const ulonglong2& rhs) const noexcept
  {
    return ulonglong2{lhs.x - rhs.x, lhs.y - rhs.y};
  }

  __host__ __device__ constexpr ulonglong4 operator()(const ulonglong4& lhs, const ulonglong4& rhs) const noexcept
  {
    return ulonglong4{lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w};
  }

  __host__ __device__ constexpr long2 operator()(const long2& lhs, const long2& rhs) const noexcept
  {
    return long2{lhs.x - rhs.x, lhs.y - rhs.y};
  }
};
_CCCL_SUPPRESS_DEPRECATED_POP

C2H_TEST("DeviceAdjacentDifference::SubtractRight works with custom difference",
         "[device][adjacent_difference]",
         all_types)
{
  using type = typename c2h::get<0, TestType>;

  constexpr int num_items = 5; // GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::gen(C2H_SEED(2), in);

  c2h::host_vector<type> h_in = in;
  c2h::host_vector<type> reference(num_items);
  std::adjacent_difference(h_in.begin(), h_in.end(), reference.begin(), ref_diff<type>{});
  std::rotate(reference.begin(), reference.begin() + 1, reference.end());
  reference.back() = h_in.back();

  adjacent_difference_subtract_right(in.begin(), num_items, cust_diff{});

  REQUIRE(reference == in);
}

C2H_TEST("DeviceAdjacentDifference::SubtractRightCopy works with custom difference",
         "[device][adjacent_difference]",
         types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(C2H_SEED(2), in);

  c2h::host_vector<type> h_in = in;
  c2h::host_vector<type> reference(num_items);
  std::adjacent_difference(h_in.begin(), h_in.end(), reference.begin(), ref_diff<type>{});
  std::rotate(reference.begin(), reference.begin() + 1, reference.end());
  reference.back() = h_in.back();

  adjacent_difference_subtract_right_copy(in.begin(), out.begin(), num_items, cust_diff{});

  REQUIRE(reference == out);
}

template <class T>
struct convertible_from_T
{
  T val_;

  convertible_from_T() = default;
  __host__ __device__ convertible_from_T(const T& val) noexcept
      : val_(val)
  {}
  __host__ __device__ convertible_from_T& operator=(const T& val) noexcept
  {
    val_ = val;
  }
  // Converting back to T helps satisfy all the machinery that T supports
  __host__ __device__ operator T() const noexcept
  {
    return val_;
  }
};

C2H_TEST("DeviceAdjacentDifference::SubtractRightCopy works with a different output type",
         "[device][adjacent_difference]",
         types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<convertible_from_T<type>> out(num_items);
  c2h::gen(C2H_SEED(2), in);

  c2h::host_vector<type> h_in = in;
  c2h::host_vector<type> reference(num_items);
  std::adjacent_difference(h_in.begin(), h_in.end(), reference.begin(), ref_diff<type>{});
  std::rotate(reference.begin(), reference.begin() + 1, reference.end());
  reference.back() = h_in.back();

  adjacent_difference_subtract_right_copy(in.begin(), out.begin(), num_items, cust_diff{});

  REQUIRE(reference == out);
}

struct check_difference
{
  int* d_error;

  template <class T>
  __device__ T operator()(const T& lhs, const T& rhs) const noexcept
  {
    const T result = rhs - lhs;
    if (result != 1)
    {
      atomicAdd(d_error, 1);
    }
    return result;
  }
};

C2H_TEST("DeviceAdjacentDifference::SubtractRightCopy works with large indexes",
         "[device][adjacent_difference][skip-cs-racecheck][skip-cs-initcheck][skip-cs-synccheck]")
{
  constexpr cuda::std::size_t num_items = 1ll << 33;
  c2h::device_vector<int> error(1);
  int* d_error = thrust::raw_pointer_cast(error.data());
  adjacent_difference_subtract_right_copy(
    thrust::counting_iterator<cuda::std::size_t>{0}, thrust::discard_iterator<>{}, num_items, check_difference{d_error});
  const int h_error = error[0];
  REQUIRE(h_error == 0);
}

struct invocation_counter
{
  __host__ explicit invocation_counter(unsigned long long* addr)
      : counts_(addr)
  {}

  template <class T>
  __device__ T operator()(const T& lhs, const T& rhs) const noexcept
  {
    // Use legacy atomics to support testing on older archs:
    atomicAdd(counts_, 1ull);
    return lhs - rhs;
  }

private:
  unsigned long long* counts_;
};

C2H_TEST("DeviceAdjacentDifference::SubtractRightCopy uses right number of invocations",
         "[device][adjacent_difference]")
{
  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<unsigned long long> counts(1, 0);
  adjacent_difference_subtract_right_copy(
    thrust::counting_iterator<cuda::std::size_t>{0},
    thrust::discard_iterator<>(),
    num_items,
    invocation_counter{thrust::raw_pointer_cast(counts.data())});

  REQUIRE(counts.front() == static_cast<unsigned long long>(num_items - 1));
}
