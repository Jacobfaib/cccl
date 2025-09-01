//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: true

// If the invocation of any non-const member function of `iterator` exits via an
// exception, the iterator acquires a singular value.

#include <cuda/std/ranges>
#include <cuda/std/tuple>

#include "../types.h"
#include "test_macros.h"

#ifndef __CUDA_ARCH__
struct ThrowOnIncrementIterator
{
  int* it_;

  using value_type       = int;
  using difference_type  = cuda::std::intptr_t;
  using iterator_concept = cuda::std::input_iterator_tag;

  ThrowOnIncrementIterator() = default;
  TEST_FUNC explicit ThrowOnIncrementIterator(int* it)
      : it_(it)
  {}

  TEST_FUNC ThrowOnIncrementIterator& operator++()
  {
    ++it_;
    throw 5;
    return *this;
  }
  TEST_FUNC void operator++(int)
  {
    ++it_;
  }

  TEST_FUNC int& operator*() const
  {
    return *it_;
  }

#  if TEST_STD_VER >= 2020
  friend bool operator==(ThrowOnIncrementIterator const&, ThrowOnIncrementIterator const&) = default;
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
  TEST_FUNC friend bool operator==(ThrowOnIncrementIterator const&, ThrowOnIncrementIterator const&);
  TEST_FUNC friend bool operator!=(ThrowOnIncrementIterator const&, ThrowOnIncrementIterator const&);
#  endif // TEST_STD_VER <=2017
};

struct ThrowOnIncrementView : IntBufferView
{
  TEST_FUNC ThrowOnIncrementIterator begin() const
  {
    return ThrowOnIncrementIterator{buffer_};
  }
  TEST_FUNC ThrowOnIncrementIterator end() const
  {
    return ThrowOnIncrementIterator{buffer_ + size_};
  }
};

// Cannot run the test at compile time because it is not allowed to throw exceptions
TEST_FUNC void test()
{
  int buffer[] = {1, 2, 3};
  {
    // zip iterator should be able to be destroyed after member function throws
    cuda::std::ranges::zip_view v{ThrowOnIncrementView{buffer}};
    auto it = v.begin();
    try
    {
      ++it;
      assert(false); // should not be reached as the above expression should throw.
    }
    catch (int e)
    {
      assert(e == 5);
    }
  }

  {
    // zip iterator should be able to be assigned after member function throws
    cuda::std::ranges::zip_view v{ThrowOnIncrementView{buffer}};
    auto it = v.begin();
    try
    {
      ++it;
      assert(false); // should not be reached as the above expression should throw.
    }
    catch (int e)
    {
      assert(e == 5);
    }
    it       = v.begin();
    auto [x] = *it;
    assert(x == 1);
  }
}

int main(int, char**)
{
  test();

  return 0;
}
#else // ^^^ !__CUDA_ARCH__ ^^^ / vvv __CUDA_ARCH__ vvv
int main(int, char**)
{
  return 0;
}
#endif // __CUDA_ARCH__
