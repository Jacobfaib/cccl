//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FWD_ITERATOR_H
#define _CUDA_STD___FWD_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/copyable.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__type_traits/enable_if.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Container>
class _CCCL_TYPE_VISIBILITY_DEFAULT __back_insert_iterator;

template <class, class = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT iterator_traits;

_LIBCUDACXX_BEGIN_HIDDEN_FRIEND_NAMESPACE

template <class _Iter>
class _CCCL_TYPE_VISIBILITY_DEFAULT reverse_iterator;

_LIBCUDACXX_END_HIDDEN_FRIEND_NAMESPACE(reverse_iterator)

// _LIBCUDACXX_BEGIN_HIDDEN_FRIEND_NAMESPACE

// #if _CCCL_HAS_CONCEPTS()
// template <input_or_output_iterator _Iter, sentinel_for<_Iter> _Sent>
//   requires(!same_as<_Iter, _Sent> && copyable<_Iter>)
// #else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
// template <class _Iter,
//           class _Sent,
//           enable_if_t<input_or_output_iterator<_Iter>, int>             = 0,
//           enable_if_t<sentinel_for<_Sent, _Iter>, int>                  = 0,
//           enable_if_t<(!same_as<_Iter, _Sent> && copyable<_Iter>), int> = 0>
// #endif // !_CCCL_HAS_CONCEPTS()
// class _CCCL_TYPE_VISIBILITY_DEFAULT common_iterator;

// _LIBCUDACXX_END_HIDDEN_FRIEND_NAMESPACE(common_iterator)

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FWD_ITERATOR_H
