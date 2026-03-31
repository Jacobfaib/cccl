//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___DRIVER_DRIVER_API_TYPES_H
#define _CUDA___DRIVER_DRIVER_API_TYPES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstdint>

#if _CCCL_HAS_CTK()
#  include <cuda.h>
#endif

_CCCL_BEGIN_NAMESPACE_CUDA_DRIVER

#if _CCCL_HAS_CTK()
using ::CUdevice;
using ::CUresult;
using ::cuuint64_t;
using enum CUdriverProcAddressQueryResult;

#  define CCCL_CU_GET_PROC_ADDRESS_DEFAULT                CU_GET_PROC_ADDRESS_DEFAULT
#  define CCCL_CU_GET_PROC_ADDRESS_SUCCESS                CU_GET_PROC_ADDRESS_SUCCESS
#  define CCCL_CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT

#  define CCCL_CUDA_SUCCESS             CUDA_SUCCESS
#  define CCCL_CUDA_ERROR_INVALID_VALUE CUDA_ERROR_INVALID_VALUE
#  define CCCL_CUDA_ERROR_UNKNOWN       CUDA_ERROR_UNKNOWN
#  define CCCL_CUDA_ERROR_NOT_SUPPORTED CUDA_ERROR_NOT_SUPPORTED

#else // ^^^ _CCCL_HAS_CTK() ^^^ / vvv !_CCCL_HAS_CTK() vvv
using CUresult                       = int;
using CUdevice                       = int;
using cuuint64_t                     = ::cuda::std::uint64_t;
using CUdriverProcAddressQueryResult = int;

#  define CCCL_CU_GET_PROC_ADDRESS_DEFAULT                0
#  define CCCL_CU_GET_PROC_ADDRESS_SUCCESS                0
#  define CCCL_CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT 2

#  define CCCL_CUDA_SUCCESS             0
#  define CCCL_CUDA_ERROR_INVALID_VALUE 1
#  define CCCL_CUDA_ERROR_NOT_SUPPORTED 801
#  define CCCL_CUDA_ERROR_UNKNOWN       999

#endif // !_CCCL_HAS_CTK()

namespace api
{
using cuGetProcAddress_v2_t = ::cuda::__driver::CUresult (*)(
  const char*, void**, int, ::cuda::__driver::cuuint64_t, ::cuda::__driver::CUdriverProcAddressQueryResult);
using cuInit_t             = ::cuda::__driver::CUresult (*)(unsigned int);
using cuDriverGetVersion_t = ::cuda::__driver::CUresult (*)(int*);
using cuDeviceGet_t        = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUdevice*, int);
} // namespace api

_CCCL_END_NAMESPACE_CUDA_DRIVER

#endif // _CUDA___DRIVER_DRIVER_API_TYPES_H
