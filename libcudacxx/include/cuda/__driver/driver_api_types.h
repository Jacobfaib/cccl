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

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

struct CUctx_st;
struct CUstream_st;
struct CUmemcpyAttributes_v1;
struct CUmemPoolHandle_st;
struct CUmemPoolProps_v1;
struct CUmemPool_v1;

_CCCL_BEGIN_NAMESPACE_CUDA_DRIVER

using CUresult                       = int;
using CUdevice                       = int;
using cuuint64_t                     = ::cuda::std::uint64_t;
using CUdriverProcAddressQueryResult = int;
using CUdevice_attribute             = int;

using CUcontext          = ::CUctx_st*;
using CUstream           = ::CUstream_st*;
using CUmemcpyAttributes = ::CUmemcpyAttributes_v1;

using CUmemoryPool        = ::CUmemPoolHandle_st;
using CUmemPoolProps      = ::CUmemPoolProps_v1;
using CUmemPool_attribute = int;

// Note: not _CCCL_OS(WINDOWS)! We care specifically about 64-bit here, not that it is windows.
#if defined(_WIN64) || defined(__LP64__)
// Don't use std::uint64_t, we want to match the driver headers exactly
using CUdeviceptr = unsigned long long; // NOLINT(google-runtime-int)
#else
using CUdeviceptr = unsigned int;
#endif

#define CCCL_CU_GET_PROC_ADDRESS_DEFAULT                0
#define CCCL_CU_GET_PROC_ADDRESS_SUCCESS                0
#define CCCL_CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT 2

#define CCCL_CUDA_SUCCESS             0
#define CCCL_CUDA_ERROR_INVALID_VALUE 1
#define CCCL_CUDA_ERROR_NOT_SUPPORTED 801
#define CCCL_CUDA_ERROR_UNKNOWN       999

namespace api
{
using cuGetProcAddress_v2_t = ::cuda::__driver::CUresult (*)(
  const char*, void**, int, ::cuda::__driver::cuuint64_t, ::cuda::__driver::CUdriverProcAddressQueryResult);
using cuInit_t = ::cuda::__driver::CUresult (*)(unsigned int);

using cuDriverGetVersion_t = ::cuda::__driver::CUresult (*)(int*);

// Device
using cuDeviceGet_t = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUdevice*, int);
using cuDeviceGetAttribute_t =
  ::cuda::__driver::CUresult (*)(int*, ::cuda::__driver::CUdevice_attribute, ::cuda::__driver::CUdevice);
using cuDeviceGetCount_t = ::cuda::__driver::CUresult (*)(int*);
using cuDeviceGetName_t  = ::cuda::__driver::CUresult (*)(char*, int, int);

// Primary context
using cuDevicePrimaryCtxRetain_t =
  ::cuda::__driver::CUresult (*)(::cuda::__driver::CUcontext, ::cuda::__driver::CUdevice);
using cuDevicePrimaryCtxRelease_t  = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUdevice);
using cuDeviceIsPrimaryCtxActive_t = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUdevice, unsigned int*, int*);

// Context
using cuCtxPushCurrent_t = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUcontext);
using cuCtxPopCurrent_t  = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUcontext*);
using cuCtxGetCurrent_t  = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUcontext*);
using cuCtxGetDevice_t   = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUdevice*);

// Memory

using cuMemcpyAsync_t =
  ::cuda::__driver::CUresult (*)(void*, const void*, ::cuda::__driver::CUdeviceptr, ::cuda::__driver::CUstream);
using cuMemcpyBatchAsync_t = ::cuda::__driver::CUresult (*)(
  ::cuda::__driver::CUdeviceptr* dsts,
  ::cuda::__driver::CUdeviceptr* srcs,
  ::cuda::std::size_t* sizes,
  ::cuda::std::size_t count,
  ::cuda::__driver::CUmemcpyAttributes* attrs,
  ::cuda::std::size_t* attrsIdxs,
  ::cuda::std::size_t numAttrs,
  ::cuda::__driver::CUstream hStream);
using cuMemsetD8Async_t = ::cuda::__driver::CUresult (*)(
  ::cuda::__driver::CUdeviceptr, unsigned char, ::cuda::std::size_t, ::cuda::__driver::CUstream);
using cuMemsetD16Async_t = ::cuda::__driver::CUresult (*)(
  ::cuda::__driver::CUdeviceptr, unsigned short, ::cuda::std::size_t, ::cuda::__driver::CUstream);
using cuMemsetD32Async_t = ::cuda::__driver::CUresult (*)(
  ::cuda::__driver::CUdeviceptr, unsigned int, ::cuda::std::size_t, ::cuda::__driver::CUstream);

using cuMemPoolCreate_t =
  ::cuda::__driver::CUresult (*)(::cuda::__driver::CUmemoryPool*, const ::cuda::__driver::CUmemPoolProps*);
using cuMemPoolDestroy_t = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUmemoryPool);
using cuMemPoolSetAttribute_t =
  ::cuda::__driver::CUresult (*)(::cuda::__driver::CUmemoryPool, ::cuda::__driver::CUmemPool_attribute, void*);
using cuMemPoolGetAttribute_t =
  ::cuda::__driver::CUresult (*)(::cuda::__driver::CUmemoryPool, ::cuda::__driver::CUmemPool_attribute, void*);
using cuMemAllocFromPoolAsync_t = ::cuda::__driver::CUresult (*)(
  ::cuda::__driver::CUdeviceptr*, ::cuda::std::size_t, ::cuda::__driver::CUmemoryPool, ::cuda::__driver::CUstream);
using cuMemPoolTrimTo_t = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUmemoryPool, ::cuda::std::size_t);

using cuMemFreeAsync_t = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUdeviceptr, ::cuda::__driver::CUstream);
} // namespace api

_CCCL_END_NAMESPACE_CUDA_DRIVER

#endif // _CUDA___DRIVER_DRIVER_API_TYPES_H
