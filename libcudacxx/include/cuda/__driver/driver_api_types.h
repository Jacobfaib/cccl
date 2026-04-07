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
struct CUmemAccessDesc_v1;
struct CUmemLocation_v1;
struct CUevent_st;
struct CUfunc_st;
struct CUkern_st;
struct CUlib_st;
struct CUgraph_st;
struct CUgraphNode_st;
struct CUgreenCtx_st;
struct CUdevResource_st;
struct CUlaunchConfig_st;
struct CUtensorMap_st;
union CUkernelNodeAttrValue;
struct CUDA_KERNEL_NODE_PARAMS_v2;

_CCCL_BEGIN_NAMESPACE_CUDA_DRIVER

// Note: not _CCCL_OS(WINDOWS)! We care specifically about 64-bit here, not that it is windows.
#if defined(_WIN64) || defined(__LP64__)
// Don't use std::uint64_t, we want to match the driver headers exactly
using CUdeviceptr = unsigned long long; // NOLINT(google-runtime-int)
#else
using CUdeviceptr = unsigned int;
#endif

using CUresult   = int;
using CUdevice   = int;
using cuuint64_t = ::cuda::std::uint64_t;

using CUmemPoolProps          = ::CUmemPoolProps_v1;
using CUmemAccessDesc         = ::CUmemAccessDesc_v1;
using CUmemLocation           = ::CUmemLocation_v1;
using CUmemcpyAttributes      = ::CUmemcpyAttributes_v1;
using CUkernelNodeAttrValue   = ::CUkernelNodeAttrValue;
using CUDA_KERNEL_NODE_PARAMS = ::CUDA_KERNEL_NODE_PARAMS_v2;

using CUcontext         = ::CUctx_st*;
using CUstream          = ::CUstream_st*;
using CUevent           = ::CUevent_st*;
using CUfunction        = ::CUfunc_st*;
using CUkernel          = ::CUkern_st*;
using CUlibrary         = ::CUlib_st*;
using CUgraph           = ::CUgraph_st*;
using CUgraphNode       = ::CUgraphNode_st*;
using CUgreenCtx        = ::CUgreenCtx_st*;
using CUdevResourceDesc = ::CUdevResource_st*;
using CUlaunchConfig    = ::CUlaunchConfig_st;
using CUtensorMap       = ::CUtensorMap_st;
using CUmemoryPool      = ::CUmemPoolHandle_st*;

using CUmemAccess_flags              = int;
using CUmemPool_attribute            = int;
using CUmemAllocationType_enum       = int;
using CUpointer_attribute            = int;
using CUfunction_attribute           = int;
using CUjit_option                   = int;
using CUlibraryOption                = int;
using CUkernelNodeAttrID             = int;
using CUtensorMapDataType            = int;
using CUtensorMapInterleave          = int;
using CUtensorMapSwizzle             = int;
using CUtensorMapL2promotion         = int;
using CUtensorMapFloatOOBfill        = int;
using CUmemorytype                   = int;
using CUmemAccess_flags              = int;
using CUmemPool_attribute            = int;
using CUmemAllocationType_enum       = int;
using CUdriverProcAddressQueryResult = int;
using CUdevice_attribute             = int;

using CUstreamCallback = void (*)(::cuda::__driver::CUstream, ::cuda::__driver::CUresult, void*);
using CUhostFn         = void (*)(void*);

#define CCCL_CU_GET_PROC_ADDRESS_DEFAULT 0

#define CCCL_CU_GET_PROC_ADDRESS_SUCCESS                0
#define CCCL_CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT 2

#define CCCL_CUDA_SUCCESS               0
#define CCCL_CUDA_ERROR_NOT_INITIALIZED 3
#define CCCL_CUDA_ERROR_INVALID_VALUE   1
#define CCCL_CUDA_ERROR_NOT_SUPPORTED   801
#define CCCL_CUDA_ERROR_UNKNOWN         999

#define CCCL_CU_EVENT_WAIT_DEFAULT       0
#define CCCL_CU_GREEN_CTX_DEFAULT_STREAM 1

#define CCCL_CU_POINTER_ATTRIBUTE_CONTEXT        1
#define CCCL_CU_POINTER_ATTRIBUTE_MEMORY_TYPE    2
#define CCCL_CU_POINTER_ATTRIBUTE_DEVICE_POINTER 3
#define CCCL_CU_POINTER_ATTRIBUTE_HOST_POINTER   4
#define CCCL_CU_POINTER_ATTRIBUTE_IS_MANAGED     9
#define CCCL_CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL 12
#define CCCL_CU_POINTER_ATTRIBUTE_MAPPED         14
#define CCCL_CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE 18

#define CCCL_CU_TENSOR_MAP_DATA_TYPE_UINT8         0
#define CCCL_CU_TENSOR_MAP_DATA_TYPE_UINT16        1
#define CCCL_CU_TENSOR_MAP_DATA_TYPE_UINT32        2
#define CCCL_CU_TENSOR_MAP_DATA_TYPE_INT32         3
#define CCCL_CU_TENSOR_MAP_DATA_TYPE_UINT64        4
#define CCCL_CU_TENSOR_MAP_DATA_TYPE_INT64         5
#define CCCL_CU_TENSOR_MAP_DATA_TYPE_FLOAT16       6
#define CCCL_CU_TENSOR_MAP_DATA_TYPE_FLOAT32       7
#define CCCL_CU_TENSOR_MAP_DATA_TYPE_FLOAT64       8
#define CCCL_CU_TENSOR_MAP_DATA_TYPE_BFLOAT16      9
#define CCCL_CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ   10
#define CCCL_CU_TENSOR_MAP_DATA_TYPE_TFLOAT32      11
#define CCCL_CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ  12
#define CCCL_CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B 13
#define CCCL_CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B  14
#define CCCL_CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B 15

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
using cuDeviceGetDefaultMemPool_t =
  ::cuda::__driver::CUresult (*)(::cuda::__driver::CUmemoryPool*, ::cuda::__driver::CUdevice);

// Primary context
using cuDevicePrimaryCtxRetain_t =
  ::cuda::__driver::CUresult (*)(::cuda::__driver::CUcontext, ::cuda::__driver::CUdevice);
using cuDevicePrimaryCtxRelease_t  = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUdevice);
using cuDevicePrimaryCtxGetState_t = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUdevice, unsigned int*, int*);

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
using cuMemPoolTrimTo_t    = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUmemoryPool, ::cuda::std::size_t);
using cuMemPoolSetAccess_t = ::cuda::__driver::CUresult (*)(
  ::cuda::__driver::CUmemoryPool, ::cuda::__driver::CUmemAccessDesc*, ::cuda::std::size_t);
using cuMemPoolGetAccess_t = ::cuda::__driver::CUresult (*)(
  ::cuda::__driver::CUmemAccess_flags*, ::cuda::__driver::CUmemoryPool, ::cuda::__driver::CUmemLocation*);
using cuMemGetDefaultMemPool_t = ::cuda::__driver::CUresult (*)(
  ::cuda::__driver::CUmemoryPool*, ::cuda::__driver::CUmemLocation*, ::cuda::__driver::CUmemAllocationType_enum);

using cuMemFreeAsync_t = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUdeviceptr, ::cuda::__driver::CUstream);
using cuMemAllocManaged_t =
  ::cuda::__driver::CUresult (*)(::cuda::__driver::CUdeviceptr*, ::cuda::std::size_t, unsigned int);
using cuMemAllocHost_t = ::cuda::__driver::CUresult (*)(void**, ::cuda::std::size_t);
using cuMemFree_t      = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUdeviceptr);
using cuMemFreeHost_t  = ::cuda::__driver::CUresult (*)(void*);

// Unified Addressing
using cuPointerGetAttribute_t =
  ::cuda::__driver::CUresult (*)(void*, ::cuda::__driver::CUpointer_attribute, ::cuda::__driver::CUdeviceptr);
using cuPointerGetAttributes_t = ::cuda::__driver::CUresult (*)(
  unsigned int, ::cuda::__driver::CUpointer_attribute*, void**, ::cuda::__driver::CUdeviceptr);

// Stream management
using cuStreamAddCallback_t =
  ::cuda::__driver::CUresult (*)(::cuda::__driver::CUstream, ::cuda::__driver::CUstreamCallback, void*, unsigned int);
using cuStreamCreateWithPriority_t = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUstream*, unsigned int, int);
using cuStreamSynchronize_t        = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUstream);
using cuStreamGetCtx_t    = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUstream, ::cuda::__driver::CUcontext*);
using cuStreamGetCtx_v2_t = ::cuda::__driver::CUresult (*)(
  ::cuda::__driver::CUstream, ::cuda::__driver::CUcontext*, ::cuda::__driver::CUgreenCtx*);
using cuStreamGetDevice_t = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUstream, ::cuda::__driver::CUdevice*);
using cuStreamWaitEvent_t =
  ::cuda::__driver::CUresult (*)(::cuda::__driver::CUstream, ::cuda::__driver::CUevent, unsigned int);
using cuStreamQuery_t       = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUstream);
using cuStreamGetPriority_t = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUstream, int*);
using cuStreamGetId_t =
  ::cuda::__driver::CUresult (*)(::cuda::__driver::CUstream, unsigned long long*); // NOLINT(google-runtime-int)
using cuStreamDestroy_t = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUstream);

// Event management
using cuEventCreate_t  = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUevent*, unsigned int);
using cuEventDestroy_t = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUevent);
using cuEventElapsedTime_t =
  ::cuda::__driver::CUresult (*)(float*, ::cuda::__driver::CUevent, ::cuda::__driver::CUevent);
using cuEventQuery_t       = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUevent);
using cuEventRecord_t      = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUevent, ::cuda::__driver::CUstream);
using cuEventSynchronize_t = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUevent);

// Library management
using cuKernelGetFunction_t = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUfunction*, ::cuda::__driver::CUkernel);
using cuKernelGetAttribute_t = ::cuda::__driver::CUresult (*)(
  int*, ::cuda::__driver::CUfunction_attribute, ::cuda::__driver::CUkernel, ::cuda::__driver::CUdevice);
using cuKernelGetName_t   = ::cuda::__driver::CUresult (*)(const char**, ::cuda::__driver::CUkernel);
using cuLibraryLoadData_t = ::cuda::__driver::CUresult (*)(
  ::cuda::__driver::CUlibrary*,
  const void*,
  ::cuda::__driver::CUjit_option*,
  void**,
  unsigned int,
  ::cuda::__driver::CUlibraryOption*,
  void**,
  unsigned int);
using cuLibraryGetKernel_t =
  ::cuda::__driver::CUresult (*)(::cuda::__driver::CUkernel*, ::cuda::__driver::CUlibrary, const char*);
using cuLibraryUnload_t    = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUlibrary);
using cuKernelGetLibrary_t = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUlibrary*, ::cuda::__driver::CUkernel);
using cuLibraryGetGlobal_t = ::cuda::__driver::CUresult (*)(
  ::cuda::__driver::CUdeviceptr*, ::cuda::std::size_t*, ::cuda::__driver::CUlibrary, const char*);
using cuLibraryGetManaged_t = ::cuda::__driver::CUresult (*)(
  ::cuda::__driver::CUdeviceptr*, ::cuda::std::size_t*, ::cuda::__driver::CUlibrary, const char*);

// Execution control
using cuFuncGetAttribute_t =
  ::cuda::__driver::CUresult (*)(int*, ::cuda::__driver::CUfunction_attribute, ::cuda::__driver::CUfunction);
using cuFuncSetAttribute_t =
  ::cuda::__driver::CUresult (*)(::cuda::__driver::CUfunction, ::cuda::__driver::CUfunction_attribute, int);
using cuLaunchHostFunc_t = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUstream, ::cuda::__driver::CUhostFn, void*);
using cuLaunchKernelEx_t =
  ::cuda::__driver::CUresult (*)(const ::cuda::__driver::CUlaunchConfig*, ::cuda::__driver::CUfunction, void**, void**);

// Graph management
using cuGraphAddKernelNode_t = ::cuda::__driver::CUresult (*)(
  ::cuda::__driver::CUgraphNode*,
  ::cuda::__driver::CUgraph,
  const ::cuda::__driver::CUgraphNode*,
  ::cuda::std::size_t,
  const ::cuda::__driver::CUDA_KERNEL_NODE_PARAMS*);
using cuGraphKernelNodeSetAttribute_t = ::cuda::__driver::CUresult (*)(
  ::cuda::__driver::CUgraphNode, ::cuda::__driver::CUkernelNodeAttrID, const ::cuda::__driver::CUkernelNodeAttrValue*);

// Peer Context Memory Access
using cuDeviceCanAccessPeer_t =
  ::cuda::__driver::CUresult (*)(int*, ::cuda::__driver::CUdevice, ::cuda::__driver::CUdevice);

// Green contexts
using cuGreenCtxCreate_t = ::cuda::__driver::CUresult (*)(
  ::cuda::__driver::CUgreenCtx*, ::cuda::__driver::CUdevResourceDesc, ::cuda::__driver::CUdevice, unsigned int);
using cuGreenCtxDestroy_t = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUgreenCtx);
using cuCtxFromGreenCtx_t = ::cuda::__driver::CUresult (*)(::cuda::__driver::CUcontext*, ::cuda::__driver::CUgreenCtx);
using cuGreenCtxGetId_t =
  ::cuda::__driver::CUresult (*)(::cuda::__driver::CUgreenCtx, unsigned long long*); // NOLINT(google-runtime-int)

// Tensor map
using cuTensorMapEncodeTiled_t = ::cuda::__driver::CUresult (*)(
  ::cuda::__driver::CUtensorMap*,
  ::cuda::__driver::CUtensorMapDataType,
  ::cuda::std::uint32_t,
  void*,
  const ::cuda::std::uint64_t*,
  const ::cuda::std::uint64_t*,
  const ::cuda::std::uint32_t*,
  const ::cuda::std::uint32_t*,
  ::cuda::__driver::CUtensorMapInterleave,
  ::cuda::__driver::CUtensorMapSwizzle,
  ::cuda::__driver::CUtensorMapL2promotion,
  ::cuda::__driver::CUtensorMapFloatOOBfill);
} // namespace api

_CCCL_END_NAMESPACE_CUDA_DRIVER

#endif // _CUDA___DRIVER_DRIVER_API_TYPES_H
