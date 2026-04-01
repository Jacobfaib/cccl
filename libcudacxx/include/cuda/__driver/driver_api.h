//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___DRIVER_DRIVER_API_H
#define _CUDA___DRIVER_DRIVER_API_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/__driver/driver_api_types.h>
#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__host_stdlib/stdexcept>
#  include <cuda/std/__internal/namespaces.h>
#  include <cuda/std/__limits/numeric_limits.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/conditional.h>
#  include <cuda/std/__type_traits/enable_if.h>
#  include <cuda/std/__type_traits/integral_constant.h>
#  include <cuda/std/__type_traits/is_enum.h>
#  include <cuda/std/__type_traits/is_same.h>
#  if _CCCL_OS(WINDOWS)
#    include <windows.h>
#  else
#    include <dlfcn.h>
#  endif

#  if __has_include(<cuda.h>)
#    include <cuda.h>

#    define _CCCLRT_CHECK_ABI_COMPATIBLE(cu_function, our_function)                                \
      static_assert(::cuda::__driver::abi_detail::check_abi_compatible(cu_function, our_function), \
                    #cu_function " and " #our_function " are not ABI compatible")
#  else
#    define _CCCLRT_CHECK_ABI_COMPATIBLE(cu_function, our_function) static_assert(true)
#  endif

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_DRIVER

namespace abi_detail
{
template <typename T1, typename T2, typename = void>
struct abi_compatible : ::cuda::std::false_type
{}; // NOLINT(readability-identifier-naming)

// Two types that are the same are always ABI compatible
template <typename T>
struct abi_compatible<T, T> : ::cuda::std::true_type
{};

// Specialization when either T1 or T2 is an enum. In this case, all that matters is that both
// are the same size
template <typename T1, typename T2>
struct abi_compatible<T1, T2, ::cuda::std::enable_if_t<::cuda::std::is_enum_v<T1> || ::cuda::std::is_enum_v<T2>>>
    : ::cuda::std::conditional_t<sizeof(T1) == sizeof(T2), ::cuda::std::true_type, ::cuda::std::false_type>
{};

// The enable_if_t is needed because without it
//
// abi_compatible<Type **, Type **>
//
// Is an ambiguous overload. It is either abi_compatible<T, T> (with T = Type **), or
// abi_compatible<T1*, T2*> (where T1 = T2 = Type *). We still need to have this overload in that
// case, because we still want to catch instances where we spoof enums with their underlying values:
//
// enum TheRealType { ... }; (underlying type is int)
// using OurType : int;
//
// In this case, a function taking OurType, or OurType * is ABI compatible with the real enum,
// but won't necessarily have the exact same pointed-to type.
template <typename T1, typename T2>
struct abi_compatible<T1*, T2*, ::cuda::std::enable_if_t<!::cuda::std::is_same_v<T1, T2>>> : abi_compatible<T1, T2>
{};

template <typename T1, typename T2>
inline constexpr bool abi_compatible_v = abi_compatible<T1, T2>::value;

// ==========================================================================================

template <typename R1, typename... Args1, typename R2, typename... Args2>
[[nodiscard]] constexpr bool check_abi_compatible(R1 (*)(Args1...), R2 (*)(Args2...))
{
  if constexpr (abi_compatible_v<R1, R2> && (sizeof...(Args1) == sizeof...(Args2)))
  {
    return std::conjunction_v<abi_compatible<Args1, Args2>...>;
  }
  return false;
}
} // namespace abi_detail

// Get the driver function by name using this macro
#  define _CCCLRT_GET_DRIVER_FUNCTION(function_name)               \
    reinterpret_cast<::cuda::__driver::api::function_name##_t>(    \
      ::cuda::__driver::__get_driver_entry_point(#function_name)); \
    _CCCLRT_CHECK_ABI_COMPATIBLE(function_name, ::cuda::__driver::api::function_name##_t{})

#  define _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(function_name, versioned_fn_name, major, minor) \
    reinterpret_cast<::cuda::__driver::api::versioned_fn_name##_t>(                             \
      ::cuda::__driver::__get_driver_entry_point(#function_name, major, minor));                \
    _CCCLRT_CHECK_ABI_COMPATIBLE(versioned_fn_name, ::cuda::__driver::api::function_name##_t{})

// cudaGetDriverEntryPoint function is deprecated
_CCCL_SUPPRESS_DEPRECATED_PUSH

//! @brief Gets the cuGetProcAddress function pointer.
[[nodiscard]] _CCCL_PUBLIC_HOST_API inline api::cuGetProcAddress_v2_t __getProcAddressFn()
{
  const char* __fn_name = "cuGetProcAddress_v2";
#  if _CCCL_OS(WINDOWS)
  static auto __driver_library = ::LoadLibraryExA("nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (__driver_library == nullptr)
  {
    _CCCL_THROW(::cuda::cuda_error, CCCL_CUDA_ERROR_UNKNOWN, "Failed to load nvcuda.dll");
  }
  static void* __fn = ::GetProcAddress(__driver_library, __fn_name);
  if (__fn == nullptr)
  {
    _CCCL_THROW(::cuda::cuda_error, CCCL_CUDA_ERROR_NOT_INITIALIZED, "Failed to get cuGetProcAddress from nvcuda.dll");
  }
#  else // ^^^ _CCCL_OS(WINDOWS) ^^^ / vvv !_CCCL_OS(WINDOWS) vvv
#    if _CCCL_OS(ANDROID)
  const char* __driver_library_name = "libcuda.so";
#    else // ^^^ _CCCL_OS(ANDROID) ^^^ / vvv !_CCCL_OS(ANDROID) vvv
  const char* __driver_library_name = "libcuda.so.1";
#    endif // ^^^ !_CCCL_OS(ANDROID) ^^^
  static void* __driver_library = ::dlopen(__driver_library_name, RTLD_NOW);
  if (__driver_library == nullptr)
  {
    _CCCL_THROW(::cuda::cuda_error, CCCL_CUDA_ERROR_UNKNOWN, "Failed to load libcuda.so.1");
  }
  static void* __fn = ::dlsym(__driver_library, __fn_name);
  if (__fn == nullptr)
  {
    _CCCL_THROW(::cuda::cuda_error, CCCL_CUDA_ERROR_NOT_INITIALIZED, "Failed to get cuGetProcAddress from libcuda.so.1");
  }
#  endif // ^^^ !_CCCL_OS(WINDOWS) ^^^
  return reinterpret_cast<api::cuGetProcAddress_v2_t>(__fn);
}

_CCCL_SUPPRESS_DEPRECATED_POP

//! @brief Makes the driver version from major and minor version.
[[nodiscard]] _CCCL_HOST_API constexpr int __make_version(int __major, int __minor) noexcept
{
  _CCCL_ASSERT(__major >= 2, "invalid major CUDA Driver version");
  _CCCL_ASSERT(__minor >= 0 && __minor < 100, "invalid minor CUDA Driver version");
  return __major * 1000 + __minor * 10;
}

//! @brief Gets the driver entry point.
//!
//! @param __get_proc_addr_fn Pointer to cuGetProcAddress function.
//! @param __name Name of the symbol to get the driver entry point for.
//! @param __major The major CTK version to get the symbol version for.
//! @param __minor The major CTK version to get the symbol version for.
//!
//! @return The address of the symbol.
//!
//! @throws @c cuda::cuda_error if the symbol cannot be obtained.
[[nodiscard]] _CCCL_HOST_API inline void*
__get_driver_entry_point_impl(api::GetProcAddress_V2 __get_proc_addr_fn, const char* __name, int __major, int __minor)
{
  void* __fn;
  ::cuda::__driver::CUdriverProcAddressQueryResult __result;
  ::cuda::__driver::CUresult __status = __get_proc_addr_fn(
    __name, &__fn, ::cuda::__driver::__make_version(__major, __minor), CCCL_CU_GET_PROC_ADDRESS_DEFAULT, &__result);
  if (__status != CCCL_CUDA_SUCCESS || __result != CCCL_CU_GET_PROC_ADDRESS_SUCCESS)
  {
    if (__status == CCCL_CUDA_ERROR_INVALID_VALUE)
    {
      _CCCL_THROW(
        ::cuda::cuda_error, CCCL_CUDA_ERROR_INVALID_VALUE, "Driver version is too low to use this API", __name);
    }
    if (__result == CCCL_CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT)
    {
      _CCCL_THROW(::cuda::cuda_error, CCCL_CUDA_ERROR_NOT_SUPPORTED, "Driver does not support this API", __name);
    }
    else
    {
      _CCCL_THROW(::cuda::cuda_error, CCCL_CUDA_ERROR_UNKNOWN, "Failed to access driver API", __name);
    }
  }
  return __fn;
}

//! @brief CUDA Driver API call wrapper. Calls a given CUDA Driver API and checks the return value.
//!
//! @param __fn A CUDA Driver function.
//! @param __err_msg Error message describing the call if the all fails.
//! @param __args The arguments to the @c __fn call.
//!
//! @throws @c cuda::cuda_error if the function call doesn't return CUDA_SUCCESS.
template <typename Fn, typename... Args>
_CCCL_HOST_API inline void __call_driver_fn(Fn __fn, const char* __err_msg, Args... __args)
{
  ::cuda::__driver::CUresult __status = __fn(__args...);
  if (__status != CCCL_CUDA_SUCCESS)
  {
    _CCCL_THROW(::cuda::cuda_error, __status, __err_msg);
  }
}

//! @brief Initializes the CUDA Driver.
//!
//! @param __get_proc_addr_fn The pointer to cuGetProcAddress function.
//!
//! @return A dummy bool value.
//!
//! @warning This function should be called only once from __get_driver_entry_point function.
[[nodiscard]] _CCCL_HOST_API inline bool __init(api::cuGetProcAddress_v2_t __get_proc_addr_fn)
{
  auto __driver_fn = reinterpret_cast<::cuda::__driver::cuInit_t>(
    ::cuda::__driver::__get_driver_entry_point_impl(__get_proc_addr_fn, "cuInit", 12, 0));
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to initialize CUDA Driver", 0);
  return true;
}

//! @brief Get a driver function pointer for a given API name and optionally specific CUDA version. This function also
//!        initializes the CUDA Driver.
//!
//! @param __name Name of the symbol to get the driver entry point for.
//! @param __major The major CTK version to get the symbol version for. Defaults to 12.
//! @param __minor The major CTK version to get the symbol version for. Defaults to 0.
//!
//! @return The address of the symbol.
//!
//! @throws @c cuda::cuda_error if the symbol cannot be obtained or the CUDA driver failed to initialize.
[[nodiscard]] _CCCL_PUBLIC_HOST_API inline void*
__get_driver_entry_point(const char* __name, [[maybe_unused]] int __major = 12, [[maybe_unused]] int __minor = 0)
{
  // Get cuGetProcAddress function and call cuInit(0) only on the first call
  static auto __get_proc_addr_fn      = ::cuda::__driver::__getProcAddressFn();
  [[maybe_unused]] static auto __init = ::cuda::__driver::__init(__get_proc_addr_fn);
  return ::cuda::__driver::__get_driver_entry_point_impl(__get_proc_addr_fn, __name, __major, __minor);
}

//! @brief Converts CUdevice to ordinal device id.
//!
//! @note Currently, CUdevice value is the same as the ordinal device id. But that might change in the future.
[[nodiscard]] _CCCL_HOST_API inline int __cudevice_to_ordinal(::cuda::__driver::CUdevice __dev) noexcept
{
  return static_cast<int>(__dev);
}

// Version management

[[nodiscard]] _CCCL_HOST_API inline int __getVersion()
{
  static int __version = []() {
    int __v;
    auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDriverGetVersion);
    ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to check CUDA driver version", &__v);
    return __v;
  }();
  return __version;
}

[[nodiscard]] _CCCL_HOST_API inline bool __version_at_least(int __major, int __minor)
{
  return ::cuda::__driver::__getVersion() >= ::cuda::__driver::__make_version(__major, __minor);
}

[[nodiscard]] _CCCL_HOST_API inline bool __version_below(int __major, int __minor)
{
  return ::cuda::__driver::__getVersion() < ::cuda::__driver::__make_version(__major, __minor);
}

// Device management

[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUdevice __deviceGet(int __ordinal)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDeviceGet);
  ::cuda::__driver::CUdevice __result;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get device", &__result, __ordinal);
  return __result;
}

[[nodiscard]] _CCCL_HOST_API inline int
__deviceGetAttribute(::cuda::__driver::CUdevice_attribute __attr, ::cuda::__driver::CUdevice __device)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDeviceGetAttribute);
  int __result;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get device attribute", &__result, __attr, __device);
  return __result;
}

[[nodiscard]] _CCCL_HOST_API inline int __deviceGetCount()
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDeviceGetCount);
  int __result;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get device count", &__result);
  return __result;
}

_CCCL_HOST_API inline void __deviceGetName(char* __name_out, int __len, int __ordinal)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDeviceGetName);

  // TODO CUdevice is just an int, we probably could just cast, but for now do the safe thing
  ::cuda::__driver::CUdevice __dev = __deviceGet(__ordinal);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to query the name of a device", __name_out, __len, __dev);
}

// Primary context management

[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUcontext __primaryCtxRetain(::cuda::__driver::CUdevice __dev)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDevicePrimaryCtxRetain);
  ::cuda::__driver::CUcontext __result;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to retain context for a device", &__result, __dev);
  return __result;
}

[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUresult
__primaryCtxReleaseNoThrow(::cuda::__driver::CUdevice __dev)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDevicePrimaryCtxRelease);
  return __driver_fn(__dev);
}

[[nodiscard]] _CCCL_HOST_API inline bool __isPrimaryCtxActive(::cuda::__driver::CUdevice __dev)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDevicePrimaryCtxGetState);
  int __result;
  unsigned int __dummy;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to check the primary ctx state", __dev, &__dummy, &__result);
  return __result == 1;
}

// Context management

_CCCL_HOST_API inline void __ctxPush(::cuda::__driver::CUcontext __ctx)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuCtxPushCurrent);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to push context", __ctx);
}

_CCCL_HOST_API inline ::cuda::__driver::CUcontext __ctxPop()
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuCtxPopCurrent);
  ::cuda::__driver::CUcontext __result;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to pop context", &__result);
  return __result;
}

[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUcontext __ctxGetCurrent()
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuCtxGetCurrent);
  ::cuda::__driver::CUcontext __result;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get current context", &__result);
  return __result;
}

[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUdevice __ctxGetDevice()
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuCtxGetDevice);
  ::cuda::__driver::CUdevice __result{};
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get current context", &__result);
  return __result;
}

// Memory management

_CCCL_HOST_API inline void
__memcpyAsync(void* __dst, const void* __src, ::cuda::std::size_t __count, ::cuda::__driver::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemcpyAsync);
  ::cuda::__driver::__call_driver_fn(
    __driver_fn,
    "Failed to perform a memcpy",
    reinterpret_cast<::cuda::__driver::CUdeviceptr>(__dst),
    reinterpret_cast<::cuda::__driver::CUdeviceptr>(__src),
    __count,
    __stream);
}

#  if _CCCL_CTK_AT_LEAST(13, 0)
_CCCL_HOST_API inline void __memcpyAsyncWithAttributes(
  void* __dst,
  const void* __src,
  ::cuda::std::size_t __count,
  ::cuda::__driver::CUstream __stream,
  ::cuda::__driver::CUmemcpyAttributes __attributes)
{
  static auto __driver_fn    = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuMemcpyBatchAsync, cuMemcpyBatchAsync, 13, 0);
  ::cuda::std::size_t __zero = 0;
  ::cuda::__driver::__call_driver_fn(
    __driver_fn,
    "Failed to perform a memcpy with attributes",
    reinterpret_cast<::cuda::__driver::CUdeviceptr*>(&__dst),
    reinterpret_cast<::cuda::__driver::CUdeviceptr*>(&__src),
    &__count,
    1,
    &__attributes,
    &__zero,
    1,
    __stream);
}

_CCCL_HOST_API inline void __memcpyBatchAsync(
  void** __dsts,
  const void** __srcs,
  const ::cuda::std::size_t* __sizes,
  ::cuda::std::size_t __count,
  ::cuda::__driver::CUmemcpyAttributes* __attributes,
  ::cuda::std::size_t* __attribute_indices,
  ::cuda::std::size_t __num_attributes,
  ::cuda::__driver::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuMemcpyBatchAsync, cuMemcpyBatchAsync, 13, 0);
  ::cuda::__driver::__call_driver_fn(
    __driver_fn,
    "Failed to perform a memcpy with attributes",
    reinterpret_cast<::cuda::__driver::CUdeviceptr*>(__dsts),
    reinterpret_cast<::cuda::__driver::CUdeviceptr*>(__srcs),
    const_cast<::cuda::std::size_t*>(__sizes),
    __count,
    __attributes,
    __attribute_indices,
    __num_attributes,
    __stream);
}

#  endif // _CCCL_CTK_AT_LEAST(13, 0)

template <typename _Tp>
_CCCL_HOST_API void
__memsetAsync(void* __dst, _Tp __value, ::cuda::std::size_t __count, ::cuda::__driver::CUstream __stream)
{
  if constexpr (sizeof(_Tp) == 1)
  {
    static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemsetD8Async);
    ::cuda::__driver::__call_driver_fn(
      __driver_fn,
      "Failed to perform a memset",
      reinterpret_cast<::cuda::__driver::CUdeviceptr>(__dst),
      __value,
      __count,
      __stream);
  }
  else if constexpr (sizeof(_Tp) == 2)
  {
    static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemsetD16Async);
    ::cuda::__driver::__call_driver_fn(
      __driver_fn,
      "Failed to perform a memset",
      reinterpret_cast<::cuda::__driver::CUdeviceptr>(__dst),
      __value,
      __count,
      __stream);
  }
  else if constexpr (sizeof(_Tp) == 4)
  {
    static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemsetD32Async);
    ::cuda::__driver::__call_driver_fn(
      __driver_fn,
      "Failed to perform a memset",
      reinterpret_cast<::cuda::__driver::CUdeviceptr>(__dst),
      __value,
      __count,
      __stream);
  }
  else
  {
    static_assert(::cuda::std::__always_false_v<_Tp>, "Unsupported type for memset");
  }
}

[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUresult
__mempoolCreateNoThrow(::cuda::__driver::CUmemoryPool* __pool, ::cuda::__driver::CUmemPoolProps* __props)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemPoolCreate);
  return __driver_fn(__pool, __props);
}

_CCCL_HOST_API inline void __mempoolSetAttribute(
  ::cuda::__driver::CUmemoryPool __pool, ::cuda::__driver::CUmemPool_attribute __attr, void* __value)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemPoolSetAttribute);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to set attribute for a memory pool", __pool, __attr, __value);
}

_CCCL_HOST_API inline ::cuda::std::size_t
__mempoolGetAttribute(::cuda::__driver::CUmemoryPool __pool, ::cuda::__driver::CUmemPool_attribute __attr)
{
  ::cuda::std::size_t __value = 0;
  static auto __driver_fn     = _CCCLRT_GET_DRIVER_FUNCTION(cuMemPoolGetAttribute);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get attribute for a memory pool", __pool, __attr, &__value);
  return __value;
}

_CCCL_HOST_API inline void __mempoolDestroy(::cuda::__driver::CUmemoryPool __pool)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemPoolDestroy);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to destroy a memory pool", __pool);
}

_CCCL_HOST_API inline ::cuda::__driver::CUdeviceptr __mallocFromPoolAsync(
  ::cuda::std::size_t __bytes, ::cuda::__driver::CUmemoryPool __pool, ::cuda::__driver::CUstream __stream)
{
  static auto __driver_fn                = _CCCLRT_GET_DRIVER_FUNCTION(cuMemAllocFromPoolAsync);
  ::cuda::__driver::CUdeviceptr __result = 0;
  ::cuda::__driver::__call_driver_fn(
    __driver_fn, "Failed to allocate memory from a memory pool", &__result, __bytes, __pool, __stream);
  return __result;
}

_CCCL_HOST_API inline void
__mempoolTrimTo(::cuda::__driver::CUmemoryPool __pool, ::cuda::std::size_t __min_bytes_to_keep)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemPoolTrimTo);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to trim a memory pool", __pool, __min_bytes_to_keep);
}

_CCCL_HOST_API inline ::cuda::__driver::CUresult
__freeAsyncNoThrow(::cuda::__driver::CUdeviceptr __dptr, ::cuda::__driver::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemFreeAsync);
  return __driver_fn(__dptr, __stream);
}

_CCCL_HOST_API inline void __mempoolSetAccess(
  ::cuda::__driver::CUmemoryPool __pool, ::cuda::__driver::CUmemAccessDesc* __descs, ::cuda::std::size_t __count)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemPoolSetAccess);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to set access of a memory pool", __pool, __descs, __count);
}

_CCCL_HOST_API inline ::cuda::__driver::CUmemAccess_flags
__mempoolGetAccess(::cuda::__driver::CUmemoryPool __pool, ::cuda::__driver::CUmemLocation* __location)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemPoolGetAccess);
  ::cuda::__driver::CUmemAccess_flags __flags;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get access of a memory pool", &__flags, __pool, __location);
  return __flags;
}

_CCCL_HOST_API inline ::cuda::__driver::CUresult __mempoolGetAccessNoThrow(
  ::cuda::__driver::CUmemAccess_flags& __flags,
  ::cuda::__driver::CUmemoryPool __pool,
  ::cuda::__driver::CUmemLocation* __location) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemPoolGetAccess);
  return __driver_fn(&__flags, __pool, __location);
}

#  if _CCCL_CTK_AT_LEAST(13, 0)
_CCCL_HOST_API inline ::cuda::__driver::CUmemoryPool __getDefaultMemPool(
  ::cuda::__driver::CUmemLocation __location, ::cuda::__driver::CUmemAllocationType_enum __allocation_type)
{
  static auto __driver_fn =
    _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuMemGetDefaultMemPool, cuMemGetDefaultMemPool, 13, 0);
  ::cuda::__driver::CUmemoryPool __result = nullptr;
  ::cuda::__driver::__call_driver_fn(
    __driver_fn, "Failed to get default memory pool", &__result, &__location, __allocation_type);
  return __result;
}
#  else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
_CCCL_HOST_API inline ::cuda::__driver::CUmemoryPool __deviceGetDefaultMemPool(::cuda::__driver::CUdevice __device)
{
  static auto __driver_fn                 = _CCCLRT_GET_DRIVER_FUNCTION(cuDeviceGetDefaultMemPool);
  ::cuda::__driver::CUmemoryPool __result = nullptr;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get default memory pool", &__result, __device);
  return __result;
}
#  endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^

_CCCL_HOST_API inline ::cuda::__driver::CUdeviceptr __mallocManaged(::cuda::std::size_t __bytes, unsigned int __flags)
{
  static auto __driver_fn                = _CCCLRT_GET_DRIVER_FUNCTION(cuMemAllocManaged);
  ::cuda::__driver::CUdeviceptr __result = 0;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to allocate managed memory", &__result, __bytes, __flags);
  return __result;
}

_CCCL_HOST_API inline void* __mallocHost(::cuda::std::size_t __bytes)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemAllocHost);
  void* __result          = nullptr;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to allocate host memory", &__result, __bytes);
  return __result;
}

_CCCL_HOST_API inline ::cuda::__driver::CUresult __freeNoThrow(::cuda::__driver::CUdeviceptr __dptr)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemFree);
  return __driver_fn(__dptr);
}

_CCCL_HOST_API inline ::cuda::__driver::CUresult __freeHostNoThrow(void* __dptr)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemFreeHost);
  return __driver_fn(__dptr);
}

// Unified Addressing

// TODO: we don't want to have these functions here, refactoring expected
template <::cuda::__driver::CUpointer_attribute _Attr>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto __pointer_attribute_value_type_t_impl() noexcept
{
  if constexpr (_Attr == CCCL_CU_POINTER_ATTRIBUTE_CONTEXT)
  {
    return ::cuda::__driver::CUcontext{};
  }
  else if constexpr (_Attr == CCCL_CU_POINTER_ATTRIBUTE_MEMORY_TYPE)
  {
    return ::cuda::__driver::CUmemorytype{};
  }
  else if constexpr (_Attr == CCCL_CU_POINTER_ATTRIBUTE_DEVICE_POINTER
                     || _Attr == CCCL_CU_POINTER_ATTRIBUTE_HOST_POINTER)
  {
    return static_cast<void*>(nullptr);
  }
  else if constexpr (_Attr == CCCL_CU_POINTER_ATTRIBUTE_IS_MANAGED || _Attr == CCCL_CU_POINTER_ATTRIBUTE_MAPPED)
  {
    return bool{};
  }
  else if constexpr (_Attr == CCCL_CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL)
  {
    return int{};
  }
  else if constexpr (_Attr == CCCL_CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE)
  {
    return ::cuda::__driver::CUmemoryPool{};
  }
  else
  {
    static_assert(::cuda::std::__always_false_v<decltype(_Attr)>, "not implemented attribute");
  }
}

template <::cuda::__driver::CUpointer_attribute _Attr>
using __pointer_attribute_value_type_t = decltype(::cuda::__driver::__pointer_attribute_value_type_t_impl<_Attr>());

template <::cuda::__driver::CUpointer_attribute _Attr>
[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUresult
__pointerGetAttributeNoThrow(__pointer_attribute_value_type_t<_Attr>& __result, const void* __ptr)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuPointerGetAttribute);
  ::cuda::__driver::CUresult __status{};
  if constexpr (::cuda::std::is_same_v<__pointer_attribute_value_type_t<_Attr>, bool>)
  {
    int __result2{};
    __status = __driver_fn(&__result2, _Attr, reinterpret_cast<::cuda::__driver::CUdeviceptr>(__ptr));
    __result = static_cast<bool>(__result2);
  }
  else
  {
    __status = __driver_fn((void*) &__result, _Attr, reinterpret_cast<::cuda::__driver::CUdeviceptr>(__ptr));
  }
  return __status;
}

template <::cuda::__driver::CUpointer_attribute _Attr>
[[nodiscard]] _CCCL_HOST_API __pointer_attribute_value_type_t<_Attr> __pointerGetAttribute(const void* __ptr)
{
  __pointer_attribute_value_type_t<_Attr> __result;
  _CCCL_TRY_CUDA_API(
    ::cuda::__driver::__pointerGetAttributeNoThrow<_Attr>, "Failed to get attribute of a pointer", __result, __ptr);
  return __result;
}

template <::cuda::std::size_t _Np>
[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUresult __pointerGetAttributesNoThrow(
  ::cuda::__driver::CUpointer_attribute (&__attrs)[_Np], void* (&__results)[_Np], const void* __ptr)
{
  static const auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuPointerGetAttributes);
  return __driver_fn(
    static_cast<unsigned>(_Np), __attrs, __results, reinterpret_cast<::cuda::__driver::CUdeviceptr>(__ptr));
}

// Stream management

_CCCL_HOST_API inline void __streamAddCallback(
  ::cuda::__driver::CUstream __stream, ::cuda::__driver::CUstreamCallback __cb, void* __data, unsigned __flags = 0)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamAddCallback);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to add a stream callback", __stream, __cb, __data, __flags);
}

[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUstream
__streamCreateWithPriority(unsigned __flags, int __priority)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamCreateWithPriority);
  ::cuda::__driver::CUstream __stream;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to create a stream", &__stream, __flags, __priority);
  return __stream;
}

_CCCL_HOST_API inline ::cuda::__driver::CUresult __streamSynchronizeNoThrow(::cuda::__driver::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamSynchronize);
  return __driver_fn(__stream);
}

_CCCL_HOST_API inline void __streamSynchronize(::cuda::__driver::CUstream __stream)
{
  auto __status = __streamSynchronizeNoThrow(__stream);
  if (__status != CCCL_CUDA_SUCCESS)
  {
    _CCCL_THROW(::cuda::cuda_error, __status, "Failed to synchronize a stream");
  }
}

[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUcontext __streamGetCtx(::cuda::__driver::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamGetCtx);
  ::cuda::__driver::CUcontext __result;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get context from a stream", __stream, &__result);
  return __result;
}

#  if _CCCL_CTK_AT_LEAST(12, 5)
struct __ctx_from_stream
{
  enum class __kind
  {
    __device,
    __green
  };

  __kind __ctx_kind_;
  union
  {
    ::cuda::__driver::CUcontext __ctx_device_;
    ::cuda::__driver::CUgreenCtx __ctx_green_;
  };
};

[[nodiscard]] _CCCL_HOST_API inline __ctx_from_stream __streamGetCtx_v2(::cuda::__driver::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuStreamGetCtx, cuStreamGetCtx_v2, 12, 5);

  ::cuda::__driver::CUcontext __ctx   = nullptr;
  ::cuda::__driver::CUgreenCtx __gctx = nullptr;
  __ctx_from_stream __result;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get context from a stream", __stream, &__ctx, &__gctx);
  if (__gctx)
  {
    __result.__ctx_kind_  = __ctx_from_stream::__kind::__green;
    __result.__ctx_green_ = __gctx;
  }
  else
  {
    __result.__ctx_kind_   = __ctx_from_stream::__kind::__device;
    __result.__ctx_device_ = __ctx;
  }
  return __result;
}
#  endif // _CCCL_CTK_AT_LEAST(12, 5)

// TODO: make this available since CUDA 12.8
#  if _CCCL_CTK_AT_LEAST(13, 0)
[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUdevice __streamGetDevice(::cuda::__driver::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuStreamGetDevice, cuStreamGetDevice, 12, 8);
  ::cuda::__driver::CUdevice __result{};
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get the device of the stream", __stream, &__result);
  return __result;
}
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

_CCCL_HOST_API inline void __streamWaitEvent(::cuda::__driver::CUstream __stream, ::cuda::__driver::CUevent __evnt)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamWaitEvent);
  ::cuda::__driver::__call_driver_fn(
    __driver_fn, "Failed to make a stream wait for an event", __stream, __evnt, CCCL_CU_EVENT_WAIT_DEFAULT);
}

[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUresult __streamQueryNoThrow(::cuda::__driver::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamQuery);
  return __driver_fn(__stream);
}

[[nodiscard]] _CCCL_HOST_API inline int __streamGetPriority(::cuda::__driver::CUstream __stream)
{
  int __priority;
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamGetPriority);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get the priority of a stream", __stream, &__priority);
  return __priority;
}

[[nodiscard]] _CCCL_HOST_API inline unsigned long long __streamGetId(::cuda::__driver::CUstream __stream)
{
  unsigned long long __id;
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamGetId);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get the ID of a stream", __stream, &__id);
  return __id;
}

[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUresult
__streamDestroyNoThrow(::cuda::__driver::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamDestroy);
  return __driver_fn(__stream);
}

// Event management

[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUevent __eventCreate(unsigned __flags)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuEventCreate);
  ::cuda::__driver::CUevent __evnt;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to create a CUDA event", &__evnt, __flags);
  return __evnt;
}

[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUresult __eventDestroyNoThrow(::cuda::__driver::CUevent __evnt)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuEventDestroy);
  return __driver_fn(__evnt);
}

[[nodiscard]] _CCCL_HOST_API inline float
__eventElapsedTime(::cuda::__driver::CUevent __start, ::cuda::__driver::CUevent __end)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuEventElapsedTime);
  float __result;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get event elapsed time", &__result, __start, __end);
  return __result;
}

[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUresult __eventQueryNoThrow(::cuda::__driver::CUevent __evnt)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuEventQuery);
  return __driver_fn(__evnt);
}

_CCCL_HOST_API inline void __eventRecord(::cuda::__driver::CUevent __evnt, ::cuda::__driver::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuEventRecord);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to record an event", __evnt, __stream);
}

_CCCL_HOST_API inline void __eventSynchronize(::cuda::__driver::CUevent __evnt)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuEventSynchronize);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to synchronize an event", __evnt);
}

// Library management

[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUfunction __kernelGetFunction(::cuda::__driver::CUkernel __kernel)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuKernelGetFunction);
  ::cuda::__driver::CUfunction __result;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get kernel function", &__result, __kernel);
  return __result;
}

[[nodiscard]] _CCCL_HOST_API inline int __kernelGetAttribute(
  ::cuda::__driver::CUfunction_attribute __attr, ::cuda::__driver::CUkernel __kernel, ::cuda::__driver::CUdevice __dev)
{
  int __value;
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuKernelGetAttribute);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get kernel attribute", &__value, __attr, __kernel, __dev);
  return __value;
}

#  if _CCCL_CTK_AT_LEAST(12, 3)
[[nodiscard]] _CCCL_HOST_API inline const char* __kernelGetName(::cuda::__driver::CUkernel __kernel)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuKernelGetName, cuKernelGetName, 12, 3);
  const char* __name;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get kernel name", &__name, __kernel);
  return __name;
}
#  endif // _CCCL_CTK_AT_LEAST(12, 3)

[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUlibrary __libraryLoadData(
  const void* __code,
  ::cuda::__driver::CUjit_option* __jit_opts,
  void** __jit_opt_vals,
  unsigned __njit_opts,
  ::cuda::__driver::CUlibraryOption* __lib_opts,
  void** __lib_opt_vals,
  unsigned __nlib_opts)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuLibraryLoadData);
  ::cuda::__driver::CUlibrary __result;
  ::cuda::__driver::__call_driver_fn(
    __driver_fn,
    "Failed to load a library from data",
    &__result,
    __code,
    __jit_opts,
    __jit_opt_vals,
    __njit_opts,
    __lib_opts,
    __lib_opt_vals,
    __nlib_opts);
  return __result;
}

[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUkernel
__libraryGetKernel(::cuda::__driver::CUlibrary __library, const char* __name)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuLibraryGetKernel);
  ::cuda::__driver::CUkernel __result;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get kernel from library", &__result, __library, __name);
  return __result;
}

[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUresult
__libraryUnloadNoThrow(::cuda::__driver::CUlibrary __library)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuLibraryUnload);
  return __driver_fn(__library);
}

#  if _CCCL_CTK_AT_LEAST(12, 5)
[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUlibrary __kernelGetLibrary(::cuda::__driver::CUkernel __kernel)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuKernelGetLibrary, cuKernelGetLibrary, 12, 5);
  ::cuda::__driver::CUlibrary __lib;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get the library from kernel", &__lib, __kernel);
  return __lib;
}
#  endif // _CCCL_CTK_AT_LEAST(12, 5)

[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUresult
__libraryGetKernelNoThrow(::cuda::__driver::CUkernel& __kernel, ::cuda::__driver::CUlibrary __lib, const char* __name)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuLibraryGetKernel);
  return __driver_fn(&__kernel, __lib, __name);
}

[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUresult __libraryGetGlobalNoThrow(
  ::cuda::__driver::CUdeviceptr& __dptr,
  ::cuda::std::size_t& __nbytes,
  ::cuda::__driver::CUlibrary __lib,
  const char* __name)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuLibraryGetGlobal);
  return __driver_fn(&__dptr, &__nbytes, __lib, __name);
}

[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUresult __libraryGetManagedNoThrow(
  ::cuda::__driver::CUdeviceptr& __dptr,
  ::cuda::std::size_t& __nbytes,
  ::cuda::__driver::CUlibrary __lib,
  const char* __name)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuLibraryGetManaged);
  return __driver_fn(&__dptr, &__nbytes, __lib, __name);
}

// Execution control

[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUresult __functionGetAttributeNoThrow(
  int& __value, ::cuda::__driver::CUfunction_attribute __attr, ::cuda::__driver::CUfunction __kernel)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuFuncGetAttribute);
  return __driver_fn(&__value, __attr, __kernel);
}

[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUresult
__functionLoadNoThrow(::cuda::__driver::CUfunction __kernel) noexcept
{
  static auto __driver_fn = reinterpret_cast<::cuda::__driver::CUresult (*)(::cuda::__driver::CUfunction)>(
    ::cuda::__driver::__get_driver_entry_point("cuFuncLoad", 12, 4));
  return __driver_fn(__kernel);
}

[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUresult __functionSetAttributeNoThrow(
  ::cuda::__driver::CUfunction __kernel, ::cuda::__driver::CUfunction_attribute __attr, int __value)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuFuncSetAttribute);
  return __driver_fn(__kernel, __attr, __value);
}

_CCCL_HOST_API inline void
__launchHostFunc(::cuda::__driver::CUstream __stream, ::cuda::__driver::CUhostFn __fn, void* __data)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuLaunchHostFunc);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to launch host function", __stream, __fn, __data);
}

_CCCL_HOST_API inline void __launchKernel(
  ::cuda::__driver::CUlaunchConfig& __config,
  ::cuda::__driver::CUfunction __kernel,
  void* __args[],
  void* __extra[] = nullptr)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuLaunchKernelEx);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to launch kernel", &__config, __kernel, __args, __extra);
}

// Graph management

[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUgraphNode __graphAddKernelNode(
  ::cuda::__driver::CUgraph __graph,
  const ::cuda::__driver::CUgraphNode __deps[],
  ::cuda::std::size_t __ndeps,
  ::cuda::__driver::CUDA_KERNEL_NODE_PARAMS& __node_params)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuGraphAddKernelNode);
  ::cuda::__driver::CUgraphNode __result;
  ::cuda::__driver::__call_driver_fn(
    __driver_fn, "Failed to add a node to a graph", &__result, __graph, __deps, __ndeps, &__node_params);
  return __result;
}

_CCCL_HOST_API inline void __graphKernelNodeSetAttribute(
  ::cuda::__driver::CUgraphNode __node,
  ::cuda::__driver::CUkernelNodeAttrID __id,
  const ::cuda::__driver::CUkernelNodeAttrValue& __value)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuGraphKernelNodeSetAttribute);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to set kernel node parameters", __node, __id, &__value);
}

// Peer Context Memory Access

[[nodiscard]] _CCCL_HOST_API inline bool
__deviceCanAccessPeer(::cuda::__driver::CUdevice __dev, ::cuda::__driver::CUdevice __peer_dev)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDeviceCanAccessPeer);
  int __result;
  ::cuda::__driver::__call_driver_fn(
    __driver_fn, "Failed to query if device can access peer's memory", &__result, __dev, __peer_dev);
  return static_cast<bool>(__result);
}

[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUresult __deviceCanAccessPeerNoThrow(
  int& __result, ::cuda::__driver::CUdevice __dev, ::cuda::__driver::CUdevice __peer_dev) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDeviceCanAccessPeer);
  return __driver_fn(&__result, __dev, __peer_dev);
}

// Green contexts

#  if _CCCL_CTK_AT_LEAST(12, 5)
// Add actual resource description input once exposure is ready
[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUgreenCtx __greenCtxCreate(::cuda::__driver::CUdevice __dev)
{
  ::cuda::__driver::CUgreenCtx __result;
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuGreenCtxCreate, cuGreenCtxCreate, 12, 5);
  ::cuda::__driver::__call_driver_fn(
    __driver_fn, "Failed to create a green context", &__result, nullptr, __dev, CCCL_CU_GREEN_CTX_DEFAULT_STREAM);
  return __result;
}

[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUresult
__greenCtxDestroyNoThrow(::cuda::__driver::CUgreenCtx __green_ctx)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuGreenCtxDestroy, cuGreenCtxDestroy, 12, 5);
  return __driver_fn(__green_ctx);
}

[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUcontext
__ctxFromGreenCtx(::cuda::__driver::CUgreenCtx __green_ctx)
{
  ::cuda::__driver::CUcontext __result;
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuCtxFromGreenCtx, cuCtxFromGreenCtx, 12, 5);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to convert a green context", &__result, __green_ctx);
  return __result;
}
#  endif // _CCCL_CTK_AT_LEAST(12, 5)

#  if _CCCL_CTK_AT_LEAST(13, 0)
[[nodiscard]] _CCCL_HOST_API inline unsigned long long __greenCtxGetId(::cuda::__driver::CUgreenCtx __green_ctx)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuGreenCtxGetId, cuGreenCtxGetId, 13, 0);
  unsigned long long __id;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get the ID of a green context", __green_ctx, &__id);
  return __id;
}
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

[[nodiscard]] _CCCL_HOST_API inline ::cuda::std::size_t
__cutensormap_size_bytes(::cuda::std::size_t __num_items, ::cuda::__driver::CUtensorMapDataType __data_type)
{
  constexpr auto __max_size = ::cuda::std::numeric_limits<::cuda::std::size_t>::max();
  switch (__data_type)
  {
    case CCCL_CU_TENSOR_MAP_DATA_TYPE_UINT8:
#  if _CCCL_CTK_AT_LEAST(12, 8)
    case CCCL_CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B:
#  endif // _CCCL_CTK_AT_LEAST(12, 8)
      return __num_items;
    case CCCL_CU_TENSOR_MAP_DATA_TYPE_UINT16:
    case CCCL_CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:
    case CCCL_CU_TENSOR_MAP_DATA_TYPE_FLOAT16:
      if (__num_items > __max_size / 2)
      {
        _CCCL_THROW(::std::invalid_argument, "Number of items must be less than or equal to 2^64 / 2");
      }
      return __num_items * 2;
    case CCCL_CU_TENSOR_MAP_DATA_TYPE_INT32:
    case CCCL_CU_TENSOR_MAP_DATA_TYPE_UINT32:
    case CCCL_CU_TENSOR_MAP_DATA_TYPE_FLOAT32:
    case CCCL_CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ:
    case CCCL_CU_TENSOR_MAP_DATA_TYPE_TFLOAT32:
    case CCCL_CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ:
      if (__num_items > __max_size / 4)
      {
        _CCCL_THROW(::std::invalid_argument, "Number of items must be less than or equal to 2^64 / 4");
      }
      return __num_items * 4;
    case CCCL_CU_TENSOR_MAP_DATA_TYPE_INT64:
    case CCCL_CU_TENSOR_MAP_DATA_TYPE_UINT64:
    case CCCL_CU_TENSOR_MAP_DATA_TYPE_FLOAT64:
      if (__num_items > __max_size / 8)
      {
        _CCCL_THROW(::std::invalid_argument, "Number of items must be less than or equal to 2^64 / 8");
      }
      return __num_items * 8;
#  if _CCCL_CTK_AT_LEAST(12, 8)
    case CCCL_CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B:
    case CCCL_CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B:
      return __num_items / 2;
#  endif // _CCCL_CTK_AT_LEAST(12, 8)
  }
  return 0; // MSVC workaround
}

[[nodiscard]] _CCCL_HOST_API inline ::cuda::__driver::CUtensorMap __tensorMapEncodeTiled(
  ::cuda::__driver::CUtensorMapDataType __tensorDataType,
  ::cuda::std::uint32_t __tensorRank,
  void* __globalAddress,
  const ::cuda::std::uint64_t* __globalDim,
  const ::cuda::std::uint64_t* __globalStrides,
  const ::cuda::std::uint32_t* __boxDim,
  const ::cuda::std::uint32_t* __elementStrides,
  ::cuda::__driver::CUtensorMapInterleave __interleave,
  ::cuda::__driver::CUtensorMapSwizzle __swizzle,
  ::cuda::__driver::CUtensorMapL2promotion __l2Promotion,
  ::cuda::__driver::CUtensorMapFloatOOBfill __oobFill)
{
  ::cuda::__driver::CUtensorMap __tensorMap{};
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuTensorMapEncodeTiled);
  __call_driver_fn(
    __driver_fn,
    "Failed to encode TMA descriptor",
    &__tensorMap,
    __tensorDataType,
    __tensorRank,
    __globalAddress,
    __globalDim,
    __globalStrides,
    __boxDim,
    __elementStrides,
    __interleave,
    __swizzle,
    __l2Promotion,
    __oobFill);
  // workaround for nvbug 5736804
  if (::cuda::__driver::__version_below(13, 2))
  {
    const auto __tensor_req_size                = __globalDim[__tensorRank - 1] * __globalStrides[__tensorRank - 1];
    ::cuda::std::size_t __tensor_req_size_bytes = 0;
    __tensor_req_size_bytes   = ::cuda::__driver::__cutensormap_size_bytes(__tensor_req_size, __tensorDataType);
    const auto __tensorMapPtr = reinterpret_cast<::cuda::std::uint64_t*>(static_cast<void*>(&__tensorMap));
    if (__tensor_req_size_bytes < 128 * 1024) // 128 KiB
    {
      __tensorMapPtr[1] &= ~(::cuda::std::uint64_t{1} << 21); // clear the bit
    }
    else
    {
      __tensorMapPtr[1] |= ::cuda::std::uint64_t{1} << 21; // set the bit
    }
  }
  return __tensorMap;
}

#  undef _CCCLRT_GET_DRIVER_FUNCTION
#  undef _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED
#  undef _CCCLRT_CHECK_ABI_COMPATIBLE

_CCCL_END_NAMESPACE_CUDA_DRIVER

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___DRIVER_DRIVER_API_H
