# Multi-GPU/multi-node (mgmn) benchmarks. Added from the top-level CCCL build when
# CCCL_ENABLE_MGMN_BENCHMARKS is set, so the CCCL targets (cudax, CUB, libcu++) already exist.

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

cccl_get_nccl(REQUIRED)

# Pull google-benchmark through CPM, which ships in the CCCL repo's cmake directory.
# CPMFindPackage prefers an existing local install and downloads the pinned tag otherwise.
include("${CCCL_SOURCE_DIR}/cmake/CPM.cmake")
CPMFindPackage(
  NAME benchmark
  VERSION 1.9.5
  GIT_REPOSITORY
  "https://github.com/google/benchmark.git"
  GIT_TAG v1.9.5
  OPTIONS
  "BENCHMARK_ENABLE_TESTING OFF"
  "BENCHMARK_ENABLE_INSTALL OFF"
)

function(add_mgmn_benchmark target source)
  add_executable(${target} ${source})
  target_link_libraries(
    ${target}
    PRIVATE
      libcudacxx::libcudacxx
      cudax::cudax
      CUB::CUB
      benchmark::benchmark
      CUDA::cuda_driver
      NCCL::nccl
  )
  target_compile_options(
    ${target}
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
  )
  set_target_properties(
    ${target}
    PROPERTIES CUDA_STANDARD 20 CXX_STANDARD 20 CUDA_ARCHITECTURES native
  )
endfunction()

add_mgmn_benchmark(cub_full_device cub_full_device.cu)
add_mgmn_benchmark(cudax_host_nccl cudax_host_nccl.cu)
add_mgmn_benchmark(cub_green_atomic cub_green_atomic.cu)
add_mgmn_benchmark(cudax_device_nccl cudax_device_nccl.cu)
target_compile_definitions(
  cudax_device_nccl
  PRIVATE NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE=1
)
