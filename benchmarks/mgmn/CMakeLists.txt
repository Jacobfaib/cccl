cmake_minimum_required(VERSION 3.24)

project(cccl_mgmn_benchmarks LANGUAGES CXX CUDA)

find_package(CUDAToolkit 13 REQUIRED)
find_package(benchmark 1.9.5 REQUIRED)

find_path(NCCL_INCLUDE_DIR nccl.h REQUIRED)
find_library(NCCL_LIBRARY nccl REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES native)

function(add_mgmn_benchmark target source)
  add_executable(${target} ${source})
  target_include_directories(
    ${target}
    PRIVATE
      ${PROJECT_SOURCE_DIR}/../../cub
      ${PROJECT_SOURCE_DIR}/../../cudax/include
      ${PROJECT_SOURCE_DIR}/../../libcudacxx/include
      ${NCCL_INCLUDE_DIR}
  )
  target_link_libraries(
    ${target}
    PRIVATE benchmark::benchmark CUDA::cuda_driver ${NCCL_LIBRARY}
  )
  target_compile_options(
    ${target}
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
  )
endfunction()

add_mgmn_benchmark(cub_full_device cub_full_device.cu)
add_mgmn_benchmark(cudax_host_nccl cudax_host_nccl.cu)
add_mgmn_benchmark(cub_green_atomic cub_green_atomic.cu)
add_mgmn_benchmark(cudax_device_nccl cudax_device_nccl.cu)
