if (NOT cudax_ENABLE_NCCL)
  return()
endif()

file(GLOB bench_srcs CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

# The `_mpi` benchmarks are multi-process and need an MPI implementation. Split them out so the
# single-process benchmarks still build when no MPI is installed.
set(mpi_bench_srcs ${bench_srcs})
list(FILTER bench_srcs EXCLUDE REGEX "_mpi\\.cu$")
list(FILTER mpi_bench_srcs INCLUDE REGEX "_mpi\\.cu$")

add_bench_dir(bench_targets ${bench_srcs})

if (mpi_bench_srcs)
  # The benchmarks call the MPI C API. FindMPI refuses to probe the C component unless C is an
  # enabled language, and the project enables only CXX and CUDA.
  enable_language(C)

  find_package(MPI QUIET COMPONENTS C)

  option(
    cudax_ENABLE_MPI
    "Enable cudax benchmarks that link against MPI."
    "${MPI_C_FOUND}"
  )

  if (cudax_ENABLE_MPI)
    if (NOT MPI_C_FOUND)
      find_package(MPI REQUIRED COMPONENTS C)
    endif()

    add_bench_dir(mpi_bench_targets ${mpi_bench_srcs})

    foreach (bench_target IN LISTS mpi_bench_targets)
      target_link_libraries(${bench_target} PRIVATE MPI::MPI_C)
    endforeach()

    list(APPEND bench_targets ${mpi_bench_targets})
  else()
    message(
      STATUS
      "cudax: MPI not found, skipping multi-process mgmn benchmarks"
    )
  endif()
endif()

foreach (bench_target IN LISTS bench_targets)
  target_link_libraries(${bench_target} PRIVATE NCCL::nccl)
endforeach()
