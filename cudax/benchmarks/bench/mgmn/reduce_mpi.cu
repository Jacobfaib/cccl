// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Strong and weak scaling of `cudax::reduce` over a growing set of GPUs, across MPI processes.
//!
//! This is the multi-process form of `reduce.cu`, which drives every GPU from one process and
//! so cannot cross a node.
//!
//! One MPI rank drives one device, picked by the rank's position on its node. Every device is
//! split into its locality domains, and one NCCL rank drives each domain, so
//! `ranks = mpi_size * domains_per_device`. The "Scaling" axis decides how the "Elements" axis
//! is interpreted:
//!
//! * strong - "Elements" is the total problem size. Each rank gets `Elements / ranks` items, so
//!   the total stays constant while the rank count grows. The ideal curve is `T(N) = T(1) / N`.
//! * weak - "Elements" is the per-rank problem size. Each rank gets `Elements` items, so the
//!   total grows as `Elements * ranks`. The ideal curve is a constant `T(N) = T(1)`.
//!
//! There is no "GPUs" axis: the rank count comes from `mpirun -n`, so one run gives one point.
//!
//! Each rank allocates its input in the default memory pool of its own locality domain. Every
//! local CUB reduction therefore reads only memory local to that domain. The single step that
//! crosses domains is the NCCL all-reduce of the one-element partial results.

#include <string>
#include <vector>

// `nvbench.cuh` includes the header that reads these, so they must be defined before it.
static void mpi_initialize(int argc, char** argv);
static void mpi_per_rank_json(::std::vector<::std::string>& args);

#define NVBENCH_MAIN_INITIALIZE_CUSTOM_PRE(argc, argv) mpi_initialize(argc, argv)
#define NVBENCH_MAIN_CUSTOM_ARGS_HANDLER(args)         mpi_per_rank_json(args)
#define NVBENCH_MAIN_FINALIZE_CUSTOM_POST()            MPI_Finalize()

//! The default finalizer calls `cudaDeviceReset()`, which cannot work here.
//!
//! Each rank drives a locality domain, and a locality domain is a green context. The driver
//! refuses the reset while one is alive: "Device cannot be reset while there are still green
//! contexts present". The domains belong to a function-local static in libcu++, so they are
//! released after `main` returns, and nothing here can release them sooner.
//!
//! nvbench turns the refusal into a throw. The rank that throws leaves `main` before
//! `MPI_Finalize()`, `mpirun` kills the other ranks, and every rank that had not yet written
//! its JSON loses its results. Drop the reset and keep the rest of the sequence.
#define NVBENCH_MAIN_FINALIZE()                                                               \
  NVBENCH_MAIN_FINALIZE_CUSTOM_PRE();                                                         \
  } /* Close a scope to ensure that the inner initialize/finalize hooks clean up in order. */ \
  NVBENCH_MAIN_FINALIZE_CUSTOM_POST();                                                        \
  } /* Close a scope to ensure that the inner initialize/finalize hooks clean up in order. */ \
  []() {}()

#include <cuda/__device/logical_device_ref.h>
#include <cuda/__event/event.h>
#include <cuda/__memory_pool/locality_domain_memory_pool.h>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/ranges>
#include <cuda/std/span>
#include <cuda/stream>

#include <cuda/experimental/__multi_gpu/algorithm/reduce/reduce.h>
#include <cuda/experimental/__multi_gpu/nccl_communicator.h>

#include <stdexcept>
#include <string_view>

#include <mpi.h>
#include <nccl.h>

#include <nvbench/nvbench.cuh>

namespace cudax = cuda::experimental;

namespace
{
using element_types = nvbench::type_list<double>;

inline constexpr int min_elements_pow2 = 28;
inline constexpr int max_elements_pow2 = 32;
inline constexpr int elements_stride   = 1;

int BENCHMARK_MPI_RANK = 0;
int BENCHMARK_MPI_SIZE = 1;
//! This process's position on its own node, which picks its device.
int BENCHMARK_NODE_RANK = 0;

enum class scaling : cuda::std::int8_t
{
  strong,
  weak,
};

constexpr scaling ALL_SCALINGS[] = {scaling::strong, scaling::weak};

[[nodiscard]] std::string_view to_string(scaling scale)
{
  switch (scale)
  {
    case scaling::strong:
      return "strong";
    case scaling::weak:
      return "weak";
  }
  throw std::runtime_error{"Unknown scaling kind: " + std::to_string(static_cast<cuda::std::int8_t>(scale))};
}

[[nodiscard]] scaling scaling_from_string(std::string_view str)
{
  for (const auto scale : ALL_SCALINGS)
  {
    if (to_string(scale) == str)
    {
      return scale;
    }
  }
  throw std::runtime_error{"unknown scaling: " + std::string{str}};
}

[[nodiscard]] std::vector<std::string> scaling_axis_values()
{
  std::vector<std::string> values;

  values.reserve(cuda::std::size(ALL_SCALINGS));
  for (const auto scale : ALL_SCALINGS)
  {
    values.emplace_back(to_string(scale));
  }

  return values;
}

struct problem_size
{
  cuda::std::size_t per_rank;
  cuda::std::size_t total;
};

[[nodiscard]] problem_size make_problem_size(scaling scale, cuda::std::size_t elements, cuda::std::size_t num_ranks)
{
  switch (scale)
  {
    case scaling::strong:
      return {elements / num_ranks, elements};
    case scaling::weak:
      return {elements, elements * num_ranks};
  }
  throw std::runtime_error{"Unknown scaling kind: " + std::to_string(static_cast<cuda::std::int8_t>(scale))};
}

//! The rank count is not the process count, and without the per-rank size an under-filled rank
//! is invisible. The JSON files are per rank, and carry nothing else that tells them apart.
void add_summary(nvbench::state& state, const problem_size& size, cuda::std::size_t num_ranks)
{
  auto& ranks = state.add_summary("mgmn/ranks");

  ranks.set_string("name", "Ranks");
  ranks.set_string("hint", "");
  ranks.set_string("description", "Locality domains taking part in the reduction");
  ranks.set_int64("value", static_cast<cuda::std::int64_t>(num_ranks));

  auto& per_rank = state.add_summary("mgmn/elements_per_rank");

  per_rank.set_string("name", "Elems/rank");
  per_rank.set_string("hint", "");
  per_rank.set_string("description", "Elements reduced by each rank");
  per_rank.set_int64("value", static_cast<cuda::std::int64_t>(size.per_rank));

  auto& rank = state.add_summary("mgmn/mpi_rank");

  rank.set_string("name", "MPI rank");
  rank.set_string("hint", "");
  rank.set_string("description", "MPI rank that produced this result");
  rank.set_int64("value", static_cast<cuda::std::int64_t>(BENCHMARK_MPI_RANK));
}

//! One rank per locality domain of this process's device. A rank reduces memory that is local
//! to its own domain, so the local reduction never crosses a domain boundary.
[[nodiscard]] std::vector<cuda::__logical_device_ref> make_ranks()
{
  const auto domains = cuda::devices[BENCHMARK_NODE_RANK].__locality_domains();

  return {domains.begin(), domains.end()};
}

//! NCCL numbers this process's domains `mpi_rank * domains + domain`, so every process must own
//! the same number of domains. The communicators are created inside one group, because separate
//! `ncclCommInitRank` calls from one process deadlock against each other.
[[nodiscard]] std::vector<cudax::nccl_communicator>
make_communicators(cuda::std::span<const cuda::__logical_device_ref> ranks)
{
  const auto local = static_cast<int>(ranks.size());
  ncclUniqueId id{};

  if (BENCHMARK_MPI_RANK == 0)
  {
    if (const auto status = ncclGetUniqueId(&id); status != ncclSuccess)
    {
      throw std::runtime_error(std::string{"ncclGetUniqueId: "} + ncclGetErrorString(status));
    }
  }
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  std::vector<ncclComm_t> raw_comms(ranks.size());

  ncclGroupStart();
  for (int i = 0; i < local; ++i)
  {
    const auto idx = static_cast<cuda::std::size_t>(i);

    cudaSetDevice(ranks[idx].underlying_device().get());
    if (const auto status =
          ncclCommInitRank(&raw_comms[idx], BENCHMARK_MPI_SIZE * local, id, BENCHMARK_MPI_RANK * local + i);
        status != ncclSuccess)
    {
      throw std::runtime_error(std::string{"ncclCommInitRank: "} + ncclGetErrorString(status));
    }
  }
  if (const auto status = ncclGroupEnd(); status != ncclSuccess)
  {
    throw std::runtime_error(std::string{"ncclGroupEnd: "} + ncclGetErrorString(status));
  }

  cudaSetDevice(BENCHMARK_NODE_RANK);

  std::vector<cudax::nccl_communicator> comms;

  comms.reserve(ranks.size());
  for (auto&& [raw_comm, rank] : cuda::std::views::zip(raw_comms, ranks))
  {
    comms.push_back(cudax::nccl_communicator::from_native_handle(raw_comm, rank));
  }

  return comms;
}

//! `stdrel` with the stop decision made unanimous.
//!
//! nvbench asks each process whether it has converged, using only its own timings. Noise and
//! discarded throttled samples make the processes answer differently. The one that answers
//! first leaves the measurement loop and never enters the next all-reduce, so every other
//! process blocks there forever.
//!
//! Only the cold measurement consults a criterion, so the benchmark must keep `no_batch`.
class mpi_stdrel final : public nvbench::stopping_criterion_base
{
public:
  mpi_stdrel()
      : nvbench::stopping_criterion_base{"mpi_stdrel", nvbench::detail::stdrel_criterion{}.get_params()}
  {}

private:
  void do_initialize() override
  {
    local_.initialize(m_params);
  }

  void do_add_measurement(nvbench::float64_t measurement) override
  {
    local_.add_measurement(measurement);
  }

  bool do_is_finished() override
  {
    int in  = local_.is_finished() ? 1 : 0;
    int out = 0;

    MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    return out != 0;
  }

  nvbench::detail::stdrel_criterion local_;
};
NVBENCH_REGISTER_CRITERION(mpi_stdrel);

//! Fork the timing stream to every rank and join them back, so the timing stream measures the
//! slowest rank.
template <typename Buffers, typename SubmitFn>
void run_forked_iteration(
  cuda::stream_ref timing_stream, Buffers& bufs, cuda::event& fork, std::vector<cuda::event>& join, SubmitFn submit)
{
  fork.record(timing_stream);
  for (auto& buf : bufs)
  {
    buf.stream().wait(fork);
  }

  submit();

  // All records before any waits: interleaving them blocks the host between records.
  for (auto&& [event, buf] : cuda::std::views::zip(join, bufs))
  {
    event.record(buf.stream());
  }

  for (auto& event : join)
  {
    timing_stream.wait(event);
  }
}

template <class T>
void reduce(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<cuda::std::size_t>(state.get_int64("Elements"));
  const auto scale    = scaling_from_string(state.get_string("Scaling"));

  const auto ranks = make_ranks();

  const auto num_ranks = static_cast<cuda::std::size_t>(BENCHMARK_MPI_SIZE) * ranks.size();
  const auto size      = make_problem_size(scale, elements, num_ranks);

  // The domains of one device share its memory, so the whole device must hold every rank it owns.
  const auto per_device = size.per_rank * ranks.size();

  if (per_device * sizeof(T) > cuda::device_attributes::total_global_memory(ranks.front().underlying_device()))
  {
    state.skip("Input does not fit in device memory");
    return;
  }

  add_summary(state, size, num_ranks);

  auto comms = make_communicators(ranks);

  // Each rank allocates in the pool of its own domain, so its local reduction reads only memory
  // that is local to that domain.
  std::vector<cuda::stream> streams;
  std::vector<cuda::device_buffer<T>> in;
  std::vector<cuda::device_buffer<T>> out;
  std::vector<cuda::event> join;
  // Recorded on the nvbench timing stream, which lives on this process's device.
  cuda::event fork{ranks.front().underlying_device()};

  streams.reserve(comms.size());
  in.reserve(comms.size());
  out.reserve(comms.size());
  join.reserve(comms.size());
  for (auto&& comm : comms)
  {
    auto domain = comm.logical_device();
    auto& pool  = cuda::__device_default_memory_pool(domain);
    auto& s     = streams.emplace_back(domain);

    in.push_back(cuda::make_buffer<T>(s, pool, size.per_rank, T{1}));
    out.push_back(cuda::make_buffer<T>(s, pool, 1, cuda::no_init));
    join.emplace_back(domain.underlying_device());
  }

  state.add_element_count(size.total);
  state.add_global_memory_reads<T>(size.total);
  // One result per rank.
  state.add_global_memory_writes<T>(comms.size());

  for (auto&& s : streams)
  {
    s.sync();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    run_forked_iteration(cuda::stream_ref{launch.get_stream().get_stream()}, in, fork, join, [&] {
      cudax::reduce(
        cudax::broadcasted,
        comms,
        in | cuda::std::views::transform([](auto& buf) {
          return cuda::std::execution::env{buf.stream(), buf.memory_resource()};
        }),
        in | cuda::std::views::transform(cuda::std::ranges::begin),
        in | cuda::std::views::transform(cuda::std::ranges::size),
        out | cuda::std::views::transform(cuda::std::ranges::begin));
    });
  });
  for (auto&& s : streams)
  {
    s.sync();
  }
  cudaDeviceSynchronize();
  MPI_Barrier(MPI_COMM_WORLD);
}
NVBENCH_BENCH_TYPES(reduce, NVBENCH_TYPE_AXES(element_types))
  .set_name("reduce_mpi")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(min_elements_pow2, max_elements_pow2, elements_stride))
  .add_string_axis("Scaling", scaling_axis_values())
  .set_stopping_criterion("mpi_stdrel");
} // namespace

void mpi_initialize(int argc, char** argv)
{
  // NCCL is driven from this one host thread, so serialized access is enough.
  int provided = MPI_THREAD_SINGLE;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  if (provided < MPI_THREAD_SERIALIZED)
  {
    throw std::runtime_error{"MPI does not provide MPI_THREAD_SERIALIZED"};
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &BENCHMARK_MPI_RANK);
  MPI_Comm_size(MPI_COMM_WORLD, &BENCHMARK_MPI_SIZE);

  MPI_Comm node = MPI_COMM_NULL;
  int node_size = 0;

  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, BENCHMARK_MPI_RANK, MPI_INFO_NULL, &node);
  MPI_Comm_rank(node, &BENCHMARK_NODE_RANK);
  MPI_Comm_size(node, &node_size);
  MPI_Comm_free(&node);

  if (node_size > static_cast<int>(cuda::devices.size()))
  {
    throw std::runtime_error{"More ranks on this node (" + std::to_string(node_size) + ") than devices ("
                             + std::to_string(cuda::devices.size()) + ")"};
  }
}

//! Give each rank its own JSON file, and let only rank 0 write the shared stdout.
//!
//! The process count is part of the name. A scaling sweep runs the same command at several rank
//! counts, and without it each run would overwrite the results of the last.
void mpi_per_rank_json(std::vector<std::string>& args)
{
  const auto tag = ".n" + std::to_string(BENCHMARK_MPI_SIZE) + ".rank" + std::to_string(BENCHMARK_MPI_RANK);

  for (cuda::std::size_t i = 1; i < args.size(); ++i)
  {
    if (args[i - 1] == "--json" || args[i - 1] == "--jsonbin")
    {
      args[i].insert(args[i].rfind('.'), tag);
    }
  }

  if (BENCHMARK_MPI_RANK != 0)
  {
    args.emplace_back("--quiet");
  }

  // Without this every process runs every state on every visible device, and the states that
  // land on another process's device fail.
  args.emplace_back("--devices");
  args.emplace_back(std::to_string(BENCHMARK_NODE_RANK));
}

NVBENCH_MAIN
