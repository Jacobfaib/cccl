// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Strong and weak scaling of `cudax::reduce` over a growing set of GPUs.
//!
//! The "GPUs" axis selects how many devices take part. Every device is split into its locality
//! domains, and one rank drives each domain, so `ranks = sum of domains over the selected
//! devices`. The "Scaling" axis decides how the "Elements" axis is interpreted:
//!
//! * strong - "Elements" is the total problem size. Each rank gets `Elements / ranks` items, so
//!   the total stays constant while the rank count grows. The ideal curve is `T(N) = T(1) / N`.
//! * weak - "Elements" is the per-rank problem size. Each rank gets `Elements` items, so the
//!   total grows as `Elements * ranks`. The ideal curve is a constant `T(N) = T(1)`.
//!
//! Both the GPU count and the per-device domain count are powers of two, so a power-of-two
//! "Elements" always divides evenly across the ranks.
//!
//! Each rank allocates its input in the default memory pool of its own locality domain. Every
//! local CUB reduction therefore reads only memory local to that domain. The single step that
//! crosses domains is the NCCL all-reduce of the one-element partial results.

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
#include <string>
#include <string_view>
#include <vector>

#include <nccl.h>

#include <nvbench/nvbench.cuh>

namespace cudax = cuda::experimental;

namespace
{
using element_types = nvbench::type_list<double>;

inline constexpr int min_elements_pow2 = 26;
inline constexpr int max_elements_pow2 = 32;
inline constexpr int elements_stride   = 1;

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

constexpr cuda::std::int64_t GPU_COUNTS[] = {1, 2, 4, 8};

[[nodiscard]] std::vector<cuda::std::int64_t> gpu_axis_values()
{
  return {cuda::std::begin(GPU_COUNTS), cuda::std::end(GPU_COUNTS)};
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

//! The rank count is not the GPU count, and without the per-rank size an under-filled rank is
//! invisible.
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
}

//! One rank per locality domain of devices `[0, num_gpus)`. A rank reduces memory that is local
//! to its own domain, so the local reduction never crosses a domain boundary.
[[nodiscard]] std::vector<cuda::__logical_device_ref> make_ranks(cuda::std::size_t num_gpus)
{
  std::vector<cuda::__logical_device_ref> ranks;

  for (const auto device : cuda::devices | cuda::std::views::take(num_gpus))
  {
    const auto domains = device.__locality_domains();

    ranks.insert(ranks.end(), domains.begin(), domains.end());
  }

  return ranks;
}

//! Several ranks may share one physical device when that device has more than one domain. NCCL
//! accepts the repeated device id, and each resulting communicator drives a distinct domain.
[[nodiscard]] std::vector<cudax::nccl_communicator>
make_communicators(cuda::std::span<const cuda::__logical_device_ref> ranks)
{
  std::vector<ncclComm_t> raw_comms(ranks.size());
  std::vector<int> devs;

  devs.reserve(ranks.size());
  for (const auto rank : ranks)
  {
    devs.push_back(rank.underlying_device().get());
  }

  if (const auto status = ncclCommInitAll(raw_comms.data(), static_cast<int>(devs.size()), devs.data());
      status != ncclSuccess)
  {
    throw std::runtime_error(std::string{"ncclCommInitAll: "} + ncclGetErrorString(status));
  }

  std::vector<cudax::nccl_communicator> comms;

  comms.reserve(ranks.size());
  for (auto&& [raw_comm, rank] : cuda::std::views::zip(raw_comms, ranks))
  {
    comms.push_back(cudax::nccl_communicator::from_native_handle(raw_comm, rank));
  }

  return comms;
}

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
  // The reduction forks from device 0, so the nvbench timing stream must live there.
  if (state.get_device().value().get_id() != 0) // NOLINT(bugprone-unchecked-optional-access)
  {
    throw std::runtime_error{"This benchmark must be run with `-d 0`"};
  }

  // A power-of-two axis reports its exponent, not its value.
  const auto elements = cuda::std::size_t{1} << state.get_int64("Elements");
  const auto num_gpus = static_cast<cuda::std::size_t>(state.get_int64("GPUs"));
  const auto scale    = scaling_from_string(state.get_string("Scaling"));
  const auto ranks    = make_ranks(num_gpus);

  if (ranks.empty())
  {
    state.skip("Not enough GPUs for this GPU count");
    return;
  }

  const auto num_ranks = ranks.size();
  const auto size      = make_problem_size(scale, elements, num_ranks);

  // The domains of one device share its memory, so the whole device must hold every rank it owns.
  const auto per_device = size.per_rank * (num_ranks / num_gpus);

  if (per_device * sizeof(T) > cuda::device_attributes::total_global_memory(cuda::devices[0]))
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
  // Recorded on the nvbench timing stream, which lives on device 0.
  cuda::event fork{cuda::devices[0]};

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
}
NVBENCH_BENCH_TYPES(reduce, NVBENCH_TYPE_AXES(element_types))
  .set_name("reduce")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(min_elements_pow2, max_elements_pow2, elements_stride))
  .add_int64_axis("GPUs", gpu_axis_values())
  .add_string_axis("Scaling", scaling_axis_values());
} // namespace
