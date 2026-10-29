// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_reduce.cuh>

#include <cuda/__device/logical_device_ref.h>
#include <cuda/__driver/driver_api.h>
#include <cuda/__event/event.h>
#include <cuda/__memory_pool/locality_domain_memory_pool.h>
#include <cuda/__runtime/api_wrapper.h>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/ranges>
#include <cuda/std/span>
#include <cuda/stream>

#include <cuda/experimental/memory_resource.cuh>
#include <cuda/experimental/mgmn/nccl_communicators>
#include <cuda/experimental/mgmn/reduce>

#include <stdexcept>
#include <string>
#include <vector>

#include <nccl.h>

#include <nvbench/nvbench.cuh>

namespace cudax = cuda::experimental;

using T = float;

namespace
{
[[nodiscard]] std::vector<cudax::mgmn::nccl_communicator>
make_communicators(cuda::device_ref device, cuda::std::span<const cuda::__logical_device_ref> domains)
{
  std::vector<ncclComm_t> raw_comms(domains.size());
  std::vector<int> devs(domains.size(), device.get());
  std::vector<cudax::mgmn::nccl_communicator> comms;
  comms.reserve(domains.size());

  if (const auto status = ncclCommInitAll(raw_comms.data(), static_cast<int>(devs.size()), devs.data());
      status != ncclSuccess)
  {
    throw std::runtime_error(std::string{"ncclCommInitAll: "} + ncclGetErrorString(status));
  }

  for (cuda::std::size_t domain = 0; domain < domains.size(); ++domain)
  {
    comms.push_back(cudax::mgmn::nccl_communicator::from_native_handle(raw_comms[domain], domains[domain]));
  }
  return comms;
}

template <typename SubmitFn>
void run_forked_iteration(
  cuda::stream_ref timing_stream,
  std::vector<cuda::stream>& streams,
  cuda::event& fork,
  std::vector<cuda::event>& join,
  SubmitFn submit)
{
  fork.record(timing_stream);
  for (auto& stream : streams)
  {
    stream.wait(fork);
  }

  submit();

  // All records before any waits: interleaving them blocks the host between records.
  for (auto&& [event, stream] : cuda::std::views::zip(join, streams))
  {
    event.record(stream);
  }
  for (auto& event : join)
  {
    timing_stream.wait(event);
  }
}

void cudax_reduce(nvbench::state& state)
{
  const auto elements = static_cast<cuda::std::size_t>(state.get_int64("Elements"));
  const auto device  = cuda::devices[state.get_device().value().get_id()]; // NOLINT(bugprone-unchecked-optional-access)
  const auto domains = device.__locality_domains();
  const auto rank_count = domains.size();

  if (rank_count < 2)
  {
    state.skip("the GPU does not expose multiple locality domains");
    return;
  }

  if (elements % rank_count != 0)
  {
    throw std::runtime_error{"Elements must divide evenly across locality domains"};
  }

  const auto per_rank = elements / rank_count;

  auto comms = make_communicators(device, domains);
  std::vector<cuda::stream> streams{domains.begin(), domains.end()};
  std::vector<cuda::device_buffer<T>> inputs;
  std::vector<cuda::device_buffer<T>> outputs;
  cuda::event fork{device};
  std::vector<cuda::event> join;

  inputs.reserve(rank_count);
  outputs.reserve(rank_count);
  join.reserve(rank_count);
  for (cuda::std::size_t rank = 0; rank < rank_count; ++rank)
  {
    auto& pool = cuda::__device_default_memory_pool(domains[rank]);

    inputs.push_back(cuda::make_buffer<T>(streams[rank], pool, per_rank, T{1}));
    outputs.push_back(cuda::make_buffer<T>(streams[rank], pool, 1, cuda::no_init));
    join.emplace_back(device);
  }

  for (auto& stream : streams)
  {
    stream.sync();
  }

  // Reject non-localized allocations before measuring locality-sensitive bandwidth.
  for (cuda::std::size_t rank = 0; rank < rank_count; ++rank)
  {
    if (inputs[rank].empty())
    {
      continue;
    }

    unsigned int ordinal        = ~0U;
    CUpointer_attribute attrs[] = {CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL};
    void* values[]              = {&ordinal};

    _CCCL_TRY_DRIVER_API(
      cuda::__driver::__pointerGetAttributesNoThrow,
      "Failed to query the locality domain of a pointer",
      attrs,
      values,
      inputs[rank].data());

    if (ordinal != domains[rank].locality_domain().domain_id)
    {
      state.skip("an input buffer did not land in its requested locality domain");
      return;
    }
  }

  // Use the localized pool for the reduction's temporary storage as well as its input.
  auto environments = inputs | cuda::std::views::transform([](auto& input) {
                        return cuda::std::execution::env{
                          input.stream(),
                          input.memory_resource(),
                          cuda::execution::require(cuda::execution::determinism::not_guaranteed)};
                      });

  state.add_element_count(elements, "Elements");
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(rank_count);

  auto& summary = state.add_summary("mgmn/locality_domains");

  summary.set_string("name", "Domains");
  summary.set_string("hint", "");
  summary.set_string("description", "Locality domains the device was split into");
  summary.set_int64("value", static_cast<cuda::std::int64_t>(rank_count));

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    run_forked_iteration(cuda::stream_ref{launch.get_stream().get_stream()}, streams, fork, join, [&] {
      cudax::mgmn::reduce(
        cudax::broadcasted,
        comms,
        environments,
        inputs | cuda::std::views::transform(cuda::std::ranges::data),
        inputs | cuda::std::views::transform(cuda::std::ranges::size),
        outputs | cuda::std::views::transform(cuda::std::ranges::data));
    });
  });
}

constexpr int min_elements_pow2 = 28;
constexpr int max_elements_pow2 = 32;
constexpr int elements_stride   = 1;
} // namespace

NVBENCH_BENCH(cudax_reduce)
  .set_name("cudax_reduce")
  .add_int64_power_of_two_axis("Elements", nvbench::range(min_elements_pow2, max_elements_pow2, elements_stride));

namespace
{
void cub_reduce(nvbench::state& state)
{
  const auto elements = static_cast<cuda::std::size_t>(state.get_int64("Elements"));
  const auto device = cuda::devices[state.get_device().value().get_id()]; // NOLINT(bugprone-unchecked-optional-access)
  cuda::stream stream{device};

  const auto input = cuda::make_device_buffer<T>(stream, device, elements, T{1});
  auto output      = cuda::make_device_buffer<T>(stream, device, 1, cuda::no_init);
  stream.sync();

  const auto env = cuda::std::execution::env{
    input.memory_resource(), cuda::execution::require(cuda::execution::determinism::not_guaranteed)};

  state.add_element_count(elements, "Elements");
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(1);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    const auto env_with_stream = cuda::std::execution::env{cuda::stream_ref{launch.get_stream().get_stream()}, env};

    _CCCL_TRY_RUNTIME_API(
      cub::DeviceReduce::Reduce,
      "cub::DeviceReduce::Reduce failed",
      input.data(),
      output.data(),
      input.size(),
      cuda::std::plus<>{},
      T{},
      env_with_stream);
  });
}
} // namespace

NVBENCH_BENCH(cub_reduce)
  .set_name("cub_reduce")
  .add_int64_power_of_two_axis("Elements", nvbench::range(min_elements_pow2, max_elements_pow2, elements_stride));

#if 0
namespace
{
void cub_reduce_striped(nvbench::state& state)
{
  const auto elements = static_cast<cuda::std::size_t>(state.get_int64("Elements"));
  const auto device  = cuda::devices[state.get_device().value().get_id()]; // NOLINT(bugprone-unchecked-optional-access)
  const auto domains = device.__locality_domains();
  const auto rank_count = domains.size();

  if (rank_count < 2)
  {
    state.skip("the GPU does not expose multiple locality domains");
    return;
  }
  if (elements % rank_count != 0)
  {
    throw std::runtime_error{"Elements must divide evenly across locality domains"};
  }
  const auto per_rank = elements / rank_count;

  cuda::stream stream{device};
  cudax::locality_domain_striped_memory_resource resource{device, per_rank * sizeof(T)};
  const auto input = cuda::make_buffer<T>(stream, resource, elements, T{1});
  auto output      = cuda::make_device_buffer<T>(stream, device, 1, cuda::no_init);
  stream.sync();

  state.add_element_count(elements, "Elements");
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(1);

  auto& summary = state.add_summary("mgmn/locality_domains");
  summary.set_string("name", "Domains");
  summary.set_string("hint", "");
  summary.set_string("description", "Locality domains the device was split into");
  summary.set_int64("value", static_cast<cuda::std::int64_t>(rank_count));

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    const auto env = cuda::std::execution::env{
      cuda::stream_ref{launch.get_stream().get_stream()},
      cuda::execution::require(cuda::execution::determinism::not_guaranteed),
      cub::striped_marker_t{static_cast<int>(rank_count)}};

    _CCCL_TRY_RUNTIME_API(
      cub::DeviceReduce::Reduce,
      "cub::DeviceReduce::Reduce failed",
      input.data(),
      output.data(),
      input.size(),
      cuda::std::plus<>{},
      T{},
      env);
  });
}
} // namespace
#else
namespace
{
void cub_reduce_striped(nvbench::state& state)
{
  const auto elements = static_cast<cuda::std::size_t>(state.get_int64("Elements"));
  const auto device  = cuda::devices[state.get_device().value().get_id()]; // NOLINT(bugprone-unchecked-optional-access)
  const auto domains = device.__locality_domains();
  const auto rank_count = domains.size();

  if (rank_count < 2)
  {
    state.skip("the GPU does not expose multiple locality domains");
    return;
  }
  if (elements % rank_count != 0)
  {
    throw std::runtime_error{"Elements must divide evenly across locality domains"};
  }
  const auto per_rank = elements / rank_count;

  cuda::stream stream{device};
  cudax::locality_domain_striped_memory_resource resource{device, per_rank * sizeof(T)};
  const auto input = cuda::make_buffer<T>(stream, resource, elements, T{1});
  auto output      = cuda::make_device_buffer<T>(stream, device, 1, cuda::no_init);
  stream.sync();

  std::vector<cuda::stream> streams{domains.begin(), domains.end()};
  cuda::event fork{device};
  std::vector<cuda::event> join;
  join.reserve(rank_count);
  for (cuda::std::size_t rank = 0; rank < rank_count; ++rank)
  {
    join.emplace_back(device);
  }

  state.add_element_count(elements, "Elements");
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_reads<T>(rank_count, "Partials");
  state.add_global_memory_writes<T>(rank_count);

  auto& summary = state.add_summary("mgmn/locality_domains");
  summary.set_string("name", "Domains");
  summary.set_string("hint", "");
  summary.set_string("description", "Locality domains the device was split into");
  summary.set_int64("value", static_cast<cuda::std::int64_t>(rank_count));

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    const cuda::stream_ref timing_stream{launch.get_stream().get_stream()};
    run_forked_iteration(timing_stream, streams, fork, join, [&] {
      for (cuda::std::size_t rank = 0; rank < rank_count; ++rank)
      {
        const auto env = cuda::std::execution::env{
          cuda::stream_ref{streams[rank]},
          cuda::__device_default_memory_pool(domains[rank]),
          cuda::execution::require(cuda::execution::determinism::not_guaranteed)};

        _CCCL_TRY_RUNTIME_API(
          cub::DeviceReduce::Reduce,
          "cub::DeviceReduce::Reduce failed",
          input.data() + rank * per_rank,
          output.data(),
          per_rank,
          cuda::std::plus<>{},
          T{},
          env);
      }
    });


}
} // namespace
#endif

NVBENCH_BENCH(cub_reduce_striped)
  .set_name("cub_reduce_striped")
  .add_int64_power_of_two_axis("Elements", nvbench::range(min_elements_pow2, max_elements_pow2, elements_stride));
