// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cuda/__event/event.h>
#include <cuda/devices>
#include <cuda/stream>

#include <cstddef>
#include <vector>

#include <nvbench/nvbench.cuh>

namespace mgmn
{
//! Element counts swept by every scenario.
//!
//! Below roughly a mebielement every scenario is dominated by fixed launch and event overhead - the
//! multi-domain variants sit at a flat ~30us from 1Ki to 4Mi elements, independent of the data - so
//! the sweep starts where the reduction itself dominates. Override on the command line with
//! `-a "Elements[pow2]=[20:28:2]"`.
inline constexpr int min_elements_pow2 = 28;
inline constexpr int max_elements_pow2 = 32;
inline constexpr int elements_stride   = 1;

//! Device NVBench selected for this state.
//!
//! NVBench runs every benchmark once per visible device, so the scenario must allocate and launch on
//! the device it was given. Taking `cuda::devices[0]` unconditionally puts the buffers on the wrong
//! device for every run after the first.
[[nodiscard]] inline cuda::device_ref state_device(nvbench::state& state)
{
  return cuda::devices[state.get_device()->get_id()];
}

//! Declare what each scenario reads and writes so NVBench reports element throughput and achieved
//! global memory bandwidth. Every scenario reduces `elements` floats down to one float per rank,
//! so the read volume is what the ranking is based on.
template <class T>
inline void add_common_throughput(nvbench::state& state, std::size_t elements, int rank_count)
{
  state.add_element_count(elements, "Elements");
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(static_cast<std::size_t>(rank_count));
}

//! Number of locality domains the scenario split the device into. NVBench already reports the
//! device's total SM count, so only the partition count is added here.
inline void add_domain_count(nvbench::state& state, int rank_count)
{
  auto& summary = state.add_summary("mgmn/locality_domains");
  summary.set_string("name", "Domains");
  summary.set_string("hint", "");
  summary.set_string("description", "Locality domains the device was split into");
  summary.set_int64("value", rank_count);
}

//! Fork the domain streams off `timing_stream` and join them back onto it around one measured
//! iteration.
//!
//! NVBench records its start and stop events on the stream it hands to the benchmark, and gates that
//! stream with a blocking kernel so the whole iteration is enqueued before any of it runs. The
//! per-domain green-context streams are created by the scenario and are unknown to NVBench, so they
//! must be tied to the timed stream explicitly at both ends:
//!
//!   - fork: every domain stream waits on `fork`, recorded on the timed stream. Without this the
//!     domains start while NVBench is still enqueueing, so part of their work runs before the start
//!     event and falls outside the measured interval.
//!   - join: the timed stream waits on one event per domain, so the stop event cannot be reached
//!     until the slowest domain has finished.
//!
//! `submit` enqueues the per-rank work onto the domain streams. `fork` and `join` are built once by
//! the caller, because creating events inside the measured region would charge the measurement for
//! that host work.
template <typename SubmitFn>
inline void run_forked_iteration(
  cuda::stream_ref timing_stream,
  std::vector<cuda::stream>& streams,
  cuda::event& fork,
  std::vector<cuda::event>& join,
  SubmitFn submit)
{
  const auto rank_count = static_cast<int>(streams.size());

  fork.record(timing_stream);
  for (int rank = 0; rank < rank_count; ++rank)
  {
    streams[rank].wait(fork);
  }

  submit();

  // All records before any waits: interleaving them blocks the host between records.
  for (int rank = 0; rank < rank_count; ++rank)
  {
    join[rank].record(streams[rank]);
  }
  for (int rank = 0; rank < rank_count; ++rank)
  {
    timing_stream.wait(join[rank]);
  }
}
} // namespace mgmn
