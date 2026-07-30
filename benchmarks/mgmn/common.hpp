// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <benchmark/benchmark.h>

namespace mgmn
{
inline constexpr std::string_view sizes_option = "--cccl-benchmark-sizes";

//! Describes how to pass element counts, appended to every parse failure so the message is
//! actionable on its own.
inline std::string sizes_usage()
{
  return std::string{"usage: "} + std::string{sizes_option} + "=<N>[,<N>...] or " + std::string{sizes_option}
       + " <N> [<N>...]  (each count must be a unique positive integer <= "
       + std::to_string((std::numeric_limits<int>::max)()) + ")";
}

inline std::size_t parse_size(std::string_view value)
{
  if (value.empty())
  {
    throw std::invalid_argument("empty element count in " + std::string{sizes_option} + ". " + sizes_usage());
  }

  std::size_t result{};
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
  if (end != value.data() + value.size() || error == std::errc::invalid_argument)
  {
    throw std::invalid_argument("'" + std::string{value} + "' is not a valid element count. " + sizes_usage());
  }
  if (error == std::errc::result_out_of_range)
  {
    throw std::invalid_argument("element count '" + std::string{value} + "' is out of range. " + sizes_usage());
  }
  if (result == 0)
  {
    throw std::invalid_argument("element count must be positive, got '0'. " + sizes_usage());
  }
  return result;
}

//! Parse one comma-separated group of counts, appending to `result`. Rejects values that cannot be
//! represented by `int` (google-benchmark's `Arg` takes `int64_t`, but the sizes are also used as
//! `int` element counts) and duplicates, which would silently register the same case twice.
inline void parse_sizes_into(std::string_view values, std::vector<std::size_t>& result)
{
  while (true)
  {
    const auto separator   = values.find(',');
    const std::size_t size = parse_size(values.substr(0, separator));

    if (std::find(result.begin(), result.end(), size) != result.end())
    {
      throw std::invalid_argument("duplicate element count " + std::to_string(size) + ". " + sizes_usage());
    }
    result.push_back(size);

    if (separator == std::string_view::npos)
    {
      return;
    }
    values.remove_prefix(separator + 1);
  }
}

inline std::vector<std::size_t> parse_sizes(std::string_view values)
{
  std::vector<std::size_t> result;
  parse_sizes_into(values, result);
  return result;
}

//! Extract the element counts from `argv`, removing the consumed arguments so the remainder can be
//! handed to `benchmark::Initialize`.
//!
//! Accepts both `--cccl-benchmark-sizes=1,2,3` and `--cccl-benchmark-sizes 1 2 3`. In the
//! space-separated form the option consumes every following argument until the next one beginning
//! with `-`; google-benchmark's own flags are all `--benchmark_*`, so they are never swallowed.
//! Individual arguments may still be comma-separated, making `--cccl-benchmark-sizes 1,2 3` valid.
inline std::vector<std::size_t> take_sizes_option(int& argc, char** argv)
{
  const std::string prefix{std::string{sizes_option} + "="};
  std::vector<std::size_t> sizes;
  bool seen  = false;
  int output = 1;
  for (int input = 1; input < argc; ++input)
  {
    const std::string_view arg{argv[input]};
    if (arg.starts_with(prefix))
    {
      if (seen)
      {
        throw std::invalid_argument(std::string{sizes_option} + " may only be specified once. " + sizes_usage());
      }
      seen = true;
      parse_sizes_into(arg.substr(prefix.size()), sizes);
    }
    else if (arg == sizes_option)
    {
      if (seen)
      {
        throw std::invalid_argument(std::string{sizes_option} + " may only be specified once. " + sizes_usage());
      }
      seen = true;
      // Consume the following arguments up to the next option.
      int value = input + 1;
      for (; value < argc && !std::string_view{argv[value]}.starts_with('-'); ++value)
      {
        parse_sizes_into(argv[value], sizes);
      }
      if (sizes.empty())
      {
        throw std::invalid_argument(
          std::string{sizes_option} + " requires at least one element count. " + sizes_usage());
      }
      input = value - 1;
    }
    else
    {
      argv[output++] = argv[input];
    }
  }
  argc = output;
  if (!seen)
  {
    throw std::invalid_argument("missing required option " + std::string{sizes_option} + ". " + sizes_usage());
  }
  return sizes;
}

inline void set_common_counters(benchmark::State& state, std::size_t elements, unsigned int sm_count = 0)
{
  state.counters["elements"]    = static_cast<double>(elements);
  state.counters["input_bytes"] = static_cast<double>(elements * sizeof(float));
  if (sm_count != 0)
  {
    state.counters["sm_count"] = static_cast<double>(sm_count);
  }
}

template <typename BenchmarkT>
int run_benchmark(int argc, char** argv, const char* scenario, BenchmarkT benchmark_function)
{
  const auto sizes = take_sizes_option(argc, argv);
  benchmark::Initialize(&argc, argv);
  benchmark::AddCustomContext("scenario", scenario);
  auto* registration = benchmark::RegisterBenchmark(std::string{scenario} + "/elements", benchmark_function);
  registration->UseManualTime();
  for (const auto size : sizes)
  {
    registration->Arg(static_cast<std::size_t>(size));
  }
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
} // namespace mgmn
