#!/usr/bin/env bash

# Runs benchmark executables from two build trees and compares their results.
#
# The sourcing script defines artifact_dir, the NVBENCH_*_ARGS_STRING values and the
# any_failures, compares_attempted and compares_succeeded counters.
# shellcheck disable=SC2034,SC2154

set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced, not executed directly." >&2
  exit 1
fi

readonly -a COMPARE_DISPLAYS=(intervals simple explain)

die() {
  local message="$1"
  local code="${2:-2}"
  echo "${message}" >&2
  exit "${code}"
}

sanitize_label() {
  local label="$1"
  label="${label//[^a-zA-Z0-9._-]/_}"
  label="${label#_}"
  label="${label%_}"
  label="${label:-unknown}"
  printf "%s" "${label}"
}

print_shell_command() {
  # Print a shell-escaped command line so it can be copied and re-run directly.
  # Usage: print_shell_command [--env "VAR=value" ...] cmd [args...]
  local -a env_prefixes=()
  while [[ "$#" -gt 0 && "$1" == --env ]]; do
    shift
    env_prefixes+=("$1")
    shift
  done
  printf '$'
  local item=""
  for item in "${env_prefixes[@]}"; do
    printf ' %q' "${item}"
  done
  for item in "$@"; do
    printf ' %q' "${item}"
  done
  printf '\n'
}

resolve_robust_compare_script() {
  local build_path="$1"
  local nvbench_src="${build_path}/_deps/nvbench-src"
  local candidate="${nvbench_src}/python/scripts/nvbench_compare_robust.py"
  if [[ -f "${candidate}" ]]; then
    printf "%s" "${candidate}"
  fi
  return 0
}

run_target_for_side() {
  local side="$1"
  local build_path="$2"
  local target="$3"
  local json_path="$4"
  local md_path="$5"
  local run_log="$6"
  local binary_path="${build_path}/bin/${target}"
  local -a bench_cmd

  if [[ ! -x "${binary_path}" ]]; then
    echo "Benchmark binary missing: ${binary_path}" >&2
    return 127
  fi

  bench_cmd=(
    "${binary_path}"
    -d 0
    "${NVBENCH_RUN_ARGS[@]}"
    --jsonbin "${json_path}"
    --md "${md_path}"
  )

  run_grouped_logged_command \
    "[run:${side}] ${target}" \
    "${run_log}" \
    "${bench_cmd[@]}"
}

run_grouped_logged_command() {
  local label="$1"
  local log_path="$2"
  shift 2
  local started_at=0
  local elapsed_s=0
  local rc=0
  local -a pipe_statuses

  echo "::group::${label}"
  print_shell_command "$@"
  started_at="${SECONDS}"
  set +o pipefail
  "$@" 2>&1 | tee "${log_path}"
  pipe_statuses=("${PIPESTATUS[@]}")
  set -o pipefail
  if [[ "${pipe_statuses[0]}" -ne 0 ]]; then
    rc="${pipe_statuses[0]}"
  elif [[ "${pipe_statuses[1]}" -ne 0 ]]; then
    rc="${pipe_statuses[1]}"
  fi
  elapsed_s=$((SECONDS - started_at))
  echo "::endgroup::"
  if [[ "${rc}" -eq 0 ]]; then
    echo "${label} completed in ${elapsed_s}s"
  else
    echo "${label} failed in ${elapsed_s}s (rc=${rc})"
  fi
  return "${rc}"
}

compare_report_path_for_display() {
  local target_name="$1"
  local display="$2"

  if [[ "${display}" == "intervals" ]]; then
    printf "%s" "${artifact_dir}/compare/${target_name}.md"
  else
    printf "%s" "${artifact_dir}/compare/${target_name}.${display}.md"
  fi
}

compare_log_path_for_display() {
  local target_name="$1"
  local display="$2"

  if [[ "${display}" == "intervals" ]]; then
    printf "%s" "${artifact_dir}/logs/compare.${target_name}.log"
  else
    printf "%s" "${artifact_dir}/logs/compare.${target_name}.${display}.log"
  fi
}

normalize_explain_markdown_report() {
  local report_path="$1"
  local tmp_path=""

  [[ -s "${report_path}" ]] || return 0

  tmp_path="$(mktemp "${report_path}.XXXXXX")"
  if ! awk '
    function clear_array(values, key) {
      for (key in values) {
        delete values[key]
      }
    }

    function trim(value) {
      sub(/^[[:space:]]+/, "", value)
      sub(/[[:space:]]+$/, "", value)
      return value
    }

    function split_markdown_row(line, cells,    i, c, depth, cell, count) {
      clear_array(cells)
      if (substr(line, 1, 1) != "|") {
        return 0
      }

      depth = 0
      cell = ""
      count = 0
      for (i = 2; i <= length(line); i++) {
        c = substr(line, i, 1)
        if (c == "[") {
          depth++
        } else if (c == "]") {
          if (depth > 0) {
            depth--
          }
        }

        if (c == "|" && depth == 0) {
          count++
          cells[count] = cell
          cell = ""
        } else {
          cell = cell c
        }
      }

      if (cell != "") {
        count++
        cells[count] = cell
      }
      return count
    }

    function replace_interval_pipes(value,    i, c, depth, out) {
      depth = 0
      out = ""
      for (i = 1; i <= length(value); i++) {
        c = substr(value, i, 1)
        if (c == "[") {
          depth++
          out = out c
        } else if (c == "]") {
          if (depth > 0) {
            depth--
          }
          out = out c
        } else if (c == "|" && depth > 0) {
          out = out "/"
        } else {
          out = out c
        }
      }
      return out
    }

    function print_markdown_row(cells, count,    i, out) {
      out = "|"
      for (i = 1; i <= count; i++) {
        out = out cells[i] "|"
      }
      print out
    }

    {
      if (substr($0, 1, 1) != "|") {
        clear_array(interval_columns)
        interval_column_count = 0
        print
        next
      }

      cell_count = split_markdown_row($0, cells)
      if (cell_count == 0) {
        print
        next
      }

      ref_interval_column = cell_count - 6
      cmp_interval_column = cell_count - 5
      found_interval_columns = \
        ref_interval_column > 0 && \
        trim(cells[ref_interval_column]) == "Ref [Lo | Ce | Hi]" && \
        trim(cells[cmp_interval_column]) == "Cmp [Lo | Ce | Hi]" && \
        trim(cells[cell_count - 4]) == "Ref Noise" && \
        trim(cells[cell_count - 3]) == "Cmp Noise" && \
        trim(cells[cell_count - 2]) == "Reason" && \
        trim(cells[cell_count - 1]) == "Change" && \
        trim(cells[cell_count]) == "Status"

      if (found_interval_columns) {
        clear_array(interval_columns)
        interval_columns[ref_interval_column] = 1
        interval_columns[cmp_interval_column] = 1
        interval_column_count = 2
        cells[ref_interval_column] = replace_interval_pipes(cells[ref_interval_column])
        cells[cmp_interval_column] = replace_interval_pipes(cells[cmp_interval_column])
        print_markdown_row(cells, cell_count)
        next
      }

      if (interval_column_count > 0) {
        for (i in interval_columns) {
          if (i <= cell_count) {
            cells[i] = replace_interval_pipes(cells[i])
          }
        }
        print_markdown_row(cells, cell_count)
        next
      }

      print
    }
  ' "${report_path}" > "${tmp_path}"; then
    rm -f "${tmp_path}"
    return 1
  fi

  mv "${tmp_path}" "${report_path}"
}

run_compare_command_impl() {
  local label="$1"
  local compare_out="$2"
  local compare_log="$3"
  shift 3

  local -a env_prefixes=()
  while [[ "$#" -gt 0 && "$1" == --env ]]; do
    shift
    env_prefixes+=("$1")
    shift
  done

  local started_at=0
  local elapsed_s=0
  local rc=0
  local stderr_tmp=""
  stderr_tmp="$(mktemp "${compare_log}.stderr.XXXXXX")"

  : > "${compare_out}"
  : > "${compare_log}"
  echo "::group::${label}"

  local -a print_env_args=()
  local env_prefix=""
  for env_prefix in "${env_prefixes[@]}"; do
    print_env_args+=(--env "${env_prefix}")
  done
  print_shell_command "${print_env_args[@]}" "$@"

  started_at="${SECONDS}"
  if [[ "${#env_prefixes[@]}" -gt 0 ]]; then
    if env "${env_prefixes[@]}" "$@" > "${compare_out}" 2> "${stderr_tmp}"; then
      rc=0
    else
      rc=$?
    fi
  elif "$@" > "${compare_out}" 2> "${stderr_tmp}"; then
    rc=0
  else
    rc=$?
  fi
  elapsed_s=$((SECONDS - started_at))
  if [[ -s "${compare_out}" ]]; then
    cat "${compare_out}"
    cat "${compare_out}" >> "${compare_log}"
  fi
  if [[ -s "${stderr_tmp}" ]]; then
    cat "${stderr_tmp}" >&2
    cat "${stderr_tmp}" >> "${compare_log}"
  fi
  rm -f "${stderr_tmp}"
  if [[ "${rc}" -ne 0 && ! -s "${compare_out}" ]]; then
    printf "_Compare command failed with rc=%s; see \`%s\` for stderr/log output._\n" \
      "${rc}" \
      "${compare_log#"${artifact_dir}"/}" \
      > "${compare_out}"
  fi
  echo "::endgroup::"
  if [[ "${rc}" -eq 0 ]]; then
    echo "${label} completed in ${elapsed_s}s"
  else
    echo "${label} failed in ${elapsed_s}s (rc=${rc})"
  fi
  return "${rc}"
}

run_compare_command() {
  run_compare_command_impl "$@"
}

run_compare_command_with_pythonpath() {
  local label="$1"
  local compare_pythonpath="$2"
  local compare_out="$3"
  local compare_log="$4"
  shift 4

  run_compare_command_impl \
    "${label}" \
    "${compare_out}" \
    "${compare_log}" \
    --env "PYTHONPATH=${compare_pythonpath}" \
    "$@"
}

run_compare_target() {
  local target="$1"
  local compare_script_dir="$2"
  local robust_compare_script="$3"
  local legacy_compare_script="$4"
  local base_json="$5"
  local test_json="$6"

  local compare_pythonpath="${compare_script_dir}${PYTHONPATH:+:${PYTHONPATH}}"
  local compare_python="${CCCL_BENCH_COMPARE_PYTHON:-python3}"
  local display=""
  local compare_out=""
  local compare_log=""
  local robust_rc=0
  local -a compare_cmd

  for display in "${COMPARE_DISPLAYS[@]}"; do
    compare_out="$(compare_report_path_for_display "${target}" "${display}")"
    compare_log="$(compare_log_path_for_display "${target}" "${display}")"
    compare_cmd=(
      "${compare_python}"
      "${robust_compare_script}"
      "${NVBENCH_COMPARE_ARGS[@]}"
      --display
      "${display}"
      "${base_json}"
      "${test_json}"
    )

    if ! run_compare_command_with_pythonpath \
      "[compare:${display}] ${target}" \
      "${compare_pythonpath}" \
      "${compare_out}" \
      "${compare_log}" \
      "${compare_cmd[@]}"; then
      robust_rc=1
    fi
    if [[ "${display}" == "explain" ]] && ! normalize_explain_markdown_report "${compare_out}"; then
      robust_rc=1
    fi
  done

  if [[ -n "${legacy_compare_script}" ]]; then
    compare_cmd=(
      "${compare_python}"
      "${legacy_compare_script}"
      "${NVBENCH_COMPARE_LEGACY_ARGS[@]}"
      "${base_json}"
      "${test_json}"
    )
    if ! run_compare_command_with_pythonpath \
      "[compare:legacy] ${target}" \
      "${compare_pythonpath}" \
      "${artifact_dir}/compare/${target}.legacy.md" \
      "${artifact_dir}/logs/compare.${target}.legacy.log" \
      "${compare_cmd[@]}"; then
      echo "Legacy compare output is diagnostic; continuing after legacy compare failure." >&2
    fi
  fi

  return "${robust_rc}"
}

parse_quoted_args_to_nul_file() {
  local quoted_args="$1"
  local output_file="$2"
  local option_name="$3"

  QUOTED_ARGS="${quoted_args}" OPTION_NAME="${option_name}" python3 - "${output_file}" <<'PY'
import os
import shlex
import sys

quoted = os.environ.get("QUOTED_ARGS", "")
option_name = os.environ.get("OPTION_NAME", "--args")
try:
    parsed = shlex.split(quoted)
except ValueError as exc:
    print(f"Invalid {option_name}: {exc}", file=sys.stderr)
    raise SystemExit(2)

with open(sys.argv[1], "wb") as out_file:
    for arg in parsed:
        out_file.write(arg.encode("utf-8"))
        out_file.write(b"\0")
PY
}

parse_quoted_args_to_array() {
  local -n _target_array_ref="$1"
  local quoted_args="$2"
  local option_label="$3"
  local parsed_args_file=""

  _target_array_ref=()
  [[ -n "${quoted_args}" ]] || return 0

  parsed_args_file="$(mktemp "/tmp/cccl-parsed-args-XXXXXX")"
  if ! parse_quoted_args_to_nul_file "${quoted_args}" "${parsed_args_file}" "${option_label}"; then
    rm -f "${parsed_args_file}"
    return 2
  fi
  mapfile -d '' -t _target_array_ref < "${parsed_args_file}"
  rm -f "${parsed_args_file}"
}

parse_nvbench_args() {
  declare -ga NVBENCH_RUN_ARGS NVBENCH_COMPARE_ARGS NVBENCH_COMPARE_LEGACY_ARGS
  parse_quoted_args_to_array NVBENCH_RUN_ARGS "${NVBENCH_ARGS_STRING}" "--nvbench-args" \
    || die "Failed to parse --nvbench-args."
  parse_quoted_args_to_array NVBENCH_COMPARE_ARGS "${NVBENCH_COMPARE_ARGS_STRING}" "--nvbench-compare-args" \
    || die "Failed to parse --nvbench-compare-args."
  parse_quoted_args_to_array \
    NVBENCH_COMPARE_LEGACY_ARGS \
    "${NVBENCH_COMPARE_LEGACY_ARGS_STRING}" \
    "--nvbench-compare-legacy-args" \
    || die "Failed to parse --nvbench-compare-legacy-args."
}

write_compare_report_details() {
  local summary="$1"
  local compare_report_file="$2"

  if [[ ! -s "${compare_report_file}" ]]; then
    return 1
  fi

  echo
  echo "<details><summary>${summary}</summary>"
  echo
  cat "${compare_report_file}"
  echo
  echo "</details>"
  return 0
}

write_robust_summary_excerpt() {
  local compare_report_file="$1"

  if [[ ! -s "${compare_report_file}" ]]; then
    return 1
  fi

  awk '
    function trim(value) {
      sub(/^[[:space:]]+/, "", value)
      sub(/[[:space:]]+$/, "", value)
      return value
    }

    {
      if (!printing && trim($0) == "# Summary") {
        print "**Robust comparison summary**"
        print ""
        printing = 1
        skip_leading_blanks = 1
        emitted = 1
        next
      }
      if (printing) {
        if (skip_leading_blanks && trim($0) == "") {
          next
        }
        skip_leading_blanks = 0
        print
      }
    }

    END { exit emitted ? 0 : 1 }
  ' "${compare_report_file}"
}

write_benchmark_arguments() {
  echo "## Benchmark Arguments"
  if [[ -n "${NVBENCH_ARGS_STRING}" ]]; then
    echo "- NVBench args: \`${NVBENCH_ARGS_STRING}\`"
  else
    echo "- NVBench args: not specified"
  fi
  if [[ -n "${NVBENCH_COMPARE_ARGS_STRING}" ]]; then
    echo "- NVBench robust compare args: \`${NVBENCH_COMPARE_ARGS_STRING}\`"
  else
    echo "- NVBench robust compare args: not specified"
  fi
  if [[ -n "${NVBENCH_COMPARE_LEGACY_ARGS_STRING}" ]]; then
    echo "- NVBench legacy compare args: \`${NVBENCH_COMPARE_LEGACY_ARGS_STRING}\`"
  else
    echo "- NVBench legacy compare args: not specified"
  fi
  echo
}

# Returns 1 when the target has no compare reports.
write_target_report() {
  local target_name="$1"
  local heading="$2"
  local display=""
  local rc=1

  echo
  echo "### ${heading}"
  write_robust_summary_excerpt "$(compare_report_path_for_display "${target_name}" "intervals")" || true
  for display in "${COMPARE_DISPLAYS[@]}"; do
    if write_compare_report_details \
      "Robust ${display} output" \
      "$(compare_report_path_for_display "${target_name}" "${display}")"; then
      rc=0
    fi
  done
  if write_compare_report_details "Legacy output" "${artifact_dir}/compare/${target_name}.legacy.md"; then
    rc=0
  fi
  return "${rc}"
}

run_compare_targets() {
  local base_build_dir="$1"
  local test_build_dir="$2"
  shift 2
  local target=""
  local robust_compare_script=""
  local compare_script_dir=""
  local legacy_compare_script=""
  local base_run_rc=0
  local test_run_rc=0

  robust_compare_script="$(resolve_robust_compare_script "${test_build_dir}")"
  if [[ -z "${robust_compare_script}" ]]; then
    robust_compare_script="$(resolve_robust_compare_script "${base_build_dir}")"
  fi
  if [[ -z "${robust_compare_script}" ]]; then
    die "Unable to locate nvbench_compare_robust.py in build dependencies." 1
  fi
  compare_script_dir="$(dirname "${robust_compare_script}")"

  legacy_compare_script="${compare_script_dir}/nvbench_compare.py"
  if [[ ! -f "${legacy_compare_script}" ]]; then
    legacy_compare_script=""
  fi

  for target in "$@"; do
    if run_target_for_side \
      "base" \
      "${base_build_dir}" \
      "${target}" \
      "${artifact_dir}/base/${target}.json" \
      "${artifact_dir}/base/${target}.md" \
      "${artifact_dir}/logs/run.base.${target}.log"; then
      base_run_rc=0
    else
      base_run_rc=$?
      any_failures=1
    fi

    if run_target_for_side \
      "test" \
      "${test_build_dir}" \
      "${target}" \
      "${artifact_dir}/test/${target}.json" \
      "${artifact_dir}/test/${target}.md" \
      "${artifact_dir}/logs/run.test.${target}.log"; then
      test_run_rc=0
    else
      test_run_rc=$?
      any_failures=1
    fi

    if [[ "${base_run_rc}" -eq 0 && "${test_run_rc}" -eq 0 ]]; then
      compares_attempted=$((compares_attempted + 1))
      if run_compare_target \
        "${target}" \
        "${compare_script_dir}" \
        "${robust_compare_script}" \
        "${legacy_compare_script}" \
        "${artifact_dir}/base/${target}.json" \
        "${artifact_dir}/test/${target}.json"; then
        compares_succeeded=$((compares_succeeded + 1))
      else
        any_failures=1
      fi
    fi
  done
}
