#!/usr/bin/env bash

set -euo pipefail

bench_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ci/bench/common.sh
source "${bench_dir}/common.sh"

usage() {
  cat <<EOF
Usage: $0 <base-build-dir> <test-build-dir> <target> [<target> ...] \
  [--nvbench-args "<args>"] \
  [--nvbench-compare-args "<args>"] \
  [--nvbench-compare-legacy-args "<args>"]

Compare benchmark performance between two existing build trees.
Runs bin/<target> from both trees and compares the results. Nothing is configured
or built. The compare scripts come from the NVBench sources in the build trees.

Arguments:
  <base-build-dir>  Baseline build tree.
  <test-build-dir>  Comparison build tree.
  <target>          Benchmark executable name in bin/ (repeatable).

Options:
  --nvbench-args <args>     Extra args passed to benchmark binaries.
  --nvbench-compare-args <args>  Extra args passed to nvbench-compare-robust.
  --nvbench-compare-legacy-args <args>
                             Extra args passed to nvbench-compare-legacy.

Environment:
  CCCL_BENCH_ARTIFACT_ROOT   Root directory for outputs.
                             Default: "\$(pwd)/bench-artifacts"
  CCCL_BENCH_ARTIFACT_TAG    Optional explicit artifact directory name.
  CCCL_BENCH_GPU_NAME        Optional GPU name included in artifact directory names.
  CCCL_BENCH_COMPARE_PYTHON  Optional Python executable used for comparisons.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

NVBENCH_ARGS_STRING=""
NVBENCH_COMPARE_ARGS_STRING=""
NVBENCH_COMPARE_LEGACY_ARGS_STRING=""
positional=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --nvbench-args)
      [[ "$#" -ge 2 ]] || die "Missing value for --nvbench-args"
      NVBENCH_ARGS_STRING="$2"
      shift 2
      ;;
    --nvbench-compare-args)
      [[ "$#" -ge 2 ]] || die "Missing value for --nvbench-compare-args"
      NVBENCH_COMPARE_ARGS_STRING="$2"
      shift 2
      ;;
    --nvbench-compare-legacy-args)
      [[ "$#" -ge 2 ]] || die "Missing value for --nvbench-compare-legacy-args"
      NVBENCH_COMPARE_LEGACY_ARGS_STRING="$2"
      shift 2
      ;;
    -*)
      die "Unknown option: $1"
      ;;
    *)
      positional+=("$1")
      shift
      ;;
  esac
done

if [[ "${#positional[@]}" -lt 3 ]]; then
  usage >&2
  exit 2
fi

base_build_dir="$(realpath "${positional[0]}")"
test_build_dir="$(realpath "${positional[1]}")"
targets=("${positional[@]:2}")

parse_nvbench_args

timestamp="$(date -u +'%Y%m%dT%H%M%SZ')"
artifact_root="${CCCL_BENCH_ARTIFACT_ROOT:-$(pwd)/bench-artifacts}"
gpu_tag="${CCCL_BENCH_GPU_NAME:+$(sanitize_label "${CCCL_BENCH_GPU_NAME}")-}"
artifact_tag="$(sanitize_label "${CCCL_BENCH_ARTIFACT_TAG:-bench-${gpu_tag}${timestamp}}")"
artifact_dir="${artifact_root}/${artifact_tag}"

for subdir in base compare logs test; do
  mkdir -p "${artifact_dir}/${subdir}"
done

echo "Artifact directory: ${artifact_dir}"
echo "Base build dir: ${base_build_dir}"
echo "Test build dir: ${test_build_dir}"

any_failures=0
compares_attempted=0
compares_succeeded=0

run_compare_targets "${base_build_dir}" "${test_build_dir}" "${targets[@]}"

summary_file="${artifact_dir}/summary.md"
reports_emitted=0
{
  echo "# Benchmark Comparison Summary"
  echo
  echo "- Timestamp (UTC): ${timestamp}"
  echo "- GPU name: ${CCCL_BENCH_GPU_NAME:-not specified}"
  echo "- Base build dir: \`${base_build_dir}\`"
  echo "- Test build dir: \`${test_build_dir}\`"
  echo "- Targets selected: ${#targets[@]}"
  echo "- Comparisons attempted: ${compares_attempted}"
  echo "- Comparisons succeeded: ${compares_succeeded}"
  echo "- Artifact directory: \`${artifact_dir}\`"
  echo

  write_benchmark_arguments

  echo "## Compare Reports"
  for target in "${targets[@]}"; do
    if write_target_report "${target}" "\`${target}\`"; then
      reports_emitted=$((reports_emitted + 1))
    fi
  done

  if [[ "${reports_emitted}" -eq 0 ]]; then
    echo
    echo "_No per-target compare reports were produced._"
  fi
} > "${summary_file}"

echo "Wrote summary: ${summary_file}"
echo
cat "${summary_file}"

if [[ "${any_failures}" -ne 0 ]]; then
  exit 1
fi
