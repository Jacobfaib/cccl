#!/usr/bin/env python3
"""Run and summarize the independently built MGMN reduction benchmarks.

Each scenario is a separate NVBench executable, so NVBench cannot rank them against each
other. This script runs all four, reads their JSON output, and emits a per-size comparison
against the `cub_full_device` baseline.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import shutil
import subprocess
from typing import Any

SCENARIOS = (
    "cub_full_device",
    "cudax_host_nccl",
    "cub_green_atomic",
    "cudax_device_nccl",
)

BASELINE = SCENARIOS[0]

#! NVBench summary tags read out of each state. NVBench derives the bandwidth figures from
#! the element counts and byte volumes the benchmarks declare, and normalizes utilization
#! against the device's own peak bandwidth, so no peak-bandwidth table is needed here.
GPU_TIME_TAG = "nv/cold/time/gpu/mean"
GPU_NOISE_TAG = "nv/cold/time/gpu/stdev/relative"
BANDWIDTH_TAG = "nv/cold/bw/global/bytes_per_second"
UTILIZATION_TAG = "nv/cold/bw/global/utilization"
SAMPLES_TAG = "nv/cold/sample_size"


def parse_scenarios(value: str) -> tuple[str, ...]:
    """Select a subset of `SCENARIOS`, keeping their declared order.

    The order is what `write_markdown` relies on for its per-size ranking tie-break, and the
    baseline must stay first when it is present, so the selection is filtered out of
    `SCENARIOS` rather than built from the order the flag was given in.
    """
    requested = {item.strip() for item in value.split(",") if item.strip()}
    if unknown := requested - set(SCENARIOS):
        raise argparse.ArgumentTypeError(
            f"unknown scenario(s): {', '.join(sorted(unknown))}; "
            f"choose from {', '.join(SCENARIOS)}"
        )
    if not requested:
        raise argparse.ArgumentTypeError("no scenario selected")
    return tuple(s for s in SCENARIOS if s in requested)


def parse_sizes(value: str) -> list[int]:
    try:
        sizes = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "sizes must be a comma-separated list of integers"
        ) from error
    if not sizes or any(size <= 0 for size in sizes) or len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("sizes must be positive and unique")
    return sizes


def make_axis_override(args: argparse.Namespace) -> str | None:
    """Build the NVBench `-a` argument that overrides the `Elements` axis.

    Returns None when the defaults compiled into the benchmarks are wanted. The axis is
    declared power-of-two, so explicit sizes are passed as exponents and must therefore be
    exact powers of two.
    """
    if args.sizes:
        exponents = []
        for size in args.sizes:
            if size & (size - 1):
                raise ValueError(
                    f"size {size} is not a power of two; the Elements axis is pow2"
                )
            exponents.append(size.bit_length() - 1)
        return "Elements[pow2]=[" + ",".join(map(str, exponents)) + "]"
    if (
        args.min_elements_pow2 is None
        and args.max_elements_pow2 is None
        and args.stride is None
    ):
        return None
    low = args.min_elements_pow2 if args.min_elements_pow2 is not None else 28
    high = args.max_elements_pow2 if args.max_elements_pow2 is not None else 32
    stride = args.stride if args.stride is not None else 1
    if low <= 0 or high < low or stride < 1:
        raise ValueError("invalid element exponent range")
    return f"Elements[pow2]=[{low}:{high}:{stride}]"


def run(
    command: list[str],
    log: pathlib.Path,
    verbose: bool,
    dry_run: bool,
    env: dict[str, str] | None = None,
) -> int:
    print("== NCCL Environment")
    for name, value in env.items():
        if name.startswith("NCCL"):
            print(f"{name}={value}")
    if verbose or dry_run:
        print("+", " ".join(command))
    if dry_run:
        return 0
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            output.write(line)
            if verbose:
                print(line, end="")
        return process.wait()


def make_profile_command(
    ncu: str,
    report: pathlib.Path,
    benchmark_command: list[str],
    extra_args: list[str],
    profile_axis: str | None,
) -> list[str]:
    """Wrap one benchmark invocation in `ncu`.

    The profiled pass is a second, separate run: Nsight Compute serializes kernels and
    replays them, so its timings are not comparable to the measurement pass. The benchmark
    therefore runs with `--profile`, which makes NVBench execute each state one time only,
    and writes no JSON of its own.
    """
    command = [
        ncu,
        "--set",
        "full",
        "--target-processes",
        "all",
        "--export",
        str(report),
        "--force-overwrite",
    ]
    command += extra_args
    #! Drop `--json <path>` and `--min-time <value>`: the profiled pass supersedes neither
    #! the JSON nor the sampling loop of the measurement pass.
    benchmark: list[str] = []
    skip_next = False
    for item in benchmark_command:
        if skip_next:
            skip_next = False
            continue
        if item in ("--json", "--min-time", "--max-noise", "--timeout"):
            skip_next = True
            continue
        if item == "-a" and profile_axis is not None:
            skip_next = True
            continue
        benchmark.append(item)
    benchmark.append("--profile")
    if profile_axis is not None:
        benchmark += ["-a", profile_axis]
    return command + ["--"] + benchmark


def summary_value(state: dict[str, Any], tag: str) -> float | None:
    """Read the scalar `value` of the summary tagged `tag` out of one NVBench state.

    NVBench writes every summary datum as a `{name, type, value}` triple with the value
    rendered as a string, so the number is recovered here rather than taken as-is.
    """
    for summary in state.get("summaries", []):
        if summary.get("tag") != tag:
            continue
        for datum in summary.get("data", []):
            if datum.get("name") == "value":
                try:
                    return float(datum["value"])
                except (KeyError, TypeError, ValueError):
                    return None
    return None


def axis_value(state: dict[str, Any], name: str) -> int | None:
    for value in state.get("axis_values", []):
        if value.get("name") == name:
            try:
                return int(float(value["value"]))
            except (KeyError, TypeError, ValueError):
                return None
    return None


def collect(document: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Reduce one NVBench JSON document to a measurement per element count."""
    measurements: dict[int, dict[str, Any]] = {}
    for benchmark in document.get("benchmarks", []):
        for state in benchmark.get("states", []):
            elements = axis_value(state, "Elements")
            if elements is None:
                continue
            if state.get("is_skipped"):
                measurements[elements] = {
                    "status": f"skipped ({state.get('skip_reason', 'no reason given')})",
                    "seconds": None,
                }
                continue
            seconds = summary_value(state, GPU_TIME_TAG)
            if seconds is None:
                continue
            measurements[elements] = {
                "status": "ok",
                "seconds": seconds,
                "noise": summary_value(state, GPU_NOISE_TAG),
                "samples": summary_value(state, SAMPLES_TAG),
                "gb_per_second": (
                    value / 1e9
                    if (value := summary_value(state, BANDWIDTH_TAG)) is not None
                    else None
                ),
                "utilization": summary_value(state, UTILIZATION_TAG),
            }
    return measurements


def summarize(
    results: dict[str, Any], log_dir: pathlib.Path, scenarios: tuple[str, ...]
) -> dict[str, Any]:
    per_scenario = {
        scenario: collect(result["json"])
        for scenario, result in results.items()
        if result.get("status") == "ok" and "json" in result
    }
    elements_seen = sorted(
        {elements for values in per_scenario.values() for elements in values}
    )

    # Speedups are relative to the baseline, which is not run when it was excluded. Fall back to
    # the fastest scenario present, so the column stays meaningful instead of reading `n/a`.
    rows = []
    for elements in elements_seen:
        row: dict[str, Any] = {"elements": elements, "input_bytes": elements * 4}
        for scenario in scenarios:
            row[scenario] = per_scenario.get(scenario, {}).get(
                elements,
                {"status": results[scenario]["status"], "seconds": None},
            )
        if BASELINE in scenarios:
            reference = row[BASELINE]["seconds"]
        else:
            measured = [
                row[s]["seconds"] for s in scenarios if row[s]["seconds"] is not None
            ]
            reference = min(measured) if measured else None
        for scenario in scenarios:
            measurement = row[scenario]
            measurement["latency_speedup"] = (
                reference / measurement["seconds"]
                if reference and measurement["seconds"]
                else None
            )
        rows.append(row)
    return {
        "log_dir": str(log_dir),
        "scenarios": results,
        "rows": rows,
        "selected": list(scenarios),
        "baseline": BASELINE if BASELINE in scenarios else None,
    }


SPEEDUP_BAR_WIDTH = 12
UTILIZATION_BAR_WIDTH = 20


def format_bytes(count: int) -> str:
    size = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size:.0f} B"
        size /= 1024
    raise AssertionError("unreachable")


def bar(fraction: float | None, width: int) -> str:
    """Render `fraction` of a bracketed track `width` characters wide.

    A `+` terminates bars that overshoot a whole character by at least half of one, which
    recovers most of the resolution a plain `#` scale would otherwise round away. Values
    are clamped, so a scenario faster than the baseline still reads as a full track.
    """
    if fraction is None:
        return "[" + "?".center(width) + "]"
    filled = max(0.0, min(1.0, fraction)) * width
    whole = int(filled)
    half = "+" if whole < width and filled - whole >= 0.5 else ""
    return "[" + ("#" * whole + half).ljust(width) + "]"


def write_markdown(summary: dict[str, Any], path: pathlib.Path) -> None:
    scenarios = tuple(summary["selected"])
    columns = [
        ("Scenario", "left", max(len(s) for s in scenarios)),
        ("GPU ms", "right", 9),
        ("GB/s", "right", 9),
        ("Speedup", "right", 7),
        ("BWUtil", "right", 6),
    ]
    header = "  ".join(
        title.ljust(width) if align == "left" else title.rjust(width)
        for title, align, width in columns
    )
    reference_label = "vs baseline" if summary["baseline"] else "vs fastest"
    header = (
        f"    {header}"
        f"  {reference_label.center(SPEEDUP_BAR_WIDTH + 2)}"
        f"  {'vs peak'.center(UTILIZATION_BAR_WIDTH + 2)}"
    )

    lines = [
        "# MGMN benchmark summary",
        "",
        "GPU times are NVBench cold-measurement means. BWUtil is NVBench's global memory",
        "bandwidth utilization, normalized against the device's own peak bandwidth.",
        "",
    ]
    if not summary["baseline"]:
        lines += [
            f"`{BASELINE}` was not run, so speedups are relative to the fastest scenario",
            "present rather than to the baseline.",
            "",
        ]
    lines += [
        "```",
        header,
        "  " + "-" * (len(header) + 2),
    ]

    for row in summary["rows"]:
        lines.append("")
        lines.append(f"  n = {row['elements']:,}  ({format_bytes(row['input_bytes'])})")
        # Fastest first, so the ranking at each size is apparent without comparing numbers.
        # Scenarios with no measurement sort last; ties keep their declaration order.
        ordered = sorted(
            scenarios,
            key=lambda scenario: (
                row[scenario]["seconds"] is None,
                row[scenario]["seconds"] or 0.0,
            ),
        )
        # The speedup bar is scaled to the fastest scenario in the group rather than to the
        # baseline: a scenario that beats the baseline exceeds 1.0, and clamping every such case to
        # a full bar would render them indistinguishable from the baseline itself.
        speedups = [
            row[scenario]["latency_speedup"]
            for scenario in ordered
            if row[scenario]["latency_speedup"] is not None
        ]
        speedup_scale = max(speedups) if speedups else 1.0
        for scenario in ordered:
            value = row[scenario]
            if value["seconds"] is None:
                cells = [scenario, "n/a", "n/a", "n/a", "n/a"]
                bars = f"  {value['status']}"
            else:
                utilization = value.get("utilization")
                bandwidth = value.get("gb_per_second")
                # NVBench reports utilization as a fraction of peak, not a percentage.
                cells = [
                    scenario,
                    f"{value['seconds'] * 1e3:.3f}",
                    f"{bandwidth:.3f}" if bandwidth is not None else "n/a",
                    f"{value['latency_speedup']:.3f}",
                    f"{utilization * 100.0:.1f}%" if utilization is not None else "n/a",
                ]
                relative = value["latency_speedup"] / speedup_scale
                bars = (
                    f"  {bar(relative, SPEEDUP_BAR_WIDTH)}"
                    f"  {bar(utilization, UTILIZATION_BAR_WIDTH)}"
                )
            body = "  ".join(
                text.ljust(width) if align == "left" else text.rjust(width)
                for text, (_, align, width) in zip(cells, columns)
            )
            lines.append(f"    {body}{bars}")
    lines.append("```")

    profiles = [
        (scenario, result["profile"])
        for scenario, result in summary["scenarios"].items()
        if "profile" in result
    ]
    if profiles:
        lines += [
            "",
            "## Nsight Compute reports",
            "",
            "Recorded in a separate pass at one element count, with kernel replay enabled.",
            "These timings are not comparable to the table above.",
            "",
        ]
        for scenario, profile in profiles:
            lines.append(f"- `{scenario}`: {profile['status']} - `{profile['report']}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n")
    print(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", action="count", default=0)
    parser.add_argument(
        "--binary-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent.parent.parent
        / "build"
        / "preset-latest"
        / "benchmarks"
        / "mgmn",
    )
    parser.add_argument("--log-dir", type=pathlib.Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--scenarios",
        type=parse_scenarios,
        default=SCENARIOS,
        help="comma-separated subset of scenarios to run "
        f"(default: all of {', '.join(SCENARIOS)})",
    )
    # The benchmarks compile in a 2^20..2^28 sweep. These flags override that axis; leaving
    # them unset runs the compiled-in default.
    parser.add_argument("--min-elements-pow2", type=int)
    parser.add_argument("--max-elements-pow2", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument(
        "--sizes",
        type=parse_sizes,
        help="explicit element counts; each must be a power of two",
    )
    parser.add_argument("--min-time", type=float, default=0.5)
    parser.add_argument("--max-noise", type=float)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--dry-run", action="store_true")
    # Profiling is a second pass per scenario. Nsight Compute serializes and replays
    # kernels, so it must not run on top of the measurement pass.
    parser.add_argument(
        "--profile",
        action="store_true",
        help="record an Nsight Compute report per scenario in an extra pass",
    )
    parser.add_argument(
        "--ncu",
        default="ncu",
        help="Nsight Compute executable used by --profile",
    )
    parser.add_argument(
        "--ncu-args",
        default="",
        help="extra arguments passed to ncu, split on whitespace",
    )
    parser.add_argument(
        "--profile-size",
        type=int,
        help="single element count to profile; must be a power of two "
        "(default: the largest size of the measurement axis)",
    )
    args = parser.parse_args()

    axis = make_axis_override(args)
    ncu_args = args.ncu_args.split()
    # A full `--set full` collection replays every kernel many times, so the profiled pass
    # is restricted to one element count instead of the full sweep.
    profile_axis: str | None = None
    if args.profile:
        if args.profile_size is not None:
            if args.profile_size <= 0 or args.profile_size & (args.profile_size - 1):
                raise ValueError("--profile-size must be a positive power of two")
            exponent = args.profile_size.bit_length() - 1
        elif args.sizes:
            exponent = max(args.sizes).bit_length() - 1
        else:
            exponent = (
                args.max_elements_pow2 if args.max_elements_pow2 is not None else 28
            )
        profile_axis = f"Elements[pow2]=[{exponent}]"
        if not args.dry_run and shutil.which(args.ncu) is None:
            raise FileNotFoundError(f"missing Nsight Compute executable: {args.ncu}")

    root = pathlib.Path(__file__).resolve().parent
    log_dir = args.log_dir or root / "benchmark-results" / dt.datetime.now(
        dt.timezone.utc
    ).strftime("%Y_%m_%d_%H_%M_%S_UTC")
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "axis": axis,
        "device": args.device,
        "scenarios": list(args.scenarios),
        "commands": [],
        "profile_axis": profile_axis,
        "profile_commands": [],
    }
    results: dict[str, Any] = {}
    for scenario in args.scenarios:
        json_path = log_dir / f"{scenario}.json"
        executable = args.binary_dir / scenario
        if not args.dry_run and not executable.is_file():
            raise FileNotFoundError(f"missing benchmark executable: {executable}")
        # `CUDA_VISIBLE_DEVICES` already restricts the process to one GPU, which NVBench then
        # sees as device 0. Without `-d` NVBench would sweep every device it can see.
        command = [
            str(executable),
            "--json",
            str(json_path),
            "-d",
            "0",
            "--min-time",
            str(args.min_time),
        ]
        if axis is not None:
            command += ["-a", axis]
        if args.max_noise is not None:
            command += ["--max-noise", str(args.max_noise)]
        if args.timeout is not None:
            command += ["--timeout", str(args.timeout)]
        manifest["commands"].append(command)
        env = os.environ | {"CUDA_VISIBLE_DEVICES": str(args.device)}
        if "nccl" in scenario:
            env |= {
                "NCCL_MULTI_RANK_GPU_ENABLE": "1",
                "NCCL_NVLS_ENABLE": "0",
            }
        status = run(
            command, log_dir / f"{scenario}.log", args.v > 0, args.dry_run, env
        )
        results[scenario] = {
            "status": "ok" if status == 0 else f"failed ({status})",
            "log": str(log_dir / f"{scenario}.log"),
        }
        if status == 0 and not args.dry_run:
            results[scenario]["json"] = json.loads(
                json_path.read_text(encoding="utf-8")
            )
        if not args.profile:
            continue
        report = log_dir / f"{scenario}.ncu-rep"
        profile_command = make_profile_command(
            args.ncu, report, command, ncu_args, profile_axis
        )
        manifest["profile_commands"].append(profile_command)
        profile_status = run(
            profile_command,
            log_dir / f"{scenario}.ncu.log",
            args.v > 0,
            args.dry_run,
            env,
        )
        results[scenario]["profile"] = {
            "status": "ok" if profile_status == 0 else f"failed ({profile_status})",
            "report": str(report),
            "log": str(log_dir / f"{scenario}.ncu.log"),
        }
    (log_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    summary = (
        summarize(results, log_dir, args.scenarios)
        if not args.dry_run
        else {
            "log_dir": str(log_dir),
            "scenarios": results,
            "rows": [],
            "selected": list(args.scenarios),
            "baseline": BASELINE if BASELINE in args.scenarios else None,
        }
    )
    (log_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(summary, log_dir / "summary.md")
    print(log_dir / "summary.md")
    return 0 if all(value["status"] == "ok" for value in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
