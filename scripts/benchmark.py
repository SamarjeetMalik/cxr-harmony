"""Measure pipeline throughput against the project's stated target.

The project proposal sets an ingestion target of **>500 studies/hour**. This
script measures it, on both the synthetic corpus and a real hospital archive,
and writes the result to ``docs/results/benchmark.json``.

Both are reported because only reporting the synthetic figure would flatter the
result: real DICOM objects here are larger, carry more elements, and decode more
slowly than anything the generator produces. A throughput number quoted without
saying which corpus it came from is not a measurement.

Hardware is recorded alongside. Studies per hour is meaningless without the
machine attached — the same code on a workstation and a laptop differs by more
than the target's margin.

Usage:
    python scripts/benchmark.py --synthetic
    python scripts/benchmark.py --real realdata/unifesp/images
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cxr_harmony.deid import deidentify  # noqa: E402
from cxr_harmony.ingest import ingest  # noqa: E402
from cxr_harmony.synth import generate_corpus  # noqa: E402
from cxr_harmony.workspace import Workspace  # noqa: E402

#: From the project proposal's performance targets.
TARGET_STUDIES_PER_HOUR = 500

#: Recorded with every result, because the margin here is large enough to be
#: misleading without it.
SCOPE_CAVEAT = (
    "Measures local processing only: files already on disk, single process, no "
    "parallelism. It excludes network transfer from the partner site, DICOM "
    "C-STORE negotiation, and any queueing at the receiving end, which is where a "
    "real deployment's throughput ceiling almost certainly sits. The comfortable "
    "margin over target should be read as 'processing is not the bottleneck', not "
    "as 'ingestion runs 200x faster than required'."
)

BENCHMARK_KEY = b"benchmark-fixed-key-32-bytes!!!!"
RESULTS = Path(__file__).resolve().parents[1] / "docs" / "results" / "benchmark.json"


@dataclass
class StageTiming:
    stage: str
    n_studies: int
    seconds: float

    @property
    def studies_per_hour(self) -> float:
        return self.n_studies / self.seconds * 3600 if self.seconds > 0 else 0.0


@dataclass
class StageResult:
    """A stage measured over several repeats.

    Reported as a median with its range, never as a single run. Measured on this
    hardware, five identical runs over the same 400 objects spanned 71,097 to
    123,177 studies/hour — a **1.73x spread** with no code change between them.
    Two earlier single-run figures, 103,013 and 55,569, were both quoted as though
    they were the throughput; they were two draws from that distribution, and the
    apparent 46% "regression" between them was noise.

    ``meets_target`` is judged on the **slowest** run, not the median. A capacity
    claim that only holds on a good day is not a capacity claim.
    """

    stage: str
    n_studies: int
    samples: list[float]  # studies/hour, one per repeat

    @property
    def median(self) -> float:
        return statistics.median(self.samples)

    @property
    def slowest(self) -> float:
        return min(self.samples)

    @property
    def fastest(self) -> float:
        return max(self.samples)

    @property
    def spread(self) -> float:
        return self.fastest / self.slowest if self.slowest > 0 else 0.0

    @property
    def meets_target(self) -> bool:
        return self.slowest >= TARGET_STUDIES_PER_HOUR

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "n_studies": self.n_studies,
            "n_repeats": len(self.samples),
            "studies_per_hour_median": round(self.median, 1),
            "studies_per_hour_slowest": round(self.slowest, 1),
            "studies_per_hour_fastest": round(self.fastest, 1),
            "spread_ratio": round(self.spread, 2),
            "meets_target": self.meets_target,
        }


def hardware() -> dict:
    """Enough context that the number can be compared to another machine's."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "note": "single-process, single-threaded; no parallelism is used",
    }


def _time_once(src: Path, work: Path, label: str) -> list[StageTiming]:
    """One pass. Each repeat gets a fresh workspace so nothing is warm-cached."""
    ws = Workspace(work).ensure()

    start = time.perf_counter()
    ingested = ingest(src, ws)
    ingest_seconds = time.perf_counter() - start

    if ingested.n_accepted == 0:
        raise SystemExit(f"{label}: nothing accepted from {src}")

    start = time.perf_counter()
    deidentify(src, ws, key=BENCHMARK_KEY)
    deid_seconds = time.perf_counter() - start

    n = ingested.n_accepted
    return [
        StageTiming("ingest", n, ingest_seconds),
        StageTiming("deidentify", n, deid_seconds),
        StageTiming("ingest+deidentify", n, ingest_seconds + deid_seconds),
    ]


def _time_pipeline(src: Path, work_root: Path, label: str, repeats: int) -> list[StageResult]:
    """Repeat the run and keep every sample.

    Repeats exist because a single timing on a laptop is not a measurement: this
    machine produced a 1.73x spread across five identical runs. Reporting one of
    them as the throughput is how a noise excursion gets published as a
    regression.
    """
    per_stage: dict[str, list[float]] = {}
    n_studies = 0
    for attempt in range(repeats):
        for timing in _time_once(src, work_root / f"run{attempt}", label):
            per_stage.setdefault(timing.stage, []).append(timing.studies_per_hour)
            n_studies = timing.n_studies

    return [
        StageResult(stage=stage, n_studies=n_studies, samples=samples)
        for stage, samples in per_stage.items()
    ]


def benchmark_synthetic(n_patients: int, image_size: int, repeats: int) -> dict:
    """Throughput on the generated corpus, at the demo's own image size."""
    root = Path(tempfile.mkdtemp(prefix="cxrh-bench-synth-"))
    try:
        src = root / "incoming"
        truth = generate_corpus(
            src, seed=20260801, n_patients=n_patients, n_cross_site=4, image_size=image_size
        )
        timings = _time_pipeline(src, root / "work", "synthetic", repeats)
        return {
            "corpus": "synthetic",
            "n_studies": truth["n_studies"],
            "image_size": f"{image_size}x{image_size}",
            "stages": [t.to_dict() for t in timings],
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def benchmark_real(images: Path, repeats: int) -> dict:
    """Throughput on a real hospital archive.

    Files are staged into the delivery layout the pipeline expects. The staging
    copy is deliberately excluded from the timed section: it measures the disk,
    not the pipeline.
    """
    images = Path(images)
    root = Path(tempfile.mkdtemp(prefix="cxrh-bench-real-"))
    try:
        site_dir = root / "incoming" / "SITE_REAL" / "images"
        site_dir.mkdir(parents=True)
        sizes = []
        for path in sorted(images.glob("*.dcm")):
            shutil.copy2(path, site_dir / path.name)
            sizes.append(path.stat().st_size)

        if not sizes:
            raise SystemExit(f"no .dcm files under {images}")

        timings = _time_pipeline(root / "incoming", root / "work", "real", repeats)
        return {
            "corpus": "real (UNIFESP hospital archive)",
            "n_studies": len(sizes),
            "mean_file_bytes": int(sum(sizes) / len(sizes)),
            "stages": [t.to_dict() for t in timings],
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def render(result: dict) -> str:
    lines = [
        f"hardware: {result['hardware']['platform']}",
        f"          {result['hardware']['cpu_count']} logical CPUs, "
        f"{result['hardware']['note']}",
        f"target:   >{TARGET_STUDIES_PER_HOUR} studies/hour",
        "",
        "scope:    " + textwrap.fill(SCOPE_CAVEAT, width=78, subsequent_indent=" " * 10),
        "",
        f"{'corpus':<16} {'stage':<20} {'median/hr':>11} {'slowest':>10} "
        f"{'fastest':>10} {'spread':>8}",
        "-" * 92,
    ]
    for run in result["runs"]:
        short = "real archive" if run["corpus"].startswith("real") else "synthetic"
        for stage in run["stages"]:
            verdict = "PASS" if stage["meets_target"] else "BELOW TARGET"
            lines.append(
                f"{short:<16} {stage['stage']:<20} "
                f"{stage['studies_per_hour_median']:>11,.0f} "
                f"{stage['studies_per_hour_slowest']:>10,.0f} "
                f"{stage['studies_per_hour_fastest']:>10,.0f} "
                f"{stage['spread_ratio']:>7.2f}x  {verdict}"
            )
    lines += [
        "",
        f"Median of {result['repeats']} runs. The target is judged on the *slowest* run,",
        "not the median: a capacity claim that only holds on a good day is not one.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--real", type=Path, default=None)
    parser.add_argument("--patients", type=int, default=40)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Timed passes per corpus. One pass is not a measurement on a laptop.",
    )
    parser.add_argument("--json-out", type=Path, default=RESULTS)
    args = parser.parse_args()

    if not (args.synthetic or args.real):
        parser.error("choose --synthetic and/or --real PATH")

    runs = []
    if args.synthetic:
        print("benchmarking synthetic corpus ...", flush=True)
        runs.append(benchmark_synthetic(args.patients, args.image_size, args.repeats))
    if args.real:
        print(f"benchmarking real archive at {args.real} ...", flush=True)
        runs.append(benchmark_real(args.real, args.repeats))

    result = {
        "target_studies_per_hour": TARGET_STUDIES_PER_HOUR,
        "repeats": args.repeats,
        "scope_caveat": SCOPE_CAVEAT,
        "hardware": hardware(),
        "runs": runs,
    }
    print()
    print(render(result))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
