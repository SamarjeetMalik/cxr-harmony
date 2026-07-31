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
import sys
import tempfile
import textwrap
import time
from dataclasses import asdict, dataclass
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

    @property
    def meets_target(self) -> bool:
        return self.studies_per_hour >= TARGET_STUDIES_PER_HOUR

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "seconds": round(self.seconds, 3),
            "studies_per_hour": round(self.studies_per_hour, 1),
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


def _time_pipeline(src: Path, work: Path, label: str) -> list[StageTiming]:
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


def benchmark_synthetic(n_patients: int, image_size: int) -> dict:
    """Throughput on the generated corpus, at the demo's own image size."""
    root = Path(tempfile.mkdtemp(prefix="cxrh-bench-synth-"))
    try:
        src = root / "incoming"
        truth = generate_corpus(
            src, seed=20260801, n_patients=n_patients, n_cross_site=4, image_size=image_size
        )
        timings = _time_pipeline(src, root / "work", "synthetic")
        return {
            "corpus": "synthetic",
            "n_studies": truth["n_studies"],
            "image_size": f"{image_size}x{image_size}",
            "stages": [t.to_dict() for t in timings],
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def benchmark_real(images: Path) -> dict:
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

        timings = _time_pipeline(root / "incoming", root / "work", "real")
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
        f"{'corpus':<34} {'stage':<20} {'studies':>8} {'sec':>8} {'studies/hr':>12}  ",
        "-" * 92,
    ]
    for run in result["runs"]:
        for stage in run["stages"]:
            verdict = "PASS" if stage["meets_target"] else "BELOW TARGET"
            lines.append(
                f"{run['corpus']:<34} {stage['stage']:<20} {stage['n_studies']:>8} "
                f"{stage['seconds']:>8.2f} {stage['studies_per_hour']:>12,.0f}  {verdict}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--real", type=Path, default=None)
    parser.add_argument("--patients", type=int, default=40)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--json-out", type=Path, default=RESULTS)
    args = parser.parse_args()

    if not (args.synthetic or args.real):
        parser.error("choose --synthetic and/or --real PATH")

    runs = []
    if args.synthetic:
        print("benchmarking synthetic corpus ...", flush=True)
        runs.append(benchmark_synthetic(args.patients, args.image_size))
    if args.real:
        print(f"benchmarking real archive at {args.real} ...", flush=True)
        runs.append(benchmark_real(args.real))

    result = {
        "target_studies_per_hour": TARGET_STUDIES_PER_HOUR,
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
