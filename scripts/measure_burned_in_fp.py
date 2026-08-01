"""Put a number on the burned-in detector's false positives.

The README has said "it also placed occasional small spurious boxes over
high-contrast spine anatomy". That is honest and useless: *occasional* is not a
measurement, and a reader has no way to tell whether it means two boxes or two
hundred.

Measuring it properly needs ground truth — which regions of a real hospital film
genuinely hold burned-in text — and that does not exist for the UNIFESP archive.
So this uses OCR as an **independent second opinion**: a region the detector
proposed and a text recogniser reads as characters is very likely genuine text; a
region it reads as nothing is a *candidate* false positive.

**This produces an upper bound, not a rate, and the distinction is the whole
point.** OCR fails on real text too — small glyphs, low contrast, rotation,
languages its model does not cover. Every such failure is counted here as a
suspected false positive when it is really a true positive the reader missed. So
the true false-positive rate is *at most* what this reports, and probably lower.
Reporting the bound as though it were the rate would be the same error as
reporting one benchmark run as though it were the throughput.

The other direction cannot be measured at all from here: regions holding text the
detector never proposed are invisible to a method that only inspects proposals.
This says nothing about recall, and does not pretend to.

Usage:
    python scripts/measure_burned_in_fp.py --src realdata/unifesp/images
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cxr_harmony.deid.ocr import (  # noqa: E402
    TextCategory,
    available,
    describe_redactions,
)
from cxr_harmony.deid.photometric import normalise_photometric  # noqa: E402
from cxr_harmony.deid.pixels import detect_text_regions  # noqa: E402

#: Fixed so the digests are stable across runs. This measurement never leaves the
#: machine and the images are not redistributed, so a constant is acceptable here
#: and nowhere else.
MEASUREMENT_KEY = b"burned-in-fp-measurement-key!!!!"

RESULTS = Path(__file__).resolve().parents[1] / "docs" / "results" / "burned_in_fp.json"


def measure(src: Path) -> dict:
    paths = sorted(src.glob("*.dcm"))
    if not paths:
        raise SystemExit(f"no .dcm files under {src}")
    if not available():
        raise SystemExit(
            "tesseract is required for this measurement. Install it with "
            '`pip install -e ".[ocr]"` plus the binary, then rerun.'
        )

    objects_with_detections = 0
    categories: Counter[str] = Counter()
    confidences: list[float] = []
    areas_unreadable: list[float] = []
    areas_readable: list[float] = []
    total_regions = 0

    for path in paths:
        ds = pydicom.dcmread(path)
        if "PixelData" not in ds:
            continue
        # Match the pipeline exactly: 372 of these 400 objects are MONOCHROME1,
        # and detecting on an un-normalised array would measure a different
        # detector than the one that ships.
        normalise_photometric(ds)
        image = ds.pixel_array

        regions = detect_text_regions(image)
        if not regions:
            continue
        objects_with_detections += 1
        total_regions += len(regions)

        frame_area = float(image.shape[0] * image.shape[1]) or 1.0
        for region, audit in zip(
            regions,
            describe_redactions(image, regions, key=MEASUREMENT_KEY),
            strict=False,
        ):
            categories[audit.category.value] += 1
            area_pct = 100.0 * (region.width * region.height) / frame_area
            if audit.category is TextCategory.UNREADABLE:
                areas_unreadable.append(area_pct)
            else:
                areas_readable.append(area_pct)
                confidences.append(audit.confidence)

    unreadable = categories.get(TextCategory.UNREADABLE.value, 0)
    return {
        "corpus": "real (UNIFESP hospital archive)",
        "n_objects": len(paths),
        "objects_with_detections": objects_with_detections,
        "n_regions": total_regions,
        "regions_read_as_text": total_regions - unreadable,
        "regions_read_as_nothing": unreadable,
        "false_positive_upper_bound_pct": (
            round(100.0 * unreadable / total_regions, 1) if total_regions else 0.0
        ),
        "categories": dict(categories.most_common()),
        "mean_confidence_where_read": (
            round(statistics.mean(confidences), 1) if confidences else 0.0
        ),
        "median_area_pct_read_as_text": (
            round(statistics.median(areas_readable), 3) if areas_readable else 0.0
        ),
        "median_area_pct_read_as_nothing": (
            round(statistics.median(areas_unreadable), 3) if areas_unreadable else 0.0
        ),
        "interpretation": (
            "An UPPER BOUND on the false-positive rate, not the rate. A region "
            "read as nothing is a *candidate* false positive: OCR also fails on "
            "genuine text that is small, low-contrast, rotated, or in a script "
            "its model does not cover, and every such failure inflates this "
            "figure. The true rate is at most this and probably lower. Says "
            "nothing about recall: regions the detector never proposed cannot be "
            "inspected by a method that only examines proposals."
        ),
    }


def render(result: dict) -> str:
    lines = [
        f"objects              : {result['n_objects']}",
        f"  with detections    : {result['objects_with_detections']}",
        f"regions proposed     : {result['n_regions']}",
        f"  read as text       : {result['regions_read_as_text']}",
        f"  read as nothing    : {result['regions_read_as_nothing']}",
        "",
        f"false-positive UPPER BOUND: {result['false_positive_upper_bound_pct']}%",
        "  (not the rate - OCR also fails on genuine text; see interpretation)",
        "",
        "categories:",
    ]
    for name, count in result["categories"].items():
        lines.append(f"  {name:<20} {count:>5}")
    lines += [
        "",
        f"median area, read as text   : {result['median_area_pct_read_as_text']}%",
        f"median area, read as nothing: {result['median_area_pct_read_as_nothing']}%",
        "  (a much smaller median for unread regions is what a spurious-box",
        "   population looks like: small boxes over high-contrast anatomy)",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("realdata/unifesp/images"))
    parser.add_argument("--json-out", type=Path, default=RESULTS)
    args = parser.parse_args()

    result = measure(args.src)
    print(render(result))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
