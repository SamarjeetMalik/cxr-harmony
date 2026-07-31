"""Run the pipeline over a real DICOM archive and report what it found.

Used with the UNIFESP X-ray collection (Universidade Federal de Sao Paulo,
CC BY-NC-SA 4.0), a real hospital computed-radiography export. Two things it
demonstrates that synthetic data cannot:

* **Real archives are heterogeneous in ways nobody documents.** This one mixes
  MONOCHROME1 and MONOCHROME2 in a single delivery, with no flag to warn you.
* **Real archives arrive stripped.** BodyPartExamined, ViewPosition, PatientSex
  and StudyDate are all empty here, so the pipeline has to degrade gracefully
  rather than assume the tags it would like are populated.

It cannot demonstrate PHI removal: like every public collection, this one was
de-identified before release, so there is nothing left to remove. Evidence for
removal efficacy comes from the synthetic corpus, where ground truth exists.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cxr_harmony.deid import deidentify, verify_store  # noqa: E402
from cxr_harmony.ingest import ingest  # noqa: E402
from cxr_harmony.workspace import Workspace  # noqa: E402


def survey(directory: Path) -> dict:
    """What the archive actually contains, before anything touches it."""
    counters: dict[str, Counter] = {
        k: Counter()
        for k in (
            "Modality",
            "PhotometricInterpretation",
            "BodyPartExamined",
            "ViewPosition",
            "PatientSex",
            "BitsStored",
            "Manufacturer",
        )
    }
    sizes: Counter = Counter()
    n = 0
    for path in sorted(directory.rglob("*.dcm")):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception:  # noqa: BLE001
            continue
        n += 1
        for keyword, counter in counters.items():
            counter[str(getattr(ds, keyword, "") or "<empty>")] += 1
        sizes[f"{getattr(ds, 'Rows', '?')}x{getattr(ds, 'Columns', '?')}"] += 1
    return {"n": n, "fields": counters, "sizes": sizes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=Path("work-real"))
    args = parser.parse_args()

    print("=" * 72)
    print("SURVEY OF THE ARCHIVE AS RECEIVED")
    print("=" * 72)
    found = survey(args.src)
    print(f"objects: {found['n']}")
    for keyword, counter in found["fields"].items():
        top = ", ".join(f"{k}={v}" for k, v in counter.most_common(4))
        print(f"  {keyword:<26} {top}")
    sizes = ", ".join(f"{k}={v}" for k, v in found["sizes"].most_common(4))
    print(f"  {'image size':<26} {sizes}")

    # The delivery layout the pipeline expects: <site>/images/*.dcm
    staged = args.work / "incoming"
    site_dir = staged / "SITE_UNIFESP" / "images"
    site_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.src.rglob("*.dcm")):
        target = site_dir / path.name
        if not target.exists():
            target.write_bytes(path.read_bytes())

    ws = Workspace(args.work / "work").ensure()

    print()
    print("=" * 72)
    print("PIPELINE")
    print("=" * 72)
    ingested = ingest(staged, ws)
    print(f"ingest    accepted={ingested.n_accepted} quarantined={ingested.n_quarantined}")
    for reason, count in ingested.reasons().items():
        print(f"            {reason}: {count}")

    if not ingested.n_accepted:
        print("\nnothing accepted; stopping.")
        return

    result = deidentify(staged, ws, key=b"real-dicom-demonstration-key-32b")
    print(
        f"deid      objects={result.n_objects} "
        f"pixel_redacted={result.n_redacted} "
        f"photometric_converted={result.n_photometric_converted}"
    )

    after = Counter(r.photometric_interpretation for r in result.records)
    print(f"          photometric after: {dict(after)}")

    report = verify_store(ws.deid_store)
    print(f"verify    checked={report.n_checked} passed={report.passed} {report.by_kind()}")

    print()
    print("Note: this archive was de-identified by its publisher before release, so")
    print("the verification above confirms conformance, not removal efficacy. Evidence")
    print("for removal comes from the synthetic corpus, where ground truth exists.")


if __name__ == "__main__":
    main()
