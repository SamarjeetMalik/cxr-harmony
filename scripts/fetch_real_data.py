"""Fetch the public corpora used for real-data evaluation.

Neither corpus is redistributed with this repository. Open-i is CC BY-NC-ND,
which forbids derivatives, and UNIFESP is CC BY-NC-SA, whose share-alike term
would conflict with this project's MIT licence. So the data is fetched on demand
and only the measured results are committed.

  python scripts/fetch_real_data.py --openi          # 1.1 MB, no account needed
  python scripts/fetch_real_data.py --unifesp        # 268 MB, needs a Kaggle key

Neither is required to run the pipeline or the test suite.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

OPENI_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"
UNIFESP_REF = "felipekitamura/unifesp-xray-bodypart-classification"

ROOT = Path(__file__).resolve().parents[1]
REALDATA = ROOT / "realdata"


def fetch_openi() -> Path:
    """Open-i / Indiana University reports: 3,955 records with MeSH annotations."""
    REALDATA.mkdir(parents=True, exist_ok=True)
    archive = REALDATA / "NLMCXR_reports.tgz"
    target = REALDATA / "ecgen-radiology"

    if target.exists() and any(target.glob("*.xml")):
        print(f"already present: {target}")
        return target

    print(f"downloading {OPENI_URL} ...")
    urllib.request.urlretrieve(OPENI_URL, archive)  # noqa: S310 - fixed NLM URL
    print(f"  {archive.stat().st_size / 1e6:.1f} MB")

    with tarfile.open(archive) as tar:
        tar.extractall(REALDATA, filter="data")
    archive.unlink()

    count = len(list(target.rglob("*.xml")))
    print(f"extracted {count} reports to {target}")
    return target


def fetch_unifesp(n_images: int = 400) -> Path:
    """UNIFESP X-ray collection: real hospital DICOM, sampled.

    Extracted to flat filenames. The archive nests each object under its own study
    and series UID, which on Windows exceeds the 260-character path limit and
    fails mid-extraction.
    """
    out = REALDATA / "unifesp"
    images = out / "images"
    if images.exists() and len(list(images.glob("*.dcm"))) >= n_images:
        print(f"already present: {images}")
        return images

    out.mkdir(parents=True, exist_ok=True)
    print(f"downloading {UNIFESP_REF} via kaggle (268 MB) ...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", UNIFESP_REF, "-p", str(out), "-q"],
        capture_output=True,
        text=True,
    )
    archive = next(out.glob("*.zip"), None)
    if archive is None:
        print(result.stdout, result.stderr, file=sys.stderr)
        raise SystemExit(
            "download failed. A Kaggle API token at ~/.kaggle/kaggle.json is required."
        )

    images.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".dcm"))
        print(f"  archive holds {len(names)} DICOM objects; extracting {n_images}")
        for i, name in enumerate(names[:n_images]):
            (images / f"{i:04d}.dcm").write_bytes(zf.read(name))
    archive.unlink()

    print(f"extracted {len(list(images.glob('*.dcm')))} objects to {images}")
    return images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openi", action="store_true", help="Open-i reports (1.1 MB)")
    parser.add_argument("--unifesp", action="store_true", help="UNIFESP DICOM (268 MB)")
    parser.add_argument("--n-images", type=int, default=400)
    args = parser.parse_args()

    if not (args.openi or args.unifesp):
        parser.error("choose --openi and/or --unifesp")

    if args.openi:
        fetch_openi()
    if args.unifesp:
        fetch_unifesp(args.n_images)


if __name__ == "__main__":
    main()
