"""Guard the committed figures against the drift that hit the committed prose.

``docs/RESULTS.md`` claims every figure is regenerated from the JSON under
``docs/results/``. Until now nothing checked it. Editing a results file without
rerunning ``make figures`` left a committed PNG plotting superseded numbers, and
a stale plot is *harder* to catch by eye than a stale number in a table — nobody
reads a bar chart against a JSON.

Two checks at different strengths, because they are not equally portable:

``test_figures_were_built_from_the_current_results``
    Compares digests of the *inputs*. No rendering, so it is exact everywhere —
    CI, a reviewer's laptop, a container. This is the check that matters.

``test_figures_are_byte_identical_when_the_renderer_matches``
    Re-renders and compares the PNGs. Only meaningful against the matplotlib and
    FreeType that drew them: font rasterisation differs between FreeType builds,
    so a byte comparison across versions fails on pixels nobody changed. It skips
    loudly rather than failing wrongly, because a test that cries wolf on a
    version bump gets deleted, and then nothing checks rendering at all.

Between them: the first catches "the data moved and the figures did not", the
second catches "a figure was edited by hand, or a plotting change altered output
nobody re-examined".
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures"
RESULTS = ROOT / "docs" / "results"
MANIFEST = FIGURES / "MANIFEST.json"

pytestmark = pytest.mark.skipif(
    not MANIFEST.exists(),
    reason="docs/figures/MANIFEST.json absent; run `make figures` to create it",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_figures_were_built_from_the_current_results():
    """Every results JSON must be the one the committed figures were drawn from."""
    recorded = _manifest()["sources"]
    present = {p.name: _sha256(p) for p in sorted(RESULTS.glob("*.json"))}

    stale = sorted(n for n, d in present.items() if n in recorded and recorded[n] != d)
    missing = sorted(set(recorded) - set(present))
    unrecorded = sorted(set(present) - set(recorded))

    assert not stale, (
        f"{stale} changed since the figures were last built, so the committed plots "
        "show superseded numbers. Run `make figures` and commit the result. This is "
        "the same drift that put a retired throughput figure in three documents; a "
        "chart hides it better than a table does."
    )
    assert not missing, (
        f"{missing} were used to build the committed figures and are now gone. Either "
        "restore them or rebuild the figures without them."
    )
    assert not unrecorded, (
        f"{unrecorded} exist but no committed figure was built from them. Either they "
        "are new results with no figure yet, or `make figures` has not been run since "
        "they appeared."
    )


def _renderer_matches() -> tuple[bool, str]:
    import matplotlib

    recorded = _manifest()["renderer"]
    current = {
        "matplotlib": matplotlib.__version__,
        "freetype": matplotlib.ft2font.__freetype_version__,
    }
    if current == recorded:
        return True, ""
    return False, f"renderer differs: built with {recorded}, running {current}"


def test_figures_are_byte_identical_when_the_renderer_matches(tmp_path):
    """Re-render and compare, where a byte comparison is actually valid."""
    matches, why = _renderer_matches()
    if not matches:
        pytest.skip(
            f"{why}. The source-digest check still applies and is the binding one; "
            "pixel comparison is only meaningful against the renderer that drew them."
        )

    spec = importlib.util.spec_from_file_location(
        "_figs", ROOT / "scripts" / "make_results_figures.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["_figs"] = module
    try:
        spec.loader.exec_module(module)
        module.FIGURES = tmp_path  # render somewhere the repo cannot be dirtied
        module.main()

        committed = {p.name: _sha256(p) for p in sorted(FIGURES.glob("results_*.png"))}
        rebuilt = {p.name: _sha256(p) for p in sorted(tmp_path.glob("results_*.png"))}
    finally:
        sys.modules.pop("_figs", None)

    assert set(rebuilt) == set(committed), (
        f"rebuilt {sorted(rebuilt)} but {sorted(committed)} is committed; the set of "
        "figures changed without the committed set being updated."
    )
    differing = sorted(n for n in committed if committed[n] != rebuilt[n])
    assert not differing, (
        f"{differing} differ from a fresh render on the same renderer. Either a "
        "plotting change altered output that was never re-examined, or a committed "
        "PNG was edited by hand rather than generated. Run `make figures`."
    )
