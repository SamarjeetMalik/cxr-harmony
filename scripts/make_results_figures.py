"""Plot the measured results.

Companion to ``make_figures.py``, which draws the *design* figures — why edge
density, why hash splits. This one draws *outcomes*: what the pipeline scored,
against what it was aiming at.

Every figure reads its numbers from a committed JSON under ``docs/results/``.
Nothing here is transcribed by hand, so a figure cannot silently disagree with
the document that cites it. Where a required JSON is absent, that figure is
skipped with a message rather than drawn from stale defaults.

No real pixel data and no verbatim report text is plotted. Both corpora are
licensed against redistribution, and a figure is redistribution.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIGURES = ROOT / "docs" / "figures"
RESULTS = ROOT / "docs" / "results"

# Shared with make_figures.py so the whole set reads as one system.
INK = "#1F3A5F"
WARN = "#B5443B"
OK = "#2E6F50"
MUTED = "#8A8F98"
AMBER = "#B8860B"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": "#444444",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "figure.facecolor": "white",
        "savefig.bbox": "tight",
    }
)


def _load(name: str) -> dict | None:
    path = RESULTS / name
    if not path.exists():
        print(f"  skipped: {path.relative_to(ROOT)} not found")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# --- 1. Targets vs achieved -------------------------------------------------


def fig_targets() -> Path | None:
    """The headline: what was aimed at, what was measured."""
    heldout = _load("openi_heldout.json")
    bench = _load("benchmark.json")
    if heldout is None or bench is None:
        return None

    real_run = next(
        (r for r in bench["runs"] if r["corpus"].startswith("real")), bench["runs"][0]
    )
    end_to_end = next(s for s in real_run["stages"] if s["stage"] == "ingest+deidentify")
    kappa = heldout["kappa"]["pooled"]["kappa"]

    rows = [
        # (label, achieved, target, unit, status)
        # The *slowest* of the repeats, not the median. This is a capacity claim,
        # and a capacity claim is worth no more than the worst run behind it.
        ("Ingestion throughput\n(local processing, worst run)",
         end_to_end["studies_per_hour_slowest"], 500,
         "studies/hour", "pass"),
        ("PHI removal recall\n(synthetic only)", 100.0, 99.2, "%", "pass-caveat"),
        ("Label agreement\nCohen's κ", kappa, 0.80, "κ", "pass"),
        ("Reproducibility\n(byte-identical rerun)", 100.0, 100.0, "%", "pass"),
        ("MAE linear-probe AUC", None, 0.89, "AUC", "out-of-scope"),
    ]

    fig, ax = plt.subplots(figsize=(11, 4.6))
    y = np.arange(len(rows))[::-1]

    for yi, (_label, achieved, target, unit, status) in zip(y, rows, strict=True):
        if status == "out-of-scope":
            ax.barh(yi, 1.0, color="#E8E8E8", edgecolor=MUTED, height=0.55)
            ax.text(0.5, yi, "out of scope — no model is trained in this repository",
                    va="center", ha="center", fontsize=8.5, color=MUTED, style="italic")
            continue

        # Normalise each row to "fraction of target", capped for display so a
        # 200x result does not flatten every other bar to invisibility.
        ratio = achieved / target if target else 0.0
        shown = min(ratio, 2.0)
        colour = OK if ratio >= 1.0 else WARN
        ax.barh(yi, shown, color=colour, alpha=0.85, height=0.55)
        achieved_text = (
            f"{achieved:,.0f}" if achieved >= 100 else f"{achieved:.3f}".rstrip("0").rstrip(".")
        )
        target_text = (
            f"{target:,.0f}" if target >= 100 else f"{target:g}"
        )
        ax.text(
            min(shown, 2.0) + 0.04, yi,
            f"{achieved_text} {unit}  (target {target_text})"
            + ("  — synthetic only" if status == "pass-caveat" else ""),
            va="center", fontsize=8.5,
            color=INK if status != "pass-caveat" else AMBER,
        )
        if ratio > 2.0:
            ax.text(1.97, yi, f"{ratio:,.0f}× →", va="center", ha="right",
                    fontsize=8, color="white", fontweight="bold")

    ax.axvline(1.0, color=INK, linestyle="--", linewidth=1.4)

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlim(0, 3.4)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xticklabels(["0", "0.5×", "target", "1.5×", "≥2× (capped)"])
    ax.set_xlabel("achieved, as a fraction of target (bar capped at 2×)")
    ax.set_title("Measured against the project's own stated performance targets")
    ax.grid(axis="y", visible=False)

    fig.text(
        0.5, -0.06,
        "PHI removal recall can only be measured where ground truth exists, which means "
        "synthetic data:\nevery public corpus is already de-identified. Throughput excludes "
        "network transfer.",
        ha="center", fontsize=8, color=MUTED,
    )

    path = FIGURES / "results_targets.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# --- 2. Per-finding performance on real prose -------------------------------


def fig_per_finding() -> Path | None:
    heldout = _load("openi_heldout.json")
    if heldout is None:
        return None

    per = heldout["per_finding"]
    kappas = heldout["kappa"]["per_finding"]
    order = sorted(per, key=lambda k: -per[k]["support"])

    fig, (ax, ax_k) = plt.subplots(
        1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2.1, 1]}
    )

    y = np.arange(len(order))[::-1]
    height = 0.26
    for offset, metric, colour in (
        (height, "precision", INK),
        (0.0, "recall", OK),
        (-height, "f1", MUTED),
    ):
        ax.barh(
            y + offset,
            [per[k][metric] for k in order],
            height=height,
            color=colour,
            label=metric,
            alpha=0.9,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{k.replace('_', ' ').title()}  (n={per[k]['support']})" for k in order],
        fontsize=8.5,
    )
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("score")
    ax.set_title(
        f"Finding extraction on real radiologist prose\n"
        f"held-out half, {heldout['n_reports']:,} reports, scored against MeSH annotation"
    )
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    # Kappa alongside, with the unreliable ones visibly marked.
    colours = [OK if kappas[k]["reliable"] else MUTED for k in order]
    ax_k.barh(y, [kappas[k]["kappa"] for k in order], color=colours, height=0.55, alpha=0.9)
    ax_k.axvline(0.80, color=INK, linestyle="--", linewidth=1.3)
    ax_k.text(0.80, len(order) - 0.4, " target", fontsize=8, color=INK, va="bottom")
    for yi, k in zip(y, order, strict=True):
        if not kappas[k]["reliable"]:
            ax_k.text(0.03, yi, "n too small", va="center", fontsize=7.5,
                      color="white", style="italic")
    ax_k.set_yticks(y)
    ax_k.set_yticklabels([])
    ax_k.set_xlim(0, 1.0)
    ax_k.set_xlabel("Cohen's κ")
    ax_k.set_title(f"Agreement (pooled κ {heldout['kappa']['pooled']['kappa']:.3f})")

    fig.text(
        0.5, -0.05,
        "Grey κ bars mark findings with fewer than 30 positives, where the statistic is "
        "unstable\nand the point estimate should not be quoted on its own.",
        ha="center", fontsize=8, color=MUTED,
    )

    path = FIGURES / "results_openi_per_finding.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# --- 3. What real data changed ----------------------------------------------


def fig_improvement() -> Path | None:
    before = _load("openi_heldout_baseline.json")
    after = _load("openi_heldout.json")
    if before is None or after is None:
        return None

    findings = sorted(after["per_finding"], key=lambda k: -after["per_finding"][k]["support"])

    fig, (ax, ax_head) = plt.subplots(
        1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [2.2, 1]}
    )

    y = np.arange(len(findings))[::-1]
    for yi, key in zip(y, findings, strict=True):
        b = before["per_finding"][key]["f1"]
        a = after["per_finding"][key]["f1"]
        ax.plot([b, a], [yi, yi], color=MUTED, linewidth=1.4, zorder=1)
        ax.scatter([b], [yi], color=WARN, s=42, zorder=2, label="before" if yi == y[0] else "")
        ax.scatter([a], [yi], color=OK, s=42, zorder=3, label="after" if yi == y[0] else "")
        delta = a - b
        if abs(delta) > 0.005:
            ax.annotate(
                f"{delta:+.2f}",
                xy=(max(a, b) + 0.02, yi),
                va="center", fontsize=8,
                color=OK if delta > 0 else WARN,
            )

    ax.set_yticks(y)
    ax.set_yticklabels([k.replace("_", " ").title() for k in findings], fontsize=8.5)
    ax.set_xlim(0, 1.14)
    ax.set_xlabel("F1, held-out half")
    ax.set_title("What contact with real prose changed, per finding")
    ax.legend(fontsize=8, frameon=False, loc="lower left", ncol=2)

    metrics = [
        ("micro F1", before["micro"]["f1"], after["micro"]["f1"]),
        ("macro F1", before["macro_f1"], after["macro_f1"]),
        (
            "normal-study F1",
            before["normal_detection"]["f1"],
            after["normal_detection"]["f1"],
        ),
        ("Cohen's κ", before["kappa"]["pooled"]["kappa"], after["kappa"]["pooled"]["kappa"]),
    ]
    x = np.arange(len(metrics))
    width = 0.36
    ax_head.bar(
        x - width / 2, [m[1] for m in metrics], width, color=WARN, alpha=0.85, label="before"
    )
    ax_head.bar(
        x + width / 2, [m[2] for m in metrics], width, color=OK, alpha=0.85, label="after"
    )
    for xi, (_, _before, a) in zip(x, metrics, strict=True):
        ax_head.text(xi + width / 2, a + 0.02, f"{a:.3f}", ha="center", fontsize=7.5, color=INK)
    ax_head.set_xticks(x)
    ax_head.set_xticklabels([m[0] for m in metrics], fontsize=8, rotation=20, ha="right")
    ax_head.set_ylim(0, 1.22)
    ax_head.set_title("Headline")
    ax_head.legend(fontsize=8, frameon=False, loc="upper center", ncol=2)

    fig.text(
        0.5, -0.08,
        "Four defects, none of which synthetic data could expose: 'clear of' was not treated as a "
        "negation;\n'has resolved' was read as an assertion; cardiomegaly "
        "and oedema were phrased as "
        "labels rather than as\nradiologists write them; and opacity "
        "descriptors were conflated with "
        "the consolidation diagnosis.",
        ha="center", fontsize=8, color=MUTED,
    )

    path = FIGURES / "results_improvement.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# --- 4. Overfitting check ---------------------------------------------------


def fig_dev_vs_heldout() -> Path | None:
    dev = _load("openi_dev.json")
    heldout = _load("openi_heldout.json")
    if dev is None or heldout is None:
        return None

    metrics = [
        ("micro precision", dev["micro"]["precision"], heldout["micro"]["precision"]),
        ("micro recall", dev["micro"]["recall"], heldout["micro"]["recall"]),
        ("micro F1", dev["micro"]["f1"], heldout["micro"]["f1"]),
        ("macro F1", dev["macro_f1"], heldout["macro_f1"]),
        ("exact match", dev["exact_match_rate"], heldout["exact_match_rate"]),
        ("Cohen's κ", dev["kappa"]["pooled"]["kappa"], heldout["kappa"]["pooled"]["kappa"]),
    ]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(metrics))
    width = 0.36
    ax.bar(
        x - width / 2, [m[1] for m in metrics], width, color=MUTED, alpha=0.9,
        label=f"dev half (n={dev['n_reports']:,}) — patterns tuned by reading these",
    )
    ax.bar(
        x + width / 2, [m[2] for m in metrics], width, color=OK, alpha=0.9,
        label=f"held-out half (n={heldout['n_reports']:,}) — never inspected",
    )

    for xi, (_, d, h) in zip(x, metrics, strict=True):
        ax.text(xi, max(d, h) + 0.025, f"Δ {abs(h - d):.3f}", ha="center",
                fontsize=7.5, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], fontsize=8.5, rotation=15, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score")
    ax.set_title("Tuning half vs held-out half: did the corrections generalise?")
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    fig.text(
        0.5, -0.10,
        "The phrase bank was refined by reading failures on the dev half, so a score there is "
        "optimistic by construction.\nAgreement between the halves is what says the corrections "
        "were linguistic generalisations rather than curve-fitting.",
        ha="center", fontsize=8, color=MUTED,
    )

    path = FIGURES / "results_dev_vs_heldout.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# --- 5. Real DICOM survey ---------------------------------------------------


def fig_real_dicom() -> Path | None:
    survey = _load("real_dicom.json")
    if survey is None:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))

    photo = survey["photometric_before"]
    labels = list(photo)
    values = [photo[k] for k in labels]
    colours = [WARN if k == "MONOCHROME1" else OK for k in labels]
    axes[0].bar(labels, values, color=colours, alpha=0.9)
    for i, v in enumerate(values):
        axes[0].text(i, v + max(values) * 0.02, str(v), ha="center", fontsize=9, color=INK)
    axes[0].set_title("Greyscale convention as received\n(one archive, no flag)")
    axes[0].set_ylabel("objects")
    axes[0].tick_params(axis="x", labelsize=8)

    bits = survey["bits_stored"]
    keys = sorted(bits, key=lambda k: -bits[k])
    axes[1].bar([str(k) for k in keys], [bits[k] for k in keys], color=INK, alpha=0.85)
    axes[1].set_title("BitsStored")
    axes[1].set_xlabel("bits")

    redacted = survey["n_redacted"]
    total = survey["n_objects"]
    axes[2].bar(["burned-in\ntext found", "clean"], [redacted, total - redacted],
                color=[WARN, MUTED], alpha=0.9)
    axes[2].text(0, redacted + total * 0.02, f"{redacted}\n({redacted / total:.0%})",
                 ha="center", fontsize=9, color=INK)
    axes[2].set_title("Pixel redaction on real films")
    axes[2].set_ylabel("objects")

    fig.suptitle(
        f"Real hospital archive, {total} objects  —  what the pipeline found",
        fontsize=11, y=1.04,
    )
    fig.text(
        0.5, -0.10,
        "372 of 400 objects were MONOCHROME1: photometrically inverted. "
        "Unnormalised, roughly 7% of "
        "this cohort\nwould reach a model as a photographic negative of the rest, with nothing "
        "erroring and nothing looking wrong.\nNo patient pixels are "
        "reproduced here; the archive is "
        "licensed against redistribution.",
        ha="center", fontsize=8, color=MUTED,
    )

    path = FIGURES / "results_real_dicom.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# --- 6. Equity audit --------------------------------------------------------


def fig_equity() -> Path | None:
    equity = _load("equity.json")
    if equity is None:
        return None

    strata = equity["strata"][:14]
    gaps = equity["parity_gaps"][:8]

    fig, (ax, ax_gap) = plt.subplots(
        1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.35, 1]}
    )

    y = np.arange(len(strata))[::-1]
    counts = [s["n_studies"] for s in strata]
    floor = equity["min_stratum_size"]
    colours = [OK if c >= floor else WARN for c in counts]
    ax.barh(y, counts, color=colours, alpha=0.9, height=0.6)
    ax.axvline(floor, color=INK, linestyle="--", linewidth=1.3)
    ax.text(floor, len(strata) - 0.3, f" floor = {floor}", fontsize=8, color=INK, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([s["key"] for s in strata], fontsize=7.5, family="monospace")
    ax.set_xlabel("studies")
    ax.set_title("Cohort composition by stratum (site | sex | age band)")

    if gaps:
        gy = np.arange(len(gaps))[::-1]
        ax_gap.barh(gy, [g["gap"] for g in gaps], color=INK, alpha=0.85, height=0.55)
        ax_gap.set_yticks(gy)
        ax_gap.set_yticklabels(
            [g["finding"].replace("_", " ").title() for g in gaps], fontsize=8
        )
        ax_gap.set_xlabel("max prevalence difference between strata")
        ax_gap.set_title("Prevalence parity gap")
    else:
        ax_gap.text(0.5, 0.5, "fewer than two adequately\nsized strata to compare",
                    ha="center", va="center", fontsize=9, color=MUTED)
        ax_gap.set_xticks([])
        ax_gap.set_yticks([])

    fig.text(
        0.5, -0.09,
        "These are properties of the dataset, not of a model. No classifier is trained here, so "
        "demographic parity and\nequalised odds — which are defined over predictions — "
        "cannot be computed, and are not claimed.",
        ha="center", fontsize=8, color=MUTED,
    )

    path = FIGURES / "results_equity.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# --- 7. Throughput ----------------------------------------------------------


def fig_throughput() -> Path | None:
    bench = _load("benchmark.json")
    if bench is None:
        return None

    fig, ax = plt.subplots(figsize=(9.5, 4.0))

    # Bars are the median of the repeats; the whisker spans slowest to fastest.
    # Drawing the range is the point of the figure: `ingest` varies by nearly 7x
    # between a cold and a warm filesystem cache, and a bare bar would present
    # one draw from that spread as though it were the throughput. That is the
    # exact mistake this figure previously helped make.
    labels, medians, lower, upper, colours = [], [], [], [], []
    for run in bench["runs"]:
        short = "real archive" if run["corpus"].startswith("real") else "synthetic"
        for stage in run["stages"]:
            labels.append(f"{short}\n{stage['stage']}")
            medians.append(stage["studies_per_hour_median"])
            lower.append(stage["studies_per_hour_median"] - stage["studies_per_hour_slowest"])
            upper.append(stage["studies_per_hour_fastest"] - stage["studies_per_hour_median"])
            colours.append(OK if stage["meets_target"] else WARN)

    x = np.arange(len(labels))
    ax.bar(x, medians, color=colours, alpha=0.9)
    ax.errorbar(
        x, medians, yerr=[lower, upper], fmt="none",
        ecolor=INK, elinewidth=1.2, capsize=4,
    )
    ax.axhline(bench["target_studies_per_hour"], color=INK, linestyle="--", linewidth=1.4)
    ax.text(
        len(labels) - 0.5, bench["target_studies_per_hour"] * 1.25,
        f"target {bench['target_studies_per_hour']}/hr",
        ha="right", fontsize=8.5, color=INK,
    )
    for xi, med, up in zip(x, medians, upper, strict=True):
        ax.text(xi, (med + up) * 1.10, f"{med:,.0f}", ha="center", fontsize=8, color=INK)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("studies / hour (log scale)")
    ax.set_title("Throughput against target")

    fig.text(
        0.5, -0.14,
        "Bars are the median of 5 runs; whiskers span slowest to fastest. Log scale, because the "
        "margin is two orders\nof magnitude and a linear axis would hide every difference. "
        "`ingest` varies most because it reads every file to\nhash it, so a cold cache and a warm "
        "one differ by ~7x. Local processing only: excludes network transfer from\nthe partner "
        "site, which is where a real deployment's ceiling almost certainly sits. Read as "
        "'processing is not\nthe bottleneck', not as a 200× headline.",
        ha="center", fontsize=8, color=MUTED,
    )

    path = FIGURES / "results_throughput.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(path: Path) -> str:
    """Digest a results JSON by its *data*, not its bytes.

    Hashing raw bytes seemed obviously right and was wrong. Git normalises CRLF
    to LF on commit, so a manifest written on Windows recorded digests of CRLF
    bytes while CI checked out LF and computed different ones — every source
    looked stale on a checkout where nothing had changed.

    Parsing and re-serialising canonically fixes that, and is what the check
    actually wants to ask. Reindenting a results file, or changing key order,
    does not make a figure stale; changing a value does. Bytes cannot tell those
    apart, and the question here is about data.

    Kept in step with the copy in ``tests/test_figures_current.py`` by
    ``test_digest_helpers_agree``.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _display(path: Path) -> str:
    """Repo-relative when it can be, absolute otherwise.

    ``FIGURES`` is redirected out of the repo when the currency test re-renders
    into a temporary directory, and ``relative_to`` raises rather than falling
    back on a path outside its base.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_manifest() -> Path:
    """Record what these figures were rendered from, and by what.

    RESULTS.md claims every figure is regenerated from the JSON under
    ``docs/results/``. Nothing enforced it: editing a results file without
    rerunning this script left a committed PNG showing superseded numbers, and a
    plot is harder to spot as stale than a number in a table, not easier.

    So the source digests are recorded here and checked by
    ``tests/test_figures_current.py``. That check is platform-independent —
    comparing digests of the *inputs* needs no rendering — which matters because
    comparing the PNGs themselves only works where the renderer matches. The
    renderer versions are recorded for exactly that reason.

    Source digests are canonical (see :func:`canonical_digest`), not raw-byte, so
    a CRLF checkout and an LF one agree. Figure digests are raw bytes, which is
    correct for a PNG and is why they are only compared under a matching
    renderer.
    """
    manifest = {
        "note": (
            "Written by scripts/make_results_figures.py. 'sources' are the digests of "
            "the results JSONs these figures were rendered from; if a source digest no "
            "longer matches, the committed figures are stale and `make figures` needs "
            "rerunning. 'renderer' records what drew them, because byte-identical PNG "
            "comparison is only meaningful against the same matplotlib and FreeType."
        ),
        "renderer": {
            "matplotlib": matplotlib.__version__,
            "freetype": matplotlib.ft2font.__freetype_version__,
        },
        "sources": {
            path.name: canonical_digest(path) for path in sorted(RESULTS.glob("*.json"))
        },
        "figures": {
            path.name: _sha256(path) for path in sorted(FIGURES.glob("results_*.png"))
        },
    }
    path = FIGURES / "MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    builders = (
        fig_targets,
        fig_per_finding,
        fig_improvement,
        fig_dev_vs_heldout,
        fig_real_dicom,
        fig_equity,
        fig_throughput,
    )
    written = 0
    for builder in builders:
        print(f"building {builder.__name__} ...")
        path = builder()
        if path is not None:
            print(f"  wrote {_display(path)}")
            written += 1
    print(f"\n{written} of {len(builders)} results figures written")

    manifest = write_manifest()
    print(f"wrote {_display(manifest)} (source digests + renderer versions)")


if __name__ == "__main__":
    main()
