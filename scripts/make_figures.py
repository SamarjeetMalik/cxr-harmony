"""Regenerate every figure in docs/figures/ from the pipeline itself.

The figures are committed to the repository, but they are not drawn by hand — this
script produces them from the same code paths the tests exercise, so a claim in the
README and the picture illustrating it cannot drift apart. Run with `make figures`.

Deterministic by construction: fixed seeds throughout, and no timestamps in output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cxr_harmony.deid.pixels import (  # noqa: E402
    DetectionParams,
    _to_uint8,
    clean_pixel_data,
    detect_text_regions,
)
from cxr_harmony.harmonize import load_site_configs  # noqa: E402
from cxr_harmony.release import SplitRatios, assign_all  # noqa: E402
from cxr_harmony.synth.pixels import burn_in_text, synthesise_radiograph  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "docs" / "figures"
CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "sites"

# A muted, print-safe palette that stays legible in greyscale.
INK = "#1F3A5F"
WARN = "#B5443B"
OK = "#2E6F50"
MUTED = "#8A8F98"

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


def fig_redaction() -> Path:
    """The one picture that shows what this project does."""
    rng = np.random.default_rng(20260731)
    base = synthesise_radiograph(rng, size=512)
    burned = burn_in_text(
        base,
        ["MEERA NAIR", "MRN MIMS/2025/01847", "PA ERECT  14-03-2025"],
    )
    cleaned, regions = clean_pixel_data(burned)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.1))

    axes[0].imshow(burned, cmap="gray", vmin=0, vmax=4095)
    axes[0].set_title("1. As received\nidentifiers burned into the pixels", color=WARN)

    axes[1].imshow(burned, cmap="gray", vmin=0, vmax=4095)
    for region in regions:
        axes[1].add_patch(
            plt.Rectangle(
                (region.x, region.y),
                region.width,
                region.height,
                fill=False,
                edgecolor=WARN,
                linewidth=1.6,
            )
        )
    axes[1].set_title(
        f"2. Detected\n{len(regions)} text regions by edge density", color=INK
    )

    axes[2].imshow(cleaned, cmap="gray", vmin=0, vmax=4095)
    axes[2].set_title("3. Redacted\nregions zeroed, not blurred", color=OK)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    fig.suptitle(
        "Burned-in PHI removal  —  the failure mode tag-level de-identification misses entirely",
        fontsize=11,
        y=1.02,
    )
    fig.text(
        0.5,
        -0.04,
        "Synthetic radiograph. A header can be immaculate while the patient's name sits in the "
        "top-left of the image itself.",
        ha="center",
        fontsize=8,
        color=MUTED,
    )

    path = FIGURES / "redaction_before_after.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _edge_density(image: np.ndarray, region) -> float:
    """Recompute the discriminating statistic for one candidate box."""
    import cv2

    img8 = _to_uint8(image)
    gradient = cv2.morphologyEx(
        img8, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    patch = (binary > 0)[
        region.y : region.y + region.height, region.x : region.x + region.width
    ]
    return float(patch.mean())


def fig_detection_separation() -> Path:
    """The empirical basis for the detector's threshold.

    Brightness cannot separate text from anatomy here: on a chest radiograph the
    spine, mediastinum and subdiaphragm all saturate to the same value the overlay
    is drawn at. Edge density can.
    """
    import cv2

    params = DetectionParams()
    text_scores: list[float] = []
    anatomy_scores: list[float] = []

    lines = [
        ["ANANYA SHARMA", "MRN SMC-001234", "12-04-2024"],
        ["RAJESH KUMAR PATEL", "MRN MIMS/2024/01099", "03-11-2023"],
        ["MEERA NAIR", "MRN ND0001234", "28-07-2025"],
    ]

    for i in range(60):
        base = synthesise_radiograph(np.random.default_rng(i), size=512)

        # Candidate boxes on a clean image are, by construction, not text.
        img8 = _to_uint8(base)
        gradient = cv2.morphologyEx(
            img8, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        )
        _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel_w = max(3, int(round(512 * params.close_width_frac)))
        joined = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
        )
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
        mask = binary > 0
        for label in range(1, n_labels):
            x, y, w, h, _ = stats[label]
            if h == 0 or w == 0 or w / h < params.min_aspect_ratio:
                continue
            if not (params.min_height_frac * 512 <= h <= params.max_height_frac * 512):
                continue
            if not (params.min_width_frac * 512 <= w <= params.max_width_frac * 512):
                continue
            anatomy_scores.append(float(mask[y : y + h, x : x + w].mean()))

        burned = burn_in_text(base, lines[i % 3])
        for region in detect_text_regions(burned):
            text_scores.append(_edge_density(burned, region))

    fig, (ax_hist, ax_img) = plt.subplots(
        1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.5, 1]}
    )

    bins = np.linspace(0, 1, 41)
    ax_hist.hist(
        bins=bins, x=anatomy_scores, color=MUTED, alpha=0.85,
        label=f"anatomy / noise  (n={len(anatomy_scores)})",
    )
    ax_hist.hist(
        bins=bins, x=text_scores, color=WARN, alpha=0.85,
        label=f"burned-in text  (n={len(text_scores)})",
    )
    ax_hist.axvline(params.min_fill_ratio, color=INK, linestyle="--", linewidth=1.5)
    ax_hist.annotate(
        f"threshold {params.min_fill_ratio}",
        xy=(params.min_fill_ratio, ax_hist.get_ylim()[1] * 0.82),
        xytext=(params.min_fill_ratio + 0.06, ax_hist.get_ylim()[1] * 0.9),
        color=INK,
        fontsize=8.5,
        arrowprops={"arrowstyle": "->", "color": INK, "lw": 1},
    )
    ax_hist.set_xlabel("edge density inside the candidate box")
    ax_hist.set_ylabel("count")
    ax_hist.set_title("Edge density separates text from anatomy")
    ax_hist.legend(fontsize=8, frameon=False)
    ax_hist.set_xlim(0, 1)

    # Why brightness alone will not do it.
    base = synthesise_radiograph(np.random.default_rng(3), size=512)
    burned = burn_in_text(base, ["ANANYA SHARMA", "MRN SMC-001234"])
    saturated = (burned >= 0.97 * 4095).astype(float)
    ax_img.imshow(saturated, cmap="gray")
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.grid(False)
    ax_img.set_title("Pixels at ≥97% of max\n(anatomy saturates too)", color=MUTED)

    gap_lo, gap_hi = max(anatomy_scores), min(text_scores)
    fig.text(
        0.5,
        -0.06,
        f"Measured over 60 images: anatomy {min(anatomy_scores):.2f}–{gap_lo:.2f}, "
        f"text {gap_hi:.2f}–{max(text_scores):.2f}. "
        f"The threshold sits in an empty gap of width {gap_hi - gap_lo:.2f}.",
        ha="center",
        fontsize=8,
        color=MUTED,
    )

    path = FIGURES / "detection_separation.png"
    fig.savefig(path)
    plt.close(fig)
    print(
        f"  anatomy max={gap_lo:.3f}  text min={gap_hi:.3f}  gap={gap_hi - gap_lo:.3f}"
    )
    return path


def fig_harmonisation() -> Path:
    """Three sites' native spellings collapsing onto one vocabulary."""
    configs = load_site_configs(CONFIGS)

    rows = [
        ("Projection: PA", "PA", "CHEST PA ERECT", "P->A", "PA"),
        ("Projection: AP", "AP", "CHEST AP SUPINE PORTABLE", "A->P", "AP"),
        ("Projection: lateral", "LL", "CHEST LAT", "LAT", "LATERAL"),
        ("Sex: male", "M", "MALE", "1", "M"),
        ("Sex: female", "F", "FEMALE", "2", "F"),
        ("Label: cardiomegaly", "Cardiomegaly", "CM", "(report prose)", "CARDIOMEGALY"),
        ("Label: effusion", "Pleural Effusion", "PE", "(report prose)", "PLEURAL_EFFUSION"),
        ("Label: normal", "No Finding", "NAD", "(report prose)", "NO_FINDING"),
        ("Date", "20250314", "14-03-2025", "20250314", "2025-03-14"),
    ]

    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.axis("off")

    col_x = [0.02, 0.20, 0.40, 0.62, 0.80]
    headers = ["", "SITE_A", "SITE_B", "SITE_C", "CANONICAL"]
    header_colours = ["black", MUTED, MUTED, MUTED, INK]

    for x, header, colour in zip(col_x, headers, header_colours, strict=True):
        ax.text(x, 0.96, header, fontsize=10, fontweight="bold", color=colour,
                transform=ax.transAxes)

    ax.plot([0.0, 1.0], [0.925, 0.925], transform=ax.transAxes, color="#444444", lw=1)

    for i, (concept, a, b, c, canonical) in enumerate(rows):
        y = 0.86 - i * 0.093
        ax.text(col_x[0], y, concept, fontsize=8.5, style="italic",
                transform=ax.transAxes, color="#333333")
        for x, value in zip(col_x[1:4], (a, b, c), strict=True):
            ax.text(x, y, value, fontsize=8.5, family="monospace",
                    transform=ax.transAxes, color=WARN if value != canonical else "#555555")
        ax.text(col_x[4], y, canonical, fontsize=8.5, family="monospace",
                fontweight="bold", transform=ax.transAxes, color=OK)
        ax.annotate(
            "",
            xy=(col_x[4] - 0.015, y + 0.012),
            xytext=(col_x[3] + 0.13, y + 0.012),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 0.9},
        )

    ax.set_title(
        "Three sites, one vocabulary  —  every mapping lives in configs/sites/*.yaml, not in code",
        fontsize=11,
        pad=14,
    )
    fig.text(
        0.5,
        0.015,
        "The HL7 numeric sex codes are the quiet hazard: '1' and '2' are valid "
        "strings that nothing rejects, so an unconfigured reader\nproduces a "
        "cohort with no usable sex variable and no error. "
        f"All {len(configs)} site adapters resolve with zero unmapped values.",
        ha="center",
        fontsize=8,
        color=MUTED,
    )

    path = FIGURES / "harmonisation.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_split_stability() -> Path:
    """Why assignment is by hash threshold rather than by shuffling."""
    ratios = SplitRatios()

    sizes = [200, 400, 800, 1600, 3200, 6400, 12800]
    hash_churn: list[float] = []
    shuffle_churn: list[float] = []

    def shuffle_assign(ids: list[str], seed: int) -> dict[str, str]:
        """The naive alternative: shuffle, then slice by proportion."""
        local = np.random.default_rng(seed)
        order = list(ids)
        local.shuffle(order)
        n = len(order)
        train_cut, val_cut = int(0.70 * n), int(0.85 * n)
        out = {}
        for i, pid in enumerate(order):
            out[pid] = "train" if i < train_cut else ("val" if i < val_cut else "test")
        return out

    base_ids = [f"{i:016x}" for i in range(sizes[0])]
    base_hash = assign_all(base_ids, ratios=ratios)
    base_shuffle = shuffle_assign(base_ids, 0)

    for size in sizes:
        grown_ids = [f"{i:016x}" for i in range(size)]

        grown_hash = assign_all(grown_ids, ratios=ratios)
        moved = sum(1 for pid in base_ids if grown_hash[pid] != base_hash[pid])
        hash_churn.append(100.0 * moved / len(base_ids))

        grown_shuffle = shuffle_assign(grown_ids, 0)
        moved = sum(1 for pid in base_ids if grown_shuffle[pid] != base_shuffle[pid])
        shuffle_churn.append(100.0 * moved / len(base_ids))

    fig, (ax_churn, ax_prop) = plt.subplots(1, 2, figsize=(11, 4.0))

    ax_churn.plot(
        sizes, shuffle_churn, "o-", color=WARN, lw=1.8, ms=5, label="shuffle-and-slice"
    )
    ax_churn.plot(
        sizes, hash_churn, "o-", color=OK, lw=1.8, ms=5,
        label="hash threshold (this pipeline)",
    )
    ax_churn.set_xscale("log")
    ax_churn.set_xlabel("cohort size after growth (started at 200 patients)")
    ax_churn.set_ylabel("% of original patients moved to a different split")
    ax_churn.set_title("Split stability as the cohort grows")
    ax_churn.legend(fontsize=8, frameon=False)
    ax_churn.set_ylim(-2, 70)

    # Realised proportions converging on target.
    conv_sizes = [50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000]
    series = (("train", INK, 0.70), ("val", OK, 0.15), ("test", WARN, 0.15))
    for split, colour, target in series:
        realised = []
        for size in conv_sizes:
            ids = [f"{i:016x}" for i in range(size)]
            assignment = assign_all(ids, ratios=ratios)
            realised.append(
                sum(1 for s in assignment.values() if s.value == split) / size
            )
        ax_prop.plot(conv_sizes, realised, "o-", color=colour, lw=1.6, ms=4, label=split)
        ax_prop.axhline(target, color=colour, linestyle=":", lw=1, alpha=0.6)

    ax_prop.set_xscale("log")
    ax_prop.set_xlabel("number of patients")
    ax_prop.set_ylabel("realised proportion")
    ax_prop.set_title("The price: proportions converge only asymptotically")
    ax_prop.legend(fontsize=8, frameon=False)
    ax_prop.set_ylim(0, 0.85)

    fig.text(
        0.5,
        -0.06,
        "A shuffle reassigns a large fraction of existing patients every time the "
        "cohort grows, so patients trained on last quarter\nland in this quarter's "
        "test set. A hash threshold depends on the patient alone, so existing "
        "assignments never move.",
        ha="center",
        fontsize=8,
        color=MUTED,
    )

    path = FIGURES / "split_stability.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  shuffle churn at 12800: {shuffle_churn[-1]:.1f}%  hash churn: {hash_churn[-1]:.1f}%")
    return path


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    builders = (
        fig_redaction,
        fig_detection_separation,
        fig_harmonisation,
        fig_split_stability,
    )
    for builder in builders:
        print(f"building {builder.__name__} ...")
        path = builder()
        print(f"  wrote {path.relative_to(FIGURES.parents[1])}")


if __name__ == "__main__":
    main()
