"""Detection and redaction of burned-in annotation.

This is the half of de-identification that tag-based tools miss completely. A
header can be immaculate while the patient's name sits in the top-left of the
image, rendered there by the acquisition console before the object was ever
stored. Portable and mobile units are the usual offenders, which is precisely
the equipment used for the sickest patients.

Detection cannot be done by brightness alone. On a chest radiograph the spine,
the mediastinum and the subdiaphragmatic region routinely saturate to the same
value the text is drawn at, so a threshold picks up the anatomy and misses
nothing useful. What separates text from anatomy is *spatial frequency*: glyph
edges are sharp and periodic at a small scale, anatomical density gradients are
smooth. The pipeline below keys on that instead —

1. a morphological gradient, which responds to sharp edges and largely ignores
   smooth density changes;
2. Otsu thresholding of the gradient response;
3. a horizontal closing, which merges neighbouring glyphs into word and line
   blobs while leaving isolated anatomical edges as small specks;
4. geometric filtering on the resulting components.

Redaction is destructive: the region is filled with zeros in the stored pixel
data. Blurring or pixelating is not acceptable for this purpose — both have been
shown to be invertible often enough that they cannot be relied on.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TextRegion:
    """An axis-aligned box believed to contain burned-in characters."""

    x: int
    y: int
    width: int
    height: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class DetectionParams:
    """Geometry filters, expressed relative to image size so they scale."""

    #: Merged text lines are wider than they are tall.
    min_aspect_ratio: float = 1.6
    #: A glyph line occupies a small fraction of the image height.
    min_height_frac: float = 0.012
    max_height_frac: float = 0.10
    #: And a bounded fraction of its width.
    min_width_frac: float = 0.03
    max_width_frac: float = 0.85
    #: Edge density inside the box, measured on the *pre-closing* mask.
    #:
    #: A run of glyphs is edge-dense: every stroke contributes two gradient
    #: responses, and at the sizes text is actually rendered those responses
    #: nearly fill the line's bounding box. Anatomical and noise edges that happen
    #: to survive the closing are, by contrast, sparse and stringy. Measured on
    #: the synthetic corpus at 128–1024 px, text lines score 0.55–1.00 and
    #: non-text components 0.16–0.35, so the threshold sits in a wide empty gap.
    #:
    #: This must be measured before the closing, not after: closing fills a text
    #: line by construction, which drives every component to ~0.99 and destroys
    #: the signal.
    min_fill_ratio: float = 0.45
    max_fill_ratio: float = 1.0
    #: Horizontal kernel width, as a fraction of image width, used to join glyphs.
    close_width_frac: float = 0.022
    #: Boxes are grown by this fraction of their size before redaction, so that
    #: anti-aliased glyph edges are not left behind as a legible ghost.
    pad_frac: float = 0.35


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Window the image to 8 bits for the morphology, without altering the original."""
    img = image.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8)
    return (((img - lo) / (hi - lo)) * 255.0).astype(np.uint8)


def detect_text_regions(
    image: np.ndarray,
    params: DetectionParams | None = None,
) -> list[TextRegion]:
    """Return boxes likely to contain burned-in characters."""
    params = params or DetectionParams()
    height, width = image.shape[:2]
    img8 = _to_uint8(image)

    gradient = cv2.morphologyEx(
        img8, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel_w = max(3, int(round(width * params.close_width_frac)))
    joined = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    )

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    stroke_mask = binary > 0

    regions: list[TextRegion] = []
    for label in range(1, n_labels):
        x, y, w, h, _area = stats[label]
        if h == 0 or w == 0:
            continue
        if not (params.min_height_frac * height <= h <= params.max_height_frac * height):
            continue
        if not (params.min_width_frac * width <= w <= params.max_width_frac * width):
            continue
        if w / h < params.min_aspect_ratio:
            continue
        # Density of the original edge response inside the merged box.
        fill = float(stroke_mask[y : y + h, x : x + w].mean())
        if not (params.min_fill_ratio <= fill <= params.max_fill_ratio):
            continue
        regions.append(TextRegion(int(x), int(y), int(w), int(h)))

    return sorted(regions, key=lambda r: (r.y, r.x))


def redact_regions(
    image: np.ndarray,
    regions: list[TextRegion],
    params: DetectionParams | None = None,
) -> np.ndarray:
    """Fill ``regions`` with zeros, padded outwards to catch anti-aliased edges."""
    params = params or DetectionParams()
    if not regions:
        return image

    out = image.copy()
    height, width = out.shape[:2]
    for region in regions:
        pad_x = int(round(region.width * params.pad_frac * 0.15)) + 2
        pad_y = int(round(region.height * params.pad_frac)) + 2
        x0 = max(0, region.x - pad_x)
        y0 = max(0, region.y - pad_y)
        x1 = min(width, region.x + region.width + pad_x)
        y1 = min(height, region.y + region.height + pad_y)
        out[y0:y1, x0:x1] = 0
    return out


def clean_pixel_data(
    image: np.ndarray,
    params: DetectionParams | None = None,
) -> tuple[np.ndarray, list[TextRegion]]:
    """Detect and redact in one step. Returns ``(cleaned_image, regions)``."""
    regions = detect_text_regions(image, params)
    return redact_regions(image, regions, params), regions


__all__ = [
    "DetectionParams",
    "TextRegion",
    "clean_pixel_data",
    "detect_text_regions",
    "redact_regions",
]
