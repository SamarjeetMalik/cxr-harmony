"""Synthesis of radiograph-like pixel data, optionally with burned-in identifiers.

The images are not anatomically faithful and are not meant to be: nothing in this
pipeline reads them diagnostically. What they do need is enough spatial structure
that burned-in-text detection is a real problem rather than a trivial one. A
detector that thresholds bright pixels will happily flag the spine and the
diaphragm on an image like this, so the detector in :mod:`cxr_harmony.deid` has
to work for its result.
"""

from __future__ import annotations

import cv2
import numpy as np

#: Images are generated at this size and bit depth throughout.
IMAGE_SIZE = 512
BITS_STORED = 12
MAX_VALUE = (1 << BITS_STORED) - 1


def synthesise_radiograph(rng: np.random.Generator, *, size: int = IMAGE_SIZE) -> np.ndarray:
    """Return a ``uint16`` MONOCHROME2 array with coarse thoracic structure."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = size / 2.0, size / 2.0
    nx = (xx - cx) / (size / 2.0)
    ny = (yy - cy) / (size / 2.0)

    # Soft-tissue envelope: a broad ellipse that fades towards the collimation edge.
    body = np.exp(-((nx / 0.86) ** 2 + (ny / 0.95) ** 2) ** 2.0)

    # Two lung fields, radiolucent, hence darker than surrounding soft tissue.
    lung_offset = 0.34 + 0.02 * rng.standard_normal()
    left = np.exp(-(((nx + lung_offset) / 0.26) ** 2 + ((ny + 0.05) / 0.46) ** 2))
    right = np.exp(-(((nx - lung_offset) / 0.26) ** 2 + ((ny + 0.05) / 0.46) ** 2))
    lungs = np.clip(left + right, 0.0, 1.0)

    # Mediastinum and spine: a dense central column.
    spine = np.exp(-((nx / 0.075) ** 2)) * np.clip(1.0 - 0.35 * ny, 0.0, None)

    # Diaphragm: dense band across the lower third.
    diaphragm = 1.0 / (1.0 + np.exp(-(ny - 0.42) * 16.0))

    # Ribs: obliquely running periodic arcs, only visible over the lung fields.
    ribs = 0.5 + 0.5 * np.sin((ny * 15.0) + (np.abs(nx) * 7.0) + rng.uniform(0, np.pi))
    ribs = ribs * lungs

    field = (
        0.52 * body
        - 0.34 * lungs * body
        + 0.40 * spine
        + 0.26 * diaphragm * body
        + 0.07 * ribs
    )
    field += 0.012 * rng.standard_normal(field.shape)
    field = np.clip(field, 0.0, 1.0)

    return (field * MAX_VALUE).astype(np.uint16)


def burn_in_text(image: np.ndarray, lines: list[str], *, corner: str = "top-left") -> np.ndarray:
    """Render ``lines`` into the pixel data, as an acquisition console would.

    Burned-in annotation is the failure mode that tag-level de-identification
    misses entirely: the header can be immaculate while the patient's name sits
    in the top-left of the image itself.
    """
    out = image.copy()
    scale = out.shape[0] / 512.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42 * scale
    thickness = max(1, int(round(1 * scale)))
    line_height = int(round(19 * scale))
    margin = int(round(10 * scale))

    # OpenCV renders text into 8-bit images only, so the glyphs are drawn on a
    # mask and composited into the 16-bit array afterwards. Compositing rather
    # than overwriting keeps the anti-aliased edges, which is what a console
    # overlay actually looks like and what the detector will have to cope with.
    mask = np.zeros(out.shape, dtype=np.uint8)
    for i, line in enumerate(lines):
        if corner == "top-left":
            org = (margin, margin + line_height * (i + 1))
        else:  # bottom-left
            org = (margin, out.shape[0] - margin - line_height * (len(lines) - i - 1))
        cv2.putText(
            mask, line, org, font, font_scale, color=255, thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    alpha = mask.astype(np.float32) / 255.0
    blended = out.astype(np.float32) * (1.0 - alpha) + float(MAX_VALUE) * alpha
    return np.clip(blended, 0, MAX_VALUE).astype(np.uint16)
