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
    """Return a ``uint16`` MONOCHROME2 array with coarse thoracic structure.

    MONOCHROME2 convention: larger value renders brighter, and denser tissue
    attenuates more, so bone and mediastinum are bright while the air-filled lung
    fields are dark. The layout is a crude frontal chest: two lung fields either
    side of a mediastinal column that widens into a left-biased cardiac shadow,
    bounded below by diaphragm domes and crossed by rib arcs.

    It is not anatomically faithful and nothing here reads it diagnostically. It
    needs only enough structure that burned-in-text detection is a real problem:
    smooth density gradients that saturate to the same value the overlay is drawn
    at, so brightness alone cannot separate text from anatomy.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    nx = (xx - size / 2.0) / (size / 2.0)
    ny = (yy - size / 2.0) / (size / 2.0)

    jitter = 0.02 * rng.standard_normal()

    # Thoracic soft-tissue envelope: broad, slightly taller than wide, with the
    # shoulders flaring at the top and the flanks falling away at the sides.
    body = np.exp(-(((nx / 0.80) ** 2 + ((ny - 0.05) / 0.92) ** 2) ** 2.2))
    shoulders = np.exp(-(((np.abs(nx) - 0.62) / 0.30) ** 2 + ((ny + 0.72) / 0.26) ** 2))

    # Lung fields: tall ellipses, wider laterally than medially, sitting in the
    # upper two thirds. Radiolucent, so they subtract.
    offset = 0.36 + jitter
    def _lung(sign: float) -> np.ndarray:
        return np.exp(
            -(
                ((nx - sign * offset) / 0.27) ** 2
                + ((ny + 0.16) / 0.50) ** 2
            )
            ** 1.35
        )

    lungs = np.clip(_lung(1.0) + _lung(-1.0), 0.0, 1.0)

    # Mediastinum: a narrow, dense column — the one structure allowed to reach
    # the top of the scale, because the detector must cope with anatomy that
    # saturates to the same value the burned-in overlay is drawn at.
    column = np.exp(-((nx / 0.055) ** 2)) * np.clip(1.0 - 0.35 * ny, 0.0, None)
    # Cardiac silhouette: broader, softer, left-biased, and well short of saturation.
    heart = np.exp(-((((nx + 0.14) / 0.27) ** 2 + ((ny - 0.22) / 0.21) ** 2) ** 1.5))

    # Diaphragm: two domes rather than a straight band, the right sitting higher.
    dome = 0.44 - 0.13 * np.exp(-((np.abs(nx) - 0.34) / 0.26) ** 2)
    diaphragm = 1.0 / (1.0 + np.exp(-(ny - dome) * 15.0))

    # Ribs: posterior arcs running obliquely, visible only over aerated lung.
    ribs = 0.5 + 0.5 * np.sin((ny * 13.0) + (np.abs(nx) * 5.5) + rng.uniform(0, np.pi))

    field = (
        0.20
        + 0.30 * body
        + 0.12 * shoulders * body
        - 0.34 * lungs * body
        + 0.46 * column
        + 0.17 * heart * body
        + 0.20 * diaphragm * body
        + 0.05 * ribs * lungs * body
    )
    field += 0.010 * rng.standard_normal(field.shape)

    # Collimation: the exposed field stops short of the detector edge.
    collimation = np.clip(1.0 - np.exp((np.abs(nx) - 0.97) * 26.0), 0.0, 1.0) * np.clip(
        1.0 - np.exp((np.abs(ny) - 0.99) * 26.0), 0.0, 1.0
    )
    field = field * collimation

    return (np.clip(field, 0.0, 1.0) * MAX_VALUE).astype(np.uint16)


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
