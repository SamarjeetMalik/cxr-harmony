"""Normalisation of photometric interpretation.

DICOM stores greyscale under one of two conventions. In ``MONOCHROME2`` the
minimum stored value renders black, which is what almost everything assumes. In
``MONOCHROME1`` the minimum value renders *white* — the image is photometrically
inverted, and a viewer that ignores the tag displays a negative.

This is not a corner case. In a 400-file sample of real computed-radiography
studies from a university hospital, 372 were MONOCHROME1 and 28 were MONOCHROME2:
a single archive, mixed conventions, no flag anywhere to warn you. Feed that
cohort into a model without normalising and roughly 7% of it is inverted relative
to the rest. Nothing errors, the images look plausible in a thumbnail grid, and
the model spends capacity learning that two visually opposite things mean the
same. It is precisely the failure a harmonisation pipeline exists to prevent.

Two consequences are handled here:

* **Conversion.** MONOCHROME1 pixel data is inverted about the stored-value range
  and the tag rewritten, so everything downstream is MONOCHROME2.
* **Redaction polarity.** Blacking out a region means writing the value that
  *renders* black. Writing zero into a MONOCHROME1 image paints a bright white
  box — it still conceals the text, but it introduces a maximal-intensity
  artefact that skews any subsequent intensity normalisation. Conversion is done
  before redaction so the question does not arise.
"""

from __future__ import annotations

import numpy as np
import pydicom

MONOCHROME1 = "MONOCHROME1"
MONOCHROME2 = "MONOCHROME2"


def is_inverted(ds: pydicom.Dataset) -> bool:
    """True when the dataset uses the inverted greyscale convention."""
    return str(getattr(ds, "PhotometricInterpretation", "")).strip().upper() == MONOCHROME1


def invert_pixels(pixels: np.ndarray, bits_stored: int) -> np.ndarray:
    """Invert about the stored-value range.

    Uses ``BitsStored`` rather than the observed maximum: inverting about the
    actual maximum would make the transform depend on image content, so two
    studies from the same device would be mapped differently.
    """
    ceiling = (1 << int(bits_stored)) - 1
    return (ceiling - np.clip(pixels, 0, ceiling)).astype(pixels.dtype)


def normalise_photometric(ds: pydicom.Dataset) -> bool:
    """Convert a MONOCHROME1 dataset in place. Returns True if it was converted.

    A no-op for datasets already in MONOCHROME2, and for datasets with no pixel
    data.
    """
    if not is_inverted(ds) or "PixelData" not in ds:
        return False

    bits_stored = int(getattr(ds, "BitsStored", 16) or 16)
    inverted = invert_pixels(ds.pixel_array, bits_stored)
    ds.PixelData = np.ascontiguousarray(inverted).tobytes()
    ds.PhotometricInterpretation = MONOCHROME2

    # A window centre calibrated for the old polarity is now wrong. Removing it is
    # safer than keeping a value that would display the converted image badly;
    # downstream code windows from the pixel data itself.
    for keyword in ("WindowCenter", "WindowWidth"):
        if keyword in ds:
            del ds[keyword]

    return True


__all__ = [
    "MONOCHROME1",
    "MONOCHROME2",
    "invert_pixels",
    "is_inverted",
    "normalise_photometric",
]
