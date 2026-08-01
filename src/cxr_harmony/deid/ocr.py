"""Describe what burned-in redaction removed, without keeping it.

The pipeline detects burned-in text geometrically and zeroes it. That is enough
to remove PHI and not enough to audit: a site asking "what did you take off my
images?" gets no answer, and neither does anyone investigating a suspected
disclosure.

**The obvious fix is the wrong one.** Running OCR and storing the text produces a
file that contains, in plain text, every patient name and accession number the
tool exists to destroy — concentrated, indexed and far easier to exfiltrate than
the pixels were. An audit trail built that way is a worse disclosure risk than
the problem it documents.

So this records a *classification* instead:

- a :class:`TextCategory` for the shape of what was there,
- how many characters, and how confident the reader was,
- a keyed HMAC of the normalised text.

The HMAC is what makes the trail useful without making it dangerous. It cannot be
reversed — but if someone later asks "was patient X's name burned into this
study?", :func:`matches` answers it, because the question comes with the candidate
string. Verification without storage.

Three constraints hold this in place, and each has a test:

1. **It never runs before redaction, and never changes it.** Detection is
   geometric and stays that way. This module only describes boxes that were
   already chosen, so a broken or missing OCR engine cannot cause PHI to survive.
2. **Plaintext is never returned or written.** The recognised string exists as a
   local for the length of one function call.
3. **It is optional.** Without ``pytesseract`` and a ``tesseract`` binary the
   pipeline behaves exactly as before. ``make demo`` gains no dependency.

Install with ``pip install -e ".[ocr]"`` plus the system binary: ``apt-get install
tesseract-ocr`` on Debian, ``winget install tesseract-ocr.tesseract`` on Windows.
"""

from __future__ import annotations

import hmac
import re
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

# The project's own shim, not `enum.StrEnum`, which needs 3.11 while this
# supports 3.10. Caught by CI on the matrix's oldest interpreter.
from ..schema.vocab import StrEnum
from .pixels import TextRegion

__all__ = [
    "OcrUnavailable",
    "RedactionAudit",
    "TextCategory",
    "available",
    "describe_redactions",
    "matches",
]

#: Domain separator, so an audit MAC can never be confused with a pseudonym or a
#: remapped UID derived from the same key.
_MAC_DOMAIN = "burned-in-audit"

#: Below this mean word confidence, a reading is discarded as noise.
#:
#: Not a tuning knob — a correctness floor. Tesseract given a **blank** region
#: returned two characters at confidence 18.0, which arrived here as OTHER_TEXT
#: with a digest. Left alone that is corrosive in two directions: the audit trail
#: gains records of text that was never there, and
#: ``scripts/measure_burned_in_fp.py`` counts a hallucinated read as evidence
#: that a spurious box held real text — biasing the false-positive bound
#: *downwards*, which is the direction that flatters the detector.
#:
#: 40 is a judgement, not a measurement: genuine burned-in annotation in this
#: corpus reads well above it and blank-region noise well below, but the gap has
#: not been characterised the way the edge-density threshold was. The
#: false-positive figure is sensitive to this value, and says so.
MIN_CONFIDENCE = 40.0


class OcrUnavailable(RuntimeError):
    """Raised only when OCR is explicitly requested and cannot be provided."""


class TextCategory(StrEnum):
    """What kind of thing was burned in, at the coarsest useful resolution.

    Coarse on purpose. "A date was removed from the top-left corner" is what an
    audit needs; the date itself is what an audit must not accumulate.
    """

    DATE_LIKE = "DATE_LIKE"
    NAME_LIKE = "NAME_LIKE"
    ACCESSION_LIKE = "ACCESSION_LIKE"
    LATERALITY_MARKER = "LATERALITY_MARKER"
    OTHER_TEXT = "OTHER_TEXT"
    #: Read as empty. Either the box held no text, or the text defeated the
    #: reader. Those are not distinguishable from here, and conflating them is
    #: what would turn this into a false false-positive rate.
    UNREADABLE = "UNREADABLE"


#: Ordered: the first match wins, so the specific patterns precede the general.
_CATEGORY_PATTERNS: tuple[tuple[TextCategory, re.Pattern[str]], ...] = (
    # Separated or 8-digit dates, and month names. Burned-in dates are the most
    # common non-name identifier on a film.
    (
        TextCategory.DATE_LIKE,
        re.compile(
            r"(\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4})|(\b\d{8}\b)"
            r"|\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\w*\b",
            re.I,
        ),
    ),
    # Single letters or short words used as side markers. Checked before the
    # name and accession rules, which would otherwise swallow them.
    (
        TextCategory.LATERALITY_MARKER,
        re.compile(
            r"^\W*((L|R|LT|RT|LEFT|RIGHT|AP|PA|SUPINE|ERECT|PORT|PORTABLE)\W*)+$",
            re.I,
        ),
    ),
    # A long digit run, or a letter-prefixed number: accession, MRN, ABHA.
    (TextCategory.ACCESSION_LIKE, re.compile(r"\b[A-Z]{0,4}\d{5,}\b", re.I)),
    # Two or more capitalised words, or a comma-separated surname-first pair.
    (
        TextCategory.NAME_LIKE,
        re.compile(r"\b[A-Z][a-z]+\s*,\s*[A-Z][a-z]+\b|\b[A-Z][A-Z]+\s+[A-Z][A-Z]+\b"),
    ),
)


@dataclass(frozen=True)
class RedactionAudit:
    """A record of one redacted region. Deliberately not the text itself."""

    region: TextRegion
    category: TextCategory
    #: Characters recognised. Zero whenever the category is ``UNREADABLE``.
    char_count: int
    #: Mean per-word confidence the engine reported, 0-100.
    confidence: float
    #: Keyed HMAC of the normalised text, or ``None`` when nothing was read.
    #: Use :func:`matches` rather than comparing this by hand.
    digest: str | None

    def to_dict(self) -> dict:
        return {
            "region": self.region.to_dict(),
            "category": self.category.value,
            "char_count": self.char_count,
            "confidence": round(self.confidence, 1),
            "digest": self.digest,
        }


def available() -> bool:
    """Whether OCR can run here. Cheap enough to call per invocation."""
    try:
        import pytesseract
    except ImportError:
        return False
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        # pytesseract raises its own TesseractNotFoundError, and an OSError when
        # the binary is present but unrunnable. Neither is worth distinguishing:
        # both mean the same thing to a caller.
        return False
    return True


def _normalise(text: str) -> str:
    """Collapse whitespace and case so the MAC is stable across readings.

    OCR is not deterministic across versions or preprocessing, and a digest that
    changes with spacing would verify nothing.
    """
    return re.sub(r"\s+", " ", text).strip().upper()


def _categorise(text: str) -> TextCategory:
    for category, pattern in _CATEGORY_PATTERNS:
        if pattern.search(text):
            return category
    return TextCategory.OTHER_TEXT


def _digest(key: bytes, text: str) -> str:
    return hmac.new(key, f"{_MAC_DOMAIN}\x00{text}".encode(), sha256).hexdigest()


def matches(audit: RedactionAudit, candidate: str, key: bytes) -> bool:
    """Whether ``candidate`` is what this region held.

    The point of storing a MAC rather than the string. A site asking "was this
    patient's name on these films?" supplies the name; this answers without the
    name ever having been kept.
    """
    if audit.digest is None:
        return False
    return hmac.compare_digest(audit.digest, _digest(key, _normalise(candidate)))


def describe_redactions(
    image: np.ndarray,
    regions: list[TextRegion],
    *,
    key: bytes,
    required: bool = False,
) -> list[RedactionAudit]:
    """Classify each already-detected region. Returns ``[]`` when OCR is absent.

    ``image`` must be the array **before** redaction — after it, every region is
    zeros. This is the one ordering constraint, and getting it wrong produces an
    audit trail of blanks rather than a disclosure, which is the safe direction.

    Set ``required=True`` to turn a missing engine into :class:`OcrUnavailable`
    instead of an empty list. The default is silence, because the pipeline treats
    this as an optional enrichment and must not fail a de-identification run over
    a missing audit trail.
    """
    if not regions:
        return []
    if not available():
        if required:
            raise OcrUnavailable(
                "pytesseract and a tesseract binary are required for the burned-in "
                'audit trail. Install with: pip install -e ".[ocr]" plus '
                "`apt-get install tesseract-ocr` or "
                "`winget install tesseract-ocr.tesseract`."
            )
        return []

    import pytesseract

    audits: list[RedactionAudit] = []
    for region in regions:
        crop = image[
            region.y : region.y + region.height, region.x : region.x + region.width
        ]
        if crop.size == 0:
            audits.append(
                RedactionAudit(region, TextCategory.UNREADABLE, 0, 0.0, None)
            )
            continue

        text, confidence = _read(pytesseract, crop)
        normalised = _normalise(text)
        if confidence < MIN_CONFIDENCE:
            # Discard rather than record. A low-confidence read is not weak
            # evidence of text, it is evidence of noise: see MIN_CONFIDENCE.
            normalised = ""
        if not normalised:
            audits.append(
                RedactionAudit(region, TextCategory.UNREADABLE, 0, confidence, None)
            )
            continue

        audits.append(
            RedactionAudit(
                region=region,
                category=_categorise(normalised),
                char_count=len(normalised),
                confidence=confidence,
                digest=_digest(key, normalised),
            )
        )
    # `text` and `normalised` fall out of scope here and are never returned,
    # logged or written. That is the whole privacy argument, and
    # test_no_plaintext_survives_in_the_audit_record is what enforces it.
    return audits


def _read(pytesseract, crop: np.ndarray) -> tuple[str, float]:
    """Recognise a single line, returning text and mean word confidence.

    Scaled up first: burned-in annotation is often rendered small, and tesseract
    is markedly better above roughly 30px of glyph height. ``--psm 7`` tells it
    the crop is one line, which it is by construction — the detector merges
    glyphs horizontally before proposing a box.
    """
    import cv2

    scaled = _to_uint8(crop)
    if scaled.shape[0] < 32:
        factor = max(2, int(round(32 / max(scaled.shape[0], 1))))
        scaled = cv2.resize(
            scaled, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC
        )

    try:
        data = pytesseract.image_to_data(
            scaled, config="--psm 7", output_type=pytesseract.Output.DICT
        )
    except Exception:
        return "", 0.0

    words, confidences = [], []
    for word, conf in zip(data.get("text", []), data.get("conf", []), strict=False):
        if not str(word).strip():
            continue
        words.append(str(word))
        try:
            value = float(conf)
        except (TypeError, ValueError):
            continue
        if value >= 0:  # tesseract reports -1 for boxes it did not score
            confidences.append(value)

    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return " ".join(words), mean_conf


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Window a 16-bit array into 8 bits, which is all tesseract accepts."""
    if image.dtype == np.uint8:
        return image
    array = image.astype(np.float32)
    low, high = float(array.min()), float(array.max())
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    return (((array - low) / (high - low)) * 255).astype(np.uint8)
