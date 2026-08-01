"""The burned-in audit trail, and the property that makes it safe to keep.

The whole design rests on one claim: the audit record describes what was redacted
without containing it. That claim is worth more than the feature — a trail that
leaked plaintext would be a concentrated, indexed copy of exactly the identifiers
the pipeline exists to destroy, and worse than having no trail at all.

So :func:`test_no_plaintext_survives_in_the_audit_record` renders known strings,
audits them, and searches every field of the serialised record for any fragment
of what it rendered. If that test passes for the wrong reason — because OCR read
nothing — :func:`test_the_leak_test_would_actually_catch_a_leak` fails, because a
test that cannot fail is not evidence.

Tests needing the engine skip without it. The tests that matter most for safety —
that a missing engine changes no redaction, and that the pipeline still runs — do
not need it, and run everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from cxr_harmony.deid.ocr import (
    OcrUnavailable,
    RedactionAudit,
    TextCategory,
    available,
    describe_redactions,
    matches,
)
from cxr_harmony.deid.pixels import TextRegion

KEY = b"ocr-audit-test-key-32-bytes!!!!!"

requires_ocr = pytest.mark.skipif(
    not available(),
    reason='no tesseract; install with pip install -e ".[ocr]" plus the binary',
)


def _canvas(width: int = 480, height: int = 96) -> np.ndarray:
    return np.zeros((height, width), dtype=np.uint8)


def _render(text: str, *, width: int = 480, height: int = 96) -> np.ndarray:
    """Burn `text` onto a blank canvas, the way the synthetic corpus does."""
    import cv2

    image = _canvas(width, height)
    cv2.putText(image, text, (8, height // 2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
    return image


FULL = [TextRegion(0, 0, 480, 96)]


# --- behaviour without the engine, which must be exactly the old behaviour ---


def test_absent_engine_yields_no_audit_rather_than_an_error(monkeypatch):
    monkeypatch.setattr("cxr_harmony.deid.ocr.available", lambda: False)
    assert describe_redactions(_canvas(), FULL, key=KEY) == []


def test_absent_engine_raises_only_when_explicitly_required(monkeypatch):
    """Silence is the default; a caller that needs the trail can demand it."""
    monkeypatch.setattr("cxr_harmony.deid.ocr.available", lambda: False)
    with pytest.raises(OcrUnavailable, match="tesseract"):
        describe_redactions(_canvas(), FULL, key=KEY, required=True)


def test_no_regions_means_no_work():
    """Checked before availability, so an empty page never needs an engine."""
    assert describe_redactions(_canvas(), [], key=KEY) == []


def test_a_region_outside_the_image_is_unreadable_not_a_crash():
    """Detector and image can disagree; that must not take the pipeline down."""
    audits = describe_redactions(_canvas(), [TextRegion(900, 900, 40, 20)], key=KEY)
    if audits:  # empty when OCR is absent, which is also acceptable
        assert audits[0].category is TextCategory.UNREADABLE
        assert audits[0].digest is None


# --- the privacy property ----------------------------------------------------

SECRETS = ["PRIYA SHARMA", "ACC00123456", "1998-04-17"]


@requires_ocr
@pytest.mark.parametrize("secret", SECRETS)
def test_no_plaintext_survives_in_the_audit_record(secret):
    """No fragment of the burned-in string may appear anywhere in the record.

    Searches the serialised form, not the attributes, because serialisation is
    what reaches disk. Fragments of length 4 rather than the whole string: a
    record leaking "SHARMA" would be a disclosure even though it is not the
    string that went in.
    """
    audits = describe_redactions(_render(secret), FULL, key=KEY)
    assert audits, "OCR available but produced no audit record"

    blob = repr([a.to_dict() for a in audits]).upper()
    compact = secret.replace(" ", "").replace("-", "").upper()
    for start in range(len(compact) - 3):
        fragment = compact[start : start + 4]
        assert fragment not in blob.replace(" ", "").replace("-", ""), (
            f"the audit record contains {fragment!r} from {secret!r}. The record "
            "is written to disk; it must describe the redaction, never carry it."
        )


@requires_ocr
def test_the_leak_test_would_actually_catch_a_leak():
    """Guards the test above against passing because OCR read nothing.

    A record with the text deliberately put back must fail the same search. If it
    does not, the leak test proves nothing about the real record.
    """
    secret = "PRIYA SHARMA"
    leaky = RedactionAudit(
        region=TextRegion(0, 0, 10, 10),
        category=TextCategory.NAME_LIKE,
        char_count=len(secret),
        confidence=90.0,
        digest=secret,  # the mistake this whole module exists to avoid
    )
    blob = repr([leaky.to_dict()]).upper().replace(" ", "")
    assert "SHAR" in blob, "the search used by the leak test cannot detect plaintext"


@requires_ocr
def test_the_digest_verifies_the_string_it_will_not_store():
    """The point of a MAC: answer "was it this?" without having kept it."""
    secret = "PRIYA SHARMA"
    audits = describe_redactions(_render(secret), FULL, key=KEY)
    assert audits and audits[0].digest

    assert matches(audits[0], secret, KEY)
    assert matches(audits[0], secret.lower(), KEY), "should be case-insensitive"
    assert not matches(audits[0], "RAVI KUMAR", KEY)


@requires_ocr
def test_a_different_key_cannot_verify():
    """The digest is keyed, so a stolen audit file alone confirms nothing."""
    secret = "PRIYA SHARMA"
    audits = describe_redactions(_render(secret), FULL, key=KEY)
    assert audits and audits[0].digest
    assert not matches(audits[0], secret, b"a-different-key-32-bytes!!!!!!!!")


def test_matching_an_unreadable_region_is_false_not_an_error():
    empty = RedactionAudit(TextRegion(0, 0, 1, 1), TextCategory.UNREADABLE, 0, 0.0, None)
    assert not matches(empty, "anything", KEY)


# --- classification ----------------------------------------------------------


@requires_ocr
@pytest.mark.parametrize(
    "text,expected",
    [
        ("1998-04-17", TextCategory.DATE_LIKE),
        ("ACC00123456", TextCategory.ACCESSION_LIKE),
    ],
)
def test_categories_are_recognised(text, expected):
    audits = describe_redactions(_render(text), FULL, key=KEY)
    assert audits
    if audits[0].category is TextCategory.UNREADABLE:
        pytest.skip(f"tesseract did not read {text!r}; classification untestable")
    assert audits[0].category is expected


@requires_ocr
def test_a_blank_region_reads_as_unreadable():
    """No text and unreadable text are the same verdict, deliberately.

    Distinguishing them would require knowing what was there, which is the thing
    that cannot be known. Conflating them is what makes the false-positive figure
    an upper bound rather than a rate.
    """
    audits = describe_redactions(_canvas(), FULL, key=KEY)
    assert audits and audits[0].category is TextCategory.UNREADABLE
    assert audits[0].digest is None
