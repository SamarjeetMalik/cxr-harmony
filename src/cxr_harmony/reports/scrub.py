"""PHI removal from free-text reports.

Prose is where de-identification is hardest. A header is a bounded, enumerable
set of attributes; a report is not, and "compared with the film of 12-04-2024 at
Sunrise Medical College" carries three identifiers that no tag-level profile will
ever touch.

The strategy here is *structure-informed* rather than purely generic. The DICOM
header for the same study already gave us the patient's name, MRN, national ID,
accession number, institution and clinician names — so those exact strings are
redacted as literals wherever they appear, including in inflected or reordered
form. That is what production de-identification pipelines actually do, and it is
far more reliable than trying to recognise a name as a name.

Generic patterns then run as a safety net for identifiers the header did not
supply: dates, telephone numbers, and residual ``Dr. Surname`` constructions.

Dates are *shifted*, not blanked, using the same per-patient offset applied to
the header. A report that says "compared with the film of six weeks ago" is
clinically meaningful, and blanking the date destroys that while shifting
preserves it. Text and header dates therefore stay consistent with one another,
which matters because an inconsistency between them is itself a re-identification
signal.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from ..deid.extract import parse_native_date
from ..deid.pseudonym import Pseudonymiser

PLACEHOLDER_NAME = "[NAME]"
PLACEHOLDER_ID = "[ID]"
PLACEHOLDER_INSTITUTION = "[INSTITUTION]"
PLACEHOLDER_PHONE = "[PHONE]"
PLACEHOLDER_ADDRESS = "[ADDRESS]"

#: Labelled header fields whose value is redacted wholesale.
_LABELLED_FIELDS = re.compile(
    r"^(?P<label>\s*(?:Patient\s+Name|Patient|Name|MRN|UHID|Hospital\s+No|ABHA(?:\s+Number)?|"
    r"Accession(?:\s+No)?|Referred\s+by|Referring\s+Physician|Consultant|Address|Tel(?:ephone)?|"
    r"Phone|Contact)\s*:)(?P<value>.*)$",
    re.I | re.M,
)

_PHONE = re.compile(r"(?:\+?\d{1,3}[-\s]?)?(?:\d[-\s]?){9,13}\d")

#: ``Dr. Priya Menon`` and ``Dr Menon``, with or without the stop.
_DOCTOR = re.compile(r"\bDr\.?\s+[A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+){0,2}")

#: The date formats the contributing sites use in prose.
_DATE = re.compile(
    r"\b("
    r"\d{2}[-/]\d{2}[-/]\d{4}"      # 12-04-2024
    r"|\d{4}[-/]\d{2}[-/]\d{2}"      # 2024-04-12
    r"|\d{1,2}\s+[A-Z][a-z]{2}\s+\d{4}"  # 12 Apr 2024
    r")\b"
)

#: Age/sex lines expose an exact age, which is quasi-identifying above 89.
_AGE_SEX = re.compile(r"^(?P<label>\s*Age\s*/?\s*Sex\s*:)(?P<value>.*)$", re.I | re.M)

#: Two or more of the same placeholder separated only by punctuation or spaces.
_REPEATED_PLACEHOLDER = re.compile(
    r"(\[(?:NAME|ID|INSTITUTION|PHONE|ADDRESS|DATE)\])(?:[\s,]+\1)+"
)


@dataclass
class ScrubResult:
    text: str
    redaction_count: int


def _redact_literals(text: str, literals: Iterable[str], placeholder: str) -> tuple[str, int]:
    """Redact known identifier strings, and the individual words of multi-word names.

    Reports rarely repeat a name exactly as the header spells it: the header says
    ``SHARMA^RAVI`` and the body says ``Mr Sharma``. Redacting the parts as well as
    the whole is what catches that.
    """
    count = 0
    tokens: set[str] = set()
    for literal in literals:
        value = str(literal or "").strip()
        if len(value) < 3:
            continue
        tokens.add(value)
        # Split DICOM person-name components and ordinary whitespace alike.
        for part in re.split(r"[\s^,]+", value):
            if len(part) >= 4 and not part.isdigit():
                tokens.add(part)

    for token in sorted(tokens, key=len, reverse=True):
        pattern = re.compile(rf"(?<!\w){re.escape(token)}(?!\w)", re.I)
        text, n = pattern.subn(placeholder, text)
        count += n
    return text, count


def scrub_report(
    text: str,
    *,
    known_names: Iterable[str] = (),
    known_ids: Iterable[str] = (),
    known_institutions: Iterable[str] = (),
    pseudonymiser: Pseudonymiser | None = None,
    pseudo_id: str | None = None,
) -> ScrubResult:
    """Remove or shift every identifier in ``text``."""
    count = 0

    # 1. Labelled header fields, value replaced wholesale.
    def _blank_field(match: re.Match[str]) -> str:
        return f"{match.group('label')} {PLACEHOLDER_ID}"

    text, n = _LABELLED_FIELDS.subn(_blank_field, text)
    count += n

    # An exact age is quasi-identifying in the 90+ tail; the sex is not.
    def _cap_age(match: re.Match[str]) -> str:
        value = match.group("value")
        years = re.search(r"(\d{1,3})", value)
        if years and int(years.group(1)) > 89:
            value = re.sub(r"\d{1,3}", "89+", value, count=1)
        return f"{match.group('label')}{value}"

    text = _AGE_SEX.sub(_cap_age, text)

    # 2. Identifiers the header already told us about.
    text, n = _redact_literals(text, known_ids, PLACEHOLDER_ID)
    count += n
    text, n = _redact_literals(text, known_institutions, PLACEHOLDER_INSTITUTION)
    count += n
    text, n = _redact_literals(text, known_names, PLACEHOLDER_NAME)
    count += n

    # 3. Generic safety net.
    text, n = _DOCTOR.subn(PLACEHOLDER_NAME, text)
    count += n
    text, n = _PHONE.subn(PLACEHOLDER_PHONE, text)
    count += n

    # 4. Dates: shifted where we can, redacted where we cannot.
    def _shift(match: re.Match[str]) -> str:
        raw = match.group(1)
        if pseudonymiser is not None and pseudo_id:
            parsed = parse_native_date(raw)
            if parsed is not None:
                return pseudonymiser.shift_date(parsed, pseudo_id).strftime("%d-%m-%Y")
        return "[DATE]"

    text, n = _DATE.subn(_shift, text)
    count += n

    # Redacting each token of a multi-word name leaves "[NAME] [NAME]"; collapse
    # runs so the output reads as prose rather than as redaction debris.
    text = _REPEATED_PLACEHOLDER.sub(r"\1", text)

    return ScrubResult(text=text, redaction_count=count)


__all__ = [
    "PLACEHOLDER_ID",
    "PLACEHOLDER_INSTITUTION",
    "PLACEHOLDER_NAME",
    "ScrubResult",
    "scrub_report",
]
