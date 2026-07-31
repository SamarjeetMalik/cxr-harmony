"""Independent verification that de-identification actually happened.

This module deliberately does not import the profile's action table. It re-reads
what was written and checks it against the requirement, not against the
implementation. A verifier that shares the engine's notion of "which tags matter"
will agree with the engine about a tag they both forgot, which is precisely the
failure it exists to catch.

Two checks are run:

* **Structural** — no attribute from the identifying set survives with a value,
  no private or overlay group survives, and the required de-identification
  markers are present.
* **Substring** — the actual identifier strings that went in are searched for
  across every output header. This is the stronger check, and it is only possible
  because the corpus is synthetic and its ground truth is known. For a real
  delivery the structural check is what you get, which is an argument for
  validating a de-identifier against synthetic data before pointing it at
  patients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom

#: Header values are joined with a separator that cannot occur inside a DICOM
#: string value, so a search cannot match text spanning two adjacent fields.
_FIELD_SEP = chr(10)

#: Attributes that must not survive with a value. Written out independently of
#: the profile table, from the requirement rather than from the implementation.
MUST_BE_ABSENT_OR_EMPTY = (
    "PatientBirthDate",
    "PatientBirthTime",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "OtherPatientIDs",
    "OtherPatientNames",
    "InstitutionName",
    "InstitutionAddress",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "NameOfPhysiciansReadingStudy",
    "OperatorsName",
    "StationName",
    "DeviceSerialNumber",
    "AccessionNumber",
    "StudyID",
    "StudyDescription",
    "SeriesDescription",
    "ImageComments",
    "RequestingPhysician",
    "PatientComments",
    "AdditionalPatientHistory",
)


@dataclass
class Violation:
    """One way in which an output object failed verification."""

    relative_path: str
    kind: str
    detail: str

    def to_dict(self) -> dict:
        return {"relative_path": self.relative_path, "kind": self.kind, "detail": self.detail}


@dataclass
class VerificationReport:
    n_checked: int = 0
    violations: list[Violation] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations

    def by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for v in self.violations:
            counts[v.kind] = counts.get(v.kind, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict:
        return {
            "n_checked": self.n_checked,
            "passed": self.passed,
            "n_violations": len(self.violations),
            "by_kind": self.by_kind(),
            "violations": [v.to_dict() for v in self.violations[:200]],
        }


def _header_strings(ds: pydicom.Dataset) -> list[str]:
    """Every string value in the header, for substring searching."""
    values: list[str] = []
    for element in ds:
        if element.tag == 0x7FE00010:  # PixelData
            continue
        if element.VR == "SQ":
            for item in element.value or []:
                values.extend(_header_strings(item))
            continue
        try:
            value = element.value
        except Exception:  # noqa: BLE001 - unreadable element cannot hide a string
            continue
        if isinstance(value, (str, bytes)):
            values.append(value.decode("utf-8", "ignore") if isinstance(value, bytes) else value)
        elif isinstance(value, (list, tuple)):
            values.extend(str(v) for v in value)
    return values


def verify_object(path: Path, relative_path: str, phi_values: list[str]) -> list[Violation]:
    """Check one written object. Returns the violations found, possibly none."""
    ds = pydicom.dcmread(path)
    violations: list[Violation] = []

    for keyword in MUST_BE_ABSENT_OR_EMPTY:
        value = getattr(ds, keyword, None)
        if value not in (None, "", []):
            violations.append(
                Violation(relative_path, "attribute_survived", f"{keyword}={value!r}")
            )

    for element in ds:
        group = element.tag.group
        if group % 2 == 1:
            violations.append(
                Violation(relative_path, "private_tag_survived", str(element.tag))
            )
        elif 0x6000 <= group <= 0x60FF or 0x5000 <= group <= 0x50FF:
            violations.append(
                Violation(relative_path, "overlay_survived", str(element.tag))
            )

    if str(getattr(ds, "PatientIdentityRemoved", "")) != "YES":
        violations.append(Violation(relative_path, "missing_marker", "PatientIdentityRemoved"))
    if not str(getattr(ds, "DeidentificationMethod", "")):
        violations.append(Violation(relative_path, "missing_marker", "DeidentificationMethod"))
    if str(getattr(ds, "BurnedInAnnotation", "")) != "NO":
        violations.append(Violation(relative_path, "burned_in_flag", "BurnedInAnnotation"))

    if phi_values:
        haystack = _FIELD_SEP.join(_header_strings(ds)).upper()
        for phi in phi_values:
            token = str(phi).strip().upper()
            if len(token) >= 5 and token in haystack:
                violations.append(Violation(relative_path, "phi_substring", token[:40]))

    return violations


def verify_store(
    deid_store: Path,
    *,
    phi_values: list[str] | None = None,
) -> VerificationReport:
    """Re-read every object under ``deid_store`` and check it."""
    report = VerificationReport()
    store = Path(deid_store)
    if not store.exists():
        return report

    for path in sorted(store.rglob("*.dcm")):
        relative = path.relative_to(store).as_posix()
        report.n_checked += 1
        report.violations.extend(verify_object(path, relative, phi_values or []))
    return report


def pixels_are_clean(original: np.ndarray, cleaned: np.ndarray) -> bool:
    """True when every pixel the burn-in altered has been zeroed.

    Used by the test suite, where both arrays are available. It is a stricter
    statement than "a region was detected": it asserts that nothing the overlay
    touched is still readable.
    """
    changed = original != cleaned
    if not changed.any():
        return False
    return bool((cleaned[changed] == 0).all())


__all__ = [
    "MUST_BE_ABSENT_OR_EMPTY",
    "VerificationReport",
    "Violation",
    "pixels_are_clean",
    "verify_object",
    "verify_store",
]
