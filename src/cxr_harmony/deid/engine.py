"""Application of the confidentiality profile to an indexed delivery."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom
from pydicom.uid import ExplicitVRLittleEndian

from ..workspace import Workspace, read_jsonl, write_jsonl
from .extract import SiteFacts, extract_facts, read_linkage_identifiers
from .pixels import DetectionParams, clean_pixel_data
from .profile import (
    DEIDENTIFICATION_METHOD,
    DEIDENTIFICATION_METHOD_CODES,
    DUMMY_STRING,
    PATIENT_IDENTITY_REMOVED,
    Action,
    action_for,
    is_overlay_or_curve_group,
)
from .pseudonym import Pseudonymiser, load_or_create_key


@dataclass
class DeidRecord:
    """One de-identified object, as written."""

    relative_path: str
    site_id: str
    pseudo_patient_id: str
    linked_across_sites: bool
    study_uid: str
    series_uid: str
    sop_uid: str
    sha256: str
    rows: int
    columns: int
    bits_stored: int
    photometric_interpretation: str
    pixel_redacted: bool
    redacted_regions: list[dict]
    source_path: str

    def to_dict(self) -> dict:
        return {
            "relative_path": self.relative_path,
            "site_id": self.site_id,
            "pseudo_patient_id": self.pseudo_patient_id,
            "linked_across_sites": self.linked_across_sites,
            "study_uid": self.study_uid,
            "series_uid": self.series_uid,
            "sop_uid": self.sop_uid,
            "sha256": self.sha256,
            "rows": self.rows,
            "columns": self.columns,
            "bits_stored": self.bits_stored,
            "photometric_interpretation": self.photometric_interpretation,
            "pixel_redacted": self.pixel_redacted,
            "redacted_regions": self.redacted_regions,
            "source_path": self.source_path,
        }


@dataclass
class DeidResult:
    records: list[DeidRecord] = field(default_factory=list)
    facts: list[SiteFacts] = field(default_factory=list)
    uid_map: dict[str, str] = field(default_factory=dict)

    @property
    def n_objects(self) -> int:
        return len(self.records)

    @property
    def n_redacted(self) -> int:
        return sum(1 for r in self.records if r.pixel_redacted)

    @property
    def n_nationally_identified(self) -> int:
        """Patients whose pseudonym came from a national ID, so linkage is *possible*.

        Not the same as the number actually imaged at two hospitals — most of
        these appear at one site and simply happen to have an ABHA recorded.
        """
        return len({r.pseudo_patient_id for r in self.records if r.linked_across_sites})

    @property
    def n_cross_site_patients(self) -> int:
        """Patients whose studies actually span more than one site."""
        return sum(1 for sites in self.patient_sites().values() if len(sites) > 1)

    def patient_sites(self) -> dict[str, set[str]]:
        mapping: dict[str, set[str]] = {}
        for record in self.records:
            mapping.setdefault(record.pseudo_patient_id, set()).add(record.site_id)
        return mapping


def _method_code_sequence() -> pydicom.Sequence:
    """Build (0012,0064) from the standard's code values for the options applied."""
    items = []
    for code, meaning in DEIDENTIFICATION_METHOD_CODES:
        item = pydicom.Dataset()
        item.CodeValue = code
        item.CodingSchemeDesignator = "DCM"
        item.CodeMeaning = meaning
        items.append(item)
    return pydicom.Sequence(items)


def _strip_private_and_overlays(ds: pydicom.Dataset) -> int:
    """Remove every private, overlay and curve element. Returns the count removed.

    Private blocks are removed wholesale rather than allowlisted. A vendor may put
    anything in one, and this delivery demonstrably does — site B's block carries
    a duplicate of the patient's name alongside the study date.
    """
    doomed = [
        element.tag
        for element in ds
        if element.tag.group % 2 == 1 or is_overlay_or_curve_group(element.tag.group)
    ]
    for tag in doomed:
        del ds[tag]
    return len(doomed)


def _apply_tag_actions(ds: pydicom.Dataset, pseudo: Pseudonymiser, pseudo_id: str) -> None:
    """Walk the dataset and apply the profile's action for each listed attribute."""
    for element in list(ds):
        entry = action_for(element.tag.group, element.tag.element)
        if entry is None:
            continue
        action, _keyword = entry

        if action is Action.KEEP:
            continue
        if action is Action.REMOVE:
            del ds[element.tag]
        elif action is Action.BLANK:
            ds[element.tag].value = ""
        elif action is Action.DUMMY:
            ds[element.tag].value = DUMMY_STRING
        elif action is Action.PSEUDONYMISE:
            ds[element.tag].value = pseudo_id
        elif action is Action.REMAP_UID:
            current = str(element.value or "")
            if current:
                ds[element.tag].value = pseudo.remap_uid(current)
        elif action is Action.SHIFT_DATE:
            ds[element.tag].value = pseudo.shift_da(str(element.value or ""), pseudo_id)

    # Sequences can nest identifying attributes; recurse into any that survived.
    for element in ds:
        if element.VR == "SQ":
            for item in element.value or []:
                _apply_tag_actions(item, pseudo, pseudo_id)


def deidentify_dataset(
    ds: pydicom.Dataset,
    *,
    pseudo: Pseudonymiser,
    site_id: str,
    age_years: int | None = None,
    detection_params: DetectionParams | None = None,
) -> tuple[pydicom.Dataset, str, bool, list[dict]]:
    """De-identify one dataset in place.

    ``site_id`` scopes the fallback pseudonym for patients with no national
    identifier. It comes from the delivery layout rather than from InstitutionName,
    which is about to be removed and which sites spell inconsistently between
    batches anyway.

    Returns ``(dataset, pseudo_patient_id, linked_across_sites, redacted_regions)``.
    """
    national_id, local_mrn = read_linkage_identifiers(ds)
    pseudo_id, linked = pseudo.patient_pseudonym(
        national_id=national_id or None,
        site_id=site_id,
        local_mrn=local_mrn,
    )

    _strip_private_and_overlays(ds)
    _apply_tag_actions(ds, pseudo, pseudo_id)

    # Clean Pixel Data. Done after the header work so that a failure here cannot
    # leave a half-de-identified header on disk.
    regions: list[dict] = []
    if "PixelData" in ds:
        cleaned, found = clean_pixel_data(ds.pixel_array, detection_params)
        if found:
            regions = [r.to_dict() for r in found]
            ds.PixelData = np.ascontiguousarray(cleaned).tobytes()

    ds.BurnedInAnnotation = "NO"
    if age_years is not None:
        # DICOM AS format: three digits and a unit. Retained under the Retain
        # Patient Characteristics option; the birth date it was derived from is gone.
        ds.PatientAge = f"{age_years:03d}Y"
    ds.PatientIdentityRemoved = PATIENT_IDENTITY_REMOVED
    ds.DeidentificationMethod = list(DEIDENTIFICATION_METHOD)
    ds.DeidentificationMethodCodeSequence = _method_code_sequence()

    # The file-meta SOP Instance UID must track the remapped dataset UID, or the
    # object is internally inconsistent and some readers will reject it.
    if getattr(ds, "file_meta", None) is not None:
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        for tag in (0x00020016, 0x00020100, 0x00020102):  # source AE title, private info
            if tag in ds.file_meta:
                del ds.file_meta[tag]

    return ds, pseudo_id, linked, regions


def deidentify(
    src: Path,
    workspace: Workspace,
    *,
    key: bytes | None = None,
    detection_params: DetectionParams | None = None,
) -> DeidResult:
    """De-identify every object indexed in ``workspace.raw_manifest``."""
    src = Path(src)
    workspace.ensure()
    pseudo = Pseudonymiser(key or load_or_create_key(workspace.key_file))

    result = DeidResult()

    for row in read_jsonl(workspace.raw_manifest):
        source_path = row["source_path"]
        site_id = row["site_id"]
        ds = pydicom.dcmread(src / source_path)

        facts = extract_facts(ds, source_path=source_path, site_id=site_id)
        original_uids = (
            str(ds.StudyInstanceUID),
            str(ds.SeriesInstanceUID),
            str(ds.SOPInstanceUID),
        )

        ds, pseudo_id, linked, regions = deidentify_dataset(
            ds,
            pseudo=pseudo,
            site_id=site_id,
            age_years=facts.age_years,
            detection_params=detection_params,
        )

        # Site identity is not recoverable from the header any more — institution
        # tags are gone — so the store's directory layout carries it instead.
        relative = f"{site_id}/{ds.StudyInstanceUID}/{ds.SOPInstanceUID}.dcm"
        out_path = workspace.deid_store / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_as(out_path, enforce_file_format=True)

        digest = hashlib.sha256(out_path.read_bytes()).hexdigest()

        for original, remapped in zip(
            original_uids,
            (str(ds.StudyInstanceUID), str(ds.SeriesInstanceUID), str(ds.SOPInstanceUID)),
            strict=True,
        ):
            result.uid_map[original] = remapped

        result.facts.append(facts)
        result.records.append(
            DeidRecord(
                relative_path=relative,
                site_id=site_id,
                pseudo_patient_id=pseudo_id,
                linked_across_sites=linked,
                study_uid=str(ds.StudyInstanceUID),
                series_uid=str(ds.SeriesInstanceUID),
                sop_uid=str(ds.SOPInstanceUID),
                sha256=digest,
                rows=int(ds.Rows),
                columns=int(ds.Columns),
                bits_stored=int(ds.BitsStored),
                photometric_interpretation=str(ds.PhotometricInterpretation),
                pixel_redacted=bool(regions),
                redacted_regions=regions,
                source_path=source_path,
            )
        )

    result.records.sort(key=lambda r: (r.site_id, r.relative_path))
    result.facts.sort(key=lambda f: (f.site_id, f.source_path))

    write_jsonl(workspace.deid_manifest, (r.to_dict() for r in result.records))
    write_jsonl(
        workspace.root / "site_facts.jsonl", (f.to_dict() for f in result.facts)
    )
    workspace.uid_map.write_text(
        json.dumps(result.uid_map, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


__all__ = ["DeidRecord", "DeidResult", "deidentify", "deidentify_dataset"]
