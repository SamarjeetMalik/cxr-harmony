"""Generate a synthetic multi-site chest radiograph delivery.

The output mimics what three partner hospitals would actually hand over: DICOM
files carrying live direct identifiers, free-text reports, and — for one site — a
sidecar CSV of labels, each in the site's own conventions.

A ``_ground_truth.json`` file is written alongside the delivery. It records which
patients are the same person across sites and which findings were planted, so
that de-identification and label harmonisation can be scored rather than eyeballed.
No such file exists for a real delivery; it is an evaluation artefact.
"""

from __future__ import annotations

import csv
import json
import random
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian

from ..schema.vocab import Finding
from .identifiers import (
    make_abha_number,
    make_address,
    make_person_name,
    make_phone,
    make_physician_name,
)
from .pixels import BITS_STORED, IMAGE_SIZE, burn_in_text, synthesise_radiograph
from .reports import build_report
from .sites import SITES, SITES_BY_ID, SiteProfile

#: A synthetic UID arc. Deterministic construction from counters, rather than
#: ``generate_uid()``, is what makes a seeded run byte-reproducible.
UID_ROOT = "1.2.826.0.1.3680043.10.1337"

#: Digital X-Ray Image Storage - For Presentation, and Computed Radiography.
DX_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.1.1"
CR_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.1"

_POSITIVE_FINDINGS = tuple(f for f in Finding if f not in (Finding.NO_FINDING, Finding.OTHER))


@dataclass
class SynthPatient:
    """A generated person, possibly imaged at more than one site."""

    index: int
    dicom_name: str
    display_name: str
    sex: str
    birth_date: date
    abha: str | None
    #: ``site_id -> local MRN``. More than one entry means a cross-site patient.
    mrns: dict[str, str] = field(default_factory=dict)


@dataclass
class SynthStudy:
    """One generated encounter at one site."""

    patient: SynthPatient
    site: SiteProfile
    study_date: date
    accession: str
    findings: list[Finding]
    view: str
    study_uid: str
    series_uid: str
    sop_uid: str
    referring_physician: str
    reporting_radiologist: str
    burned_in: bool
    prior_date: date | None


def _uid(kind: int, counter: int) -> str:
    return f"{UID_ROOT}.{kind}.{counter}"


def _format_da(value: date) -> str:
    return value.strftime("%Y%m%d")


def _build_patients(rng: random.Random, n_patients: int, n_cross_site: int) -> list[SynthPatient]:
    patients: list[SynthPatient] = []
    for i in range(n_patients):
        sex = rng.choice(("M", "F"))
        dicom_name, display_name = make_person_name(rng, sex)
        # Ages skew adult; a handful of paediatric and very elderly cases are included
        # so that the 89-year cap in the schema is actually exercised.
        age = rng.choice(
            [rng.randint(1, 17)] * 1 + [rng.randint(18, 64)] * 6 + [rng.randint(65, 96)] * 3
        )
        birth = date(2026, 1, 1) - timedelta(days=age * 365 + rng.randint(0, 364))
        patients.append(
            SynthPatient(
                index=i,
                dicom_name=dicom_name,
                display_name=display_name,
                sex=sex,
                birth_date=birth,
                abha=None,
            )
        )

    # Cross-site patients must carry a national identifier, since local MRNs cannot
    # link across hospitals. They are chosen first so the overlap is guaranteed.
    cross_site = rng.sample(patients, k=min(n_cross_site, len(patients)))
    for p in cross_site:
        p.abha = make_abha_number(rng)

    for p in patients:
        if p.abha is None and rng.random() < 0.55:
            p.abha = make_abha_number(rng)

    return patients


def _assign_studies(
    rng: random.Random,
    patients: list[SynthPatient],
    cross_site_ids: set[int],
) -> list[SynthStudy]:
    studies: list[SynthStudy] = []
    mrn_counters = {s.site_id: 1000 for s in SITES}
    accession_counter = 100000
    uid_counter = 1

    for patient in patients:
        if patient.index in cross_site_ids:
            sites = rng.sample(SITES, k=2)
        else:
            sites = [rng.choice(SITES)]

        prior: date | None = None
        for site in sites:
            mrn_counters[site.site_id] += 1
            n_studies = rng.choices([1, 2, 3], weights=[6, 3, 1])[0]
            base_date = date(2023, 1, 1) + timedelta(days=rng.randint(0, 900))
            # A paediatric patient can otherwise be handed a study that predates
            # their own birth, which produces an unresolvable age downstream.
            earliest = patient.birth_date + timedelta(days=1)
            if base_date < earliest:
                base_date = earliest

            mrn = site.mrn_template.format(n=mrn_counters[site.site_id], year=base_date.year)
            patient.mrns[site.site_id] = mrn

            for k in range(n_studies):
                study_date = base_date + timedelta(days=k * rng.randint(20, 240))
                if study_date > date(2025, 12, 31):
                    study_date = date(2025, 12, 31)

                n_pos = rng.choices([0, 1, 2], weights=[5, 4, 1])[0]
                findings = (
                    [Finding.NO_FINDING]
                    if n_pos == 0
                    else list(rng.sample(_POSITIVE_FINDINGS, k=n_pos))
                )

                accession_counter += 1
                ref_dicom, ref_display = make_physician_name(rng)
                rad_dicom, rad_display = make_physician_name(rng)

                studies.append(
                    SynthStudy(
                        patient=patient,
                        site=site,
                        study_date=study_date,
                        accession=f"ACC{accession_counter}",
                        findings=findings,
                        view=rng.choices(["PA", "AP", "LATERAL"], weights=[6, 3, 1])[0],
                        study_uid=_uid(1, uid_counter),
                        series_uid=_uid(2, uid_counter),
                        sop_uid=_uid(3, uid_counter),
                        referring_physician=ref_display,
                        reporting_radiologist=rad_display,
                        burned_in=rng.random() < site.burn_in_rate,
                        prior_date=prior,
                    )
                )
                uid_counter += 1
                prior = study_date

    studies.sort(key=lambda s: (s.site.site_id, s.accession))
    return studies


def _build_dataset(study: SynthStudy, pixels: np.ndarray, phone: str, address: str) -> Dataset:
    """Assemble a DICOM dataset carrying the site's conventions and its PHI."""
    site = study.site
    p = study.patient
    is_cr = site.site_id == "SITE_C"

    ds = Dataset()

    # --- Direct identifiers. Everything in this block must not survive de-identification.
    ds.PatientName = p.dicom_name
    ds.PatientID = p.mrns[site.site_id]
    ds.PatientBirthDate = _format_da(p.birth_date)
    ds.PatientSex = site.sex_encoding[p.sex]
    ds.AccessionNumber = study.accession
    ds.ReferringPhysicianName = study.referring_physician.replace("Dr. ", "").replace(" ", "^")
    ds.PerformingPhysicianName = study.reporting_radiologist.replace("Dr. ", "").replace(" ", "^")
    ds.OperatorsName = "TECH^" + str(1000 + (study.patient.index % 40))
    ds.InstitutionName = site.institution_name
    ds.InstitutionAddress = site.institution_address
    ds.PatientAddress = address
    ds.PatientTelephoneNumbers = phone
    if p.abha:
        ds.OtherPatientIDs = p.abha
    ds.StudyID = study.accession[-6:]

    # --- Dates, encoded per the site's own habit.
    if site.date_channel == "standard":
        ds.StudyDate = _format_da(study.study_date)
        ds.SeriesDate = _format_da(study.study_date)
        ds.AcquisitionDate = _format_da(study.study_date)
        ds.ContentDate = _format_da(study.study_date)
    else:
        # Site B leaves the standard date tags empty and files the date in a
        # private block in DD-MM-YYYY, which is a legacy RIS export habit.
        ds.StudyDate = ""
        ds.ContentDate = ""
    ds.StudyTime = "0930"

    # --- Study / series identity.
    ds.StudyInstanceUID = study.study_uid
    ds.SeriesInstanceUID = study.series_uid
    ds.SOPInstanceUID = study.sop_uid
    ds.SOPClassUID = CR_SOP_CLASS if is_cr else DX_SOP_CLASS
    ds.Modality = "CR" if is_cr else "DX"
    ds.BodyPartExamined = site.body_part
    ds.SeriesNumber = 1
    ds.InstanceNumber = 1
    ds.PatientOrientation = ""

    # --- Projection, wherever this site chooses to record it.
    native_view = site.view_encoding[study.view]
    if site.view_channel == "view_position_tag":
        ds.ViewPosition = native_view
        ds.SeriesDescription = "CHEST"
    else:
        ds.ViewPosition = ""
        ds.SeriesDescription = native_view

    if site.laterality_encoding and study.view == "LATERAL":
        ds.Laterality = site.laterality_encoding[
            "L" if study.patient.index % 2 == 0 else "R"
        ]

    ds.PatientPosition = "ERECT" if study.view != "AP" else "SUPINE"

    # --- Labels, for the site that ships them in the header.
    if site.label_channel == "image_comments":
        ds.ImageComments = ";".join(site.label_encoding[f] for f in study.findings)

    # --- Private block. Site B's date lives here; the block also carries a
    # duplicate of the patient name, which is exactly why blanket private-tag
    # removal is part of the profile rather than an optional extra.
    if site.date_channel == "private_ddmmyyyy":
        block = ds.private_block(0x0033, "MIMS RIS EXPORT", create=True)
        block.add_new(0x01, "LO", study.study_date.strftime("%d-%m-%Y"))
        block.add_new(0x02, "LO", p.display_name)
        block.add_new(0x03, "LO", p.mrns[site.site_id])

    # --- Pixel data.
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns = pixels.shape
    ds.BitsAllocated = 16
    ds.BitsStored = BITS_STORED
    ds.HighBit = BITS_STORED - 1
    ds.PixelRepresentation = 0
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.BurnedInAnnotation = "YES" if study.burned_in else "NO"
    ds.PixelData = pixels.tobytes()

    # --- Nested sequences.
    #
    # Identifiers do not only live at the top level. Enhanced objects, structured
    # reports and ordinary scheduled-procedure records all nest them one or two
    # levels down, and a de-identifier that walks only the top level leaves them
    # untouched while reporting success. These are generated so the recursion is
    # exercised against something rather than assumed.
    request_item = Dataset()
    request_item.AccessionNumber = study.accession
    request_item.RequestedProcedureID = f"RP{study.accession[-5:]}"
    request_item.RequestingPhysician = study.referring_physician.replace("Dr. ", "")
    request_item.StudyInstanceUID = study.study_uid

    # A sequence inside a sequence: the level a shallow recursion misses.
    referenced_item = Dataset()
    referenced_item.ReferencedSOPInstanceUID = study.sop_uid
    referenced_item.ReferencedSOPClassUID = ds.SOPClassUID

    # And a third level, carrying an identifier of a *different kind* from the
    # two above it. Depth two is where a hand-written recursion usually stops
    # being wrong, so testing only to depth two cannot distinguish real recursion
    # from a loop that happens to run twice. The identifier here is a name rather
    # than a UID so the level is distinguishable in a failure: a surviving
    # physician name at depth three is unambiguous, whereas a surviving UID could
    # have leaked from either of the levels above.
    purpose_item = Dataset()
    purpose_item.ReferencedSOPInstanceUID = study.sop_uid
    purpose_item.RequestingPhysician = study.referring_physician.replace("Dr. ", "")
    referenced_item.PurposeOfReferenceCodeSequence = Sequence([purpose_item])

    request_item.ReferencedImageSequence = Sequence([referenced_item])

    ds.RequestAttributesSequence = Sequence([request_item])

    equipment_item = Dataset()
    equipment_item.InstitutionName = site.institution_name
    equipment_item.InstitutionAddress = site.institution_address
    equipment_item.StationName = f"{site.site_id}-CONSOLE-1"
    equipment_item.DeviceSerialNumber = f"SN{1000 + study.patient.index}"
    ds.ContributingEquipmentSequence = Sequence([equipment_item])

    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.ImplementationClassUID = f"{UID_ROOT}.0.1"
    ds.file_meta.SourceApplicationEntityTitle = f"{site.site_id}_PACS"

    return ds


def generate_corpus(
    out_dir: Path,
    *,
    seed: int = 20260731,
    n_patients: int = 48,
    n_cross_site: int = 8,
    image_size: int = IMAGE_SIZE,
) -> dict:
    """Write a three-site delivery to ``out_dir``. Returns the ground-truth record."""
    out_dir = Path(out_dir)
    rng = random.Random(seed)
    npr = np.random.default_rng(seed)

    patients = _build_patients(rng, n_patients, n_cross_site)
    # _build_patients guarantees the first n_cross_site sampled patients carry a
    # national identifier; those are the ones imaged at two hospitals.
    cross_site_ids = {p.index for p in patients if p.abha}
    cross_site_ids = set(sorted(cross_site_ids)[:n_cross_site])

    studies = _assign_studies(rng, patients, cross_site_ids)

    for site in SITES:
        (out_dir / site.site_id / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / site.site_id / "reports").mkdir(parents=True, exist_ok=True)

    sidecar_rows: dict[str, list[dict]] = {s.site_id: [] for s in SITES}

    for study in studies:
        site = study.site
        p = study.patient
        age = int((study.study_date - p.birth_date).days // 365)
        phone = make_phone(rng)
        address = make_address(rng, site.city)

        pixels = synthesise_radiograph(npr, size=image_size)
        if study.burned_in:
            pixels = burn_in_text(
                pixels,
                [
                    p.display_name.upper(),
                    f"MRN {p.mrns[site.site_id]}",
                    study.study_date.strftime("%d-%m-%Y"),
                ],
            )

        image_path = out_dir / site.site_id / "images" / f"{study.accession}.dcm"
        # Site C emits a legacy 'P->A' beam-direction string, which is not a
        # conformant CS value. That non-conformance is the point — a real archive
        # contains it and the pipeline has to survive it — so pydicom's validation
        # warning is suppressed here rather than the data being sanitised.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Invalid value for VR CS")
            ds = _build_dataset(study, pixels, phone, address)
            ds.save_as(image_path, enforce_file_format=True)

        date_text = (
            study.study_date.strftime("%d-%m-%Y")
            if site.date_channel == "private_ddmmyyyy"
            else study.study_date.strftime("%d %b %Y")
        )
        report = build_report(
            rng=rng,
            findings=study.findings,
            patient_display_name=p.display_name,
            mrn=p.mrns[site.site_id],
            abha=p.abha,
            age=age,
            sex=p.sex,
            study_date_text=date_text,
            accession=study.accession,
            referring_physician=f"Dr. {study.referring_physician.replace('Dr. ', '')}",
            reporting_radiologist=f"Dr. {study.reporting_radiologist.replace('Dr. ', '')}",
            institution_name=site.institution_name,
            institution_address=site.institution_address,
            phone=phone,
            examination=site.view_encoding[study.view],
            prior_study_date=(
                study.prior_date.strftime("%d-%m-%Y") if study.prior_date else None
            ),
        )
        (out_dir / site.site_id / "reports" / f"{study.accession}.txt").write_text(
            report, encoding="utf-8"
        )

        if site.label_channel == "sidecar_csv":
            sidecar_rows[site.site_id].append(
                {
                    "accession_no": study.accession,
                    "mrn": p.mrns[site.site_id],
                    "study_dt": study.study_date.strftime("%d-%m-%Y"),
                    "view": site.view_encoding[study.view],
                    "findings_codes": "|".join(
                        site.label_encoding[f] for f in study.findings
                    ),
                }
            )

    for site_id, rows in sidecar_rows.items():
        if not rows:
            continue
        path = out_dir / site_id / "labels.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    ground_truth = {
        "seed": seed,
        "n_patients": len(patients),
        "n_studies": len(studies),
        "cross_site_patients": [
            {
                "abha": p.abha,
                "sites": sorted(p.mrns.keys()),
                "mrns": p.mrns,
            }
            for p in patients
            if len(p.mrns) > 1
        ],
        "studies": [
            {
                "accession": s.accession,
                "site_id": s.site.site_id,
                "study_uid": s.study_uid,
                "view": s.view,
                "findings": [f.value for f in s.findings],
                "burned_in": s.burned_in,
                "study_date": s.study_date.isoformat(),
                "patient_index": s.patient.index,
            }
            for s in studies
        ],
        "phi_values": sorted(
            {p.display_name for p in patients}
            | {m for p in patients for m in p.mrns.values()}
            | {p.abha for p in patients if p.abha}
            | {s.institution_name for s in SITES}
        ),
    }
    (out_dir / "_ground_truth.json").write_text(
        json.dumps(ground_truth, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return ground_truth


__all__ = ["generate_corpus", "SITES_BY_ID", "UID_ROOT"]
