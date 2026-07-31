"""Export the catalogue as FHIR R4 resources.

The catalogue is the pipeline's own shape. FHIR is the shape the rest of a
hospital speaks, and a cohort that cannot be handed to a clinical system in a
standard form is a cohort that lives permanently in one team's tooling.

Three resource types, which is what a chest-radiograph cohort actually needs:

* ``ImagingStudy`` — the study, its series, and their instances, with the
  modality and projection.
* ``DiagnosticReport`` — the report, its conclusion, and references to the study
  and to the observations drawn from it.
* ``Observation`` — one per finding, coded so a downstream system can act on it
  rather than parse prose.

**No patient names, and no ``Patient`` resource.** FHIR's data model expects
``ImagingStudy.subject`` to reference a ``Patient``, and the obvious thing is to
emit one. It is not emitted here, because a ``Patient`` resource is a container
designed to hold a name, a birth date and an address, and offering one invites
the next person to populate it. The subject is instead a reference carrying only
the pseudonym, with the display name deliberately absent.

This module builds JSON and nothing else — no server, no network, no
``fhirclient`` dependency. A bundle it produces can be POSTed to a HAPI FHIR
server, but that is the deployer's business.

SNOMED CT codes are used where the finding has an unambiguous one. Where it does
not, a local code in this project's own system is emitted rather than a plausible
guess: a wrong SNOMED code is worse than an honest local one, because it looks
authoritative to everything downstream.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..schema.models import CanonicalDataset
from ..schema.vocab import Finding, LabelSource, ViewPosition

#: Local coding system, for findings with no unambiguous SNOMED equivalent.
LOCAL_SYSTEM = "https://github.com/SamarjeetMalik/cxr-harmony/fhir/CodeSystem/finding"
SNOMED = "http://snomed.info/sct"
DICOM_DCM = "http://dicom.nema.org/resources/ontology/DCM"
LOINC = "http://loinc.org"

#: SNOMED CT concepts, used only where the mapping is unambiguous.
FINDING_CODES: dict[Finding, tuple[str, str, str]] = {
    Finding.CARDIOMEGALY: (SNOMED, "8186001", "Cardiomegaly"),
    Finding.PLEURAL_EFFUSION: (SNOMED, "60046008", "Pleural effusion"),
    Finding.CONSOLIDATION: (SNOMED, "95436008", "Pulmonary consolidation"),
    Finding.PNEUMOTHORAX: (SNOMED, "36118008", "Pneumothorax"),
    Finding.PULMONARY_EDEMA: (SNOMED, "19242006", "Pulmonary oedema"),
    Finding.ATELECTASIS: (SNOMED, "46621007", "Atelectasis"),
    Finding.NODULE: (SNOMED, "27925004", "Nodule of lung"),
    Finding.FRACTURE: (SNOMED, "125605004", "Fracture of bone"),
    Finding.TUBERCULOSIS: (SNOMED, "154283005", "Pulmonary tuberculosis"),
    Finding.NO_FINDING: (SNOMED, "17621005", "Normal"),
    # Deliberately local: "an abnormality outside the canonical vocabulary" has no
    # SNOMED concept, and inventing one would be a lie a downstream system trusts.
    Finding.OTHER: (LOCAL_SYSTEM, "OTHER", "Abnormality outside the canonical vocabulary"),
}

#: DICOM projection codes.
VIEW_CODES: dict[ViewPosition, tuple[str, str]] = {
    ViewPosition.PA: ("R-10206", "postero-anterior"),
    ViewPosition.AP: ("R-10214", "antero-posterior"),
    ViewPosition.LATERAL: ("R-10232", "lateral"),
}


@dataclass
class BundleStats:
    n_studies: int
    n_reports: int
    n_observations: int

    def to_dict(self) -> dict:
        return {
            "n_studies": self.n_studies,
            "n_reports": self.n_reports,
            "n_observations": self.n_observations,
            "n_entries": self.n_studies + self.n_reports + self.n_observations,
        }


def _subject_reference(pseudo_id: str) -> dict[str, Any]:
    """A subject reference that carries the pseudonym and nothing else.

    No ``display``. FHIR permits a human-readable label there and every viewer
    shows it, which makes it exactly the field a name ends up in.
    """
    return {"identifier": {"system": f"{LOCAL_SYSTEM}/pseudonym", "value": pseudo_id}}


def imaging_study_resource(dataset: CanonicalDataset, study_uid: str) -> dict[str, Any]:
    """Build one ``ImagingStudy``."""
    study = next(s for s in dataset.studies if s.study_uid == study_uid)
    series = [s for s in dataset.series if s.study_uid == study_uid]
    by_series = {s.series_uid: s for s in series}

    instances: dict[str, list] = {}
    for instance in dataset.instances:
        if instance.series_uid in by_series:
            instances.setdefault(instance.series_uid, []).append(instance)

    series_entries = []
    for s in series:
        entry: dict[str, Any] = {
            "uid": s.series_uid,
            "modality": {"system": DICOM_DCM, "code": study.modality},
            "numberOfInstances": len(instances.get(s.series_uid, [])),
            "instance": [
                {
                    "uid": i.sop_uid,
                    "sopClass": {
                        "system": "urn:ietf:rfc:3986",
                        "code": "urn:oid:1.2.840.10008.5.1.4.1.1.1",
                    },
                }
                for i in sorted(instances.get(s.series_uid, []), key=lambda i: i.sop_uid)
            ],
        }
        code = VIEW_CODES.get(s.view_position)
        if code is not None:
            entry["bodySite"] = {
                "system": SNOMED,
                "code": "51185008",
                "display": "Thoracic structure",
            }
            entry["description"] = f"Chest {code[1]}"
        series_entries.append(entry)

    resource: dict[str, Any] = {
        "resourceType": "ImagingStudy",
        "id": _fhir_id(study_uid),
        "identifier": [{"system": "urn:dicom:uid", "value": f"urn:oid:{study_uid}"}],
        "status": "available",
        "subject": _subject_reference(study.pseudo_patient_id),
        "modality": [{"system": DICOM_DCM, "code": study.modality}],
        "numberOfSeries": len(series),
        "numberOfInstances": sum(len(v) for v in instances.values()),
        "series": series_entries,
    }
    if study.study_date is not None:
        # Date-only, and already shifted by a per-patient offset upstream. The
        # interval between a patient's studies is real; the calendar date is not.
        resource["started"] = study.study_date.isoformat()
    return resource


def observation_resources(dataset: CanonicalDataset, study_uid: str) -> list[dict[str, Any]]:
    """One ``Observation`` per finding on a study."""
    out = []
    for index, label in enumerate(
        sorted(
            (label for label in dataset.labels if label.study_uid == study_uid),
            key=lambda label: label.finding.value,
        )
    ):
        study = next(s for s in dataset.studies if s.study_uid == study_uid)
        system, code, display = FINDING_CODES[label.finding]
        resource = {
            "resourceType": "Observation",
            "id": f"{_fhir_id(study_uid)}-obs-{index}",
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "imaging",
                        }
                    ]
                }
            ],
            "code": {"coding": [{"system": system, "code": code, "display": display}]},
            "subject": _subject_reference(study.pseudo_patient_id),
            "valueBoolean": bool(label.present),
            # Provenance travels with the observation. A label a rule pulled out of
            # prose is weaker evidence than one a site exported, and a consumer that
            # cannot tell them apart will treat them as equal.
            "method": {
                "coding": [
                    {
                        "system": f"{LOCAL_SYSTEM}/method",
                        "code": label.source.value,
                        "display": _method_display(label.source),
                    }
                ]
            },
        }
        out.append(resource)
    return out


def diagnostic_report_resource(
    dataset: CanonicalDataset, study_uid: str, observation_ids: list[str]
) -> dict[str, Any] | None:
    """Build the ``DiagnosticReport``, or ``None`` if the study has no report."""
    report = next((r for r in dataset.reports if r.study_uid == study_uid), None)
    if report is None:
        return None
    study = next(s for s in dataset.studies if s.study_uid == study_uid)

    resource: dict[str, Any] = {
        "resourceType": "DiagnosticReport",
        "id": f"{_fhir_id(study_uid)}-report",
        "status": "final",
        "category": [{"coding": [{"system": DICOM_DCM, "code": "CR", "display": "Radiology"}]}],
        "code": {
            "coding": [{"system": LOINC, "code": "30746-2", "display": "Chest X-ray study"}]
        },
        "subject": _subject_reference(study.pseudo_patient_id),
        "imagingStudy": [{"reference": f"ImagingStudy/{_fhir_id(study_uid)}"}],
        "result": [{"reference": f"Observation/{oid}"} for oid in observation_ids],
    }
    if study.study_date is not None:
        resource["effectiveDateTime"] = study.study_date.isoformat()

    # The impression only. Findings prose is the richest residual
    # re-identification surface in the cohort and does not belong in a resource
    # that will be handed to another system by default.
    from ..schema.vocab import ReportSection

    impression = report.sections.get(ReportSection.IMPRESSION, "").strip()
    if impression:
        resource["conclusion"] = impression
    return resource


def build_bundle(dataset: CanonicalDataset) -> tuple[dict[str, Any], BundleStats]:
    """Build a FHIR R4 ``collection`` Bundle for the whole cohort."""
    entries: list[dict[str, Any]] = []
    n_reports = n_observations = 0

    for study in sorted(dataset.studies, key=lambda s: s.study_uid):
        entries.append(_entry(imaging_study_resource(dataset, study.study_uid)))

        observations = observation_resources(dataset, study.study_uid)
        for observation in observations:
            entries.append(_entry(observation))
        n_observations += len(observations)

        report = diagnostic_report_resource(
            dataset, study.study_uid, [o["id"] for o in observations]
        )
        if report is not None:
            entries.append(_entry(report))
            n_reports += 1

    bundle = {"resourceType": "Bundle", "type": "collection", "entry": entries}
    return bundle, BundleStats(len(dataset.studies), n_reports, n_observations)


def write_bundle(dataset: CanonicalDataset, out_dir: Path) -> tuple[Path, BundleStats]:
    """Write ``bundle.json`` into ``out_dir``."""
    bundle, stats = build_bundle(dataset)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "bundle.json"
    path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return path, stats


def _entry(resource: dict[str, Any]) -> dict[str, Any]:
    return {
        "fullUrl": f"urn:uuid:{resource['resourceType'].lower()}-{resource['id']}",
        "resource": resource,
    }


def _fhir_id(uid: str) -> str:
    """A FHIR id: at most 64 characters of ``[A-Za-z0-9.-]``.

    A DICOM UID is already legal, but the leading digits and dots are awkward in
    URLs, so it is prefixed.
    """
    return f"s{uid.replace('.', '-')}"[:64]


def _method_display(source: LabelSource) -> str:
    return {
        LabelSource.SITE_STRUCTURED: "Structured export from the contributing site",
        LabelSource.REPORT_RULE: "Rule-based extraction from report text",
        LabelSource.RADIOLOGIST: "Radiologist adjudication",
    }[source]


__all__ = [
    "FINDING_CODES",
    "LOCAL_SYSTEM",
    "BundleStats",
    "build_bundle",
    "diagnostic_report_resource",
    "imaging_study_resource",
    "observation_resources",
    "write_bundle",
]
