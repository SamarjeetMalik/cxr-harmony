"""DICOM PS3.15 Annex E confidentiality profile, as a tag action table.

The configuration implemented here is the **Basic Application Level
Confidentiality Profile** with three named options applied:

* *Clean Pixel Data* — burned-in annotation is detected and blacked out
  (:mod:`cxr_harmony.deid.pixels`).
* *Retain Longitudinal Temporal Information with Modified Dates* — dates are
  shifted by a per-patient offset rather than removed, so the interval between a
  baseline and a follow-up film survives while the calendar date does not.
* *Retain Patient Characteristics* — sex and age are kept, because a chest
  radiograph cohort stratified by neither is not much use for the fairness
  analysis this data exists to support.

Three options are deliberately **not** taken: Retain Device Identity, Retain
Institution Identity, and Retain Safe Private. Device serial numbers and station
names identify a room and therefore, in a small hospital, a patient. Private
blocks are removed wholesale rather than allowlisted, because a vendor is free to
put anything in one and this delivery demonstrably does — see the site B block
that carries a duplicate of the patient's name.

Action codes follow the standard: ``X`` remove, ``Z`` replace with a zero-length
value, ``D`` replace with a non-zero dummy, ``U`` replace with a remapped UID,
``K`` keep. ``SHIFT`` is this implementation's realisation of the Modified Dates
option. Attributes absent from the table are retained, per the standard; private
attributes and overlays are handled separately and unconditionally.
"""

from __future__ import annotations

from enum import Enum


class Action(str, Enum):
    """What to do with one attribute."""

    REMOVE = "X"
    BLANK = "Z"
    DUMMY = "D"
    REMAP_UID = "U"
    KEEP = "K"
    SHIFT_DATE = "SHIFT"
    PSEUDONYMISE = "PSEUDO"


#: ``(group, element) -> (action, keyword)``. The keyword is carried for readable
#: audit output; the tag numbers are what is actually matched.
TAG_ACTIONS: dict[tuple[int, int], tuple[Action, str]] = {
    # --- Identity of the patient -------------------------------------------
    (0x0010, 0x0010): (Action.PSEUDONYMISE, "PatientName"),
    (0x0010, 0x0020): (Action.PSEUDONYMISE, "PatientID"),
    (0x0010, 0x0021): (Action.REMOVE, "IssuerOfPatientID"),
    (0x0010, 0x0030): (Action.REMOVE, "PatientBirthDate"),
    (0x0010, 0x0032): (Action.REMOVE, "PatientBirthTime"),
    (0x0010, 0x0050): (Action.REMOVE, "PatientInsurancePlanCodeSequence"),
    (0x0010, 0x1000): (Action.REMOVE, "OtherPatientIDs"),
    (0x0010, 0x1001): (Action.REMOVE, "OtherPatientNames"),
    (0x0010, 0x1002): (Action.REMOVE, "OtherPatientIDsSequence"),
    (0x0010, 0x1005): (Action.REMOVE, "PatientBirthName"),
    (0x0010, 0x1040): (Action.REMOVE, "PatientAddress"),
    (0x0010, 0x1060): (Action.REMOVE, "PatientMotherBirthName"),
    (0x0010, 0x2154): (Action.REMOVE, "PatientTelephoneNumbers"),
    (0x0010, 0x2160): (Action.REMOVE, "EthnicGroup"),
    (0x0010, 0x21B0): (Action.REMOVE, "AdditionalPatientHistory"),
    (0x0010, 0x21C0): (Action.REMOVE, "PregnancyStatus"),
    (0x0010, 0x4000): (Action.REMOVE, "PatientComments"),
    # Retained under the Retain Patient Characteristics option.
    (0x0010, 0x0040): (Action.KEEP, "PatientSex"),
    (0x0010, 0x1010): (Action.KEEP, "PatientAge"),
    (0x0010, 0x1020): (Action.KEEP, "PatientSize"),
    (0x0010, 0x1030): (Action.KEEP, "PatientWeight"),
    # --- Identity of clinicians and institutions ----------------------------
    (0x0008, 0x0080): (Action.REMOVE, "InstitutionName"),
    (0x0008, 0x0081): (Action.REMOVE, "InstitutionAddress"),
    (0x0008, 0x0082): (Action.REMOVE, "InstitutionCodeSequence"),
    (0x0008, 0x0090): (Action.BLANK, "ReferringPhysicianName"),
    (0x0008, 0x0092): (Action.REMOVE, "ReferringPhysicianAddress"),
    (0x0008, 0x0094): (Action.REMOVE, "ReferringPhysicianTelephoneNumbers"),
    (0x0008, 0x0096): (Action.REMOVE, "ReferringPhysicianIdentificationSequence"),
    (0x0008, 0x1040): (Action.REMOVE, "InstitutionalDepartmentName"),
    (0x0008, 0x1048): (Action.REMOVE, "PhysiciansOfRecord"),
    (0x0008, 0x1049): (Action.REMOVE, "PhysiciansOfRecordIdentificationSequence"),
    (0x0008, 0x1050): (Action.REMOVE, "PerformingPhysicianName"),
    (0x0008, 0x1052): (Action.REMOVE, "PerformingPhysicianIdentificationSequence"),
    (0x0008, 0x1060): (Action.REMOVE, "NameOfPhysiciansReadingStudy"),
    (0x0008, 0x1062): (Action.REMOVE, "PhysiciansReadingStudyIdentificationSequence"),
    (0x0008, 0x1070): (Action.REMOVE, "OperatorsName"),
    (0x0008, 0x1072): (Action.REMOVE, "OperatorIdentificationSequence"),
    (0x0032, 0x1032): (Action.REMOVE, "RequestingPhysician"),
    (0x0032, 0x1033): (Action.REMOVE, "RequestingService"),
    (0x0040, 0xA075): (Action.REMOVE, "ContentCreatorName"),
    (0x4008, 0x0114): (Action.REMOVE, "PhysicianApprovingInterpretation"),
    (0x4008, 0x011A): (Action.REMOVE, "PhysicianReadingStudyIdentificationSequence"),
    # --- Accession, order and visit identifiers ------------------------------
    (0x0008, 0x0050): (Action.BLANK, "AccessionNumber"),
    (0x0020, 0x0010): (Action.BLANK, "StudyID"),
    (0x0038, 0x0010): (Action.REMOVE, "AdmissionID"),
    (0x0038, 0x0011): (Action.REMOVE, "IssuerOfAdmissionID"),
    (0x0038, 0x0300): (Action.REMOVE, "CurrentPatientLocation"),
    (0x0038, 0x0400): (Action.REMOVE, "PatientInstitutionResidence"),
    (0x0038, 0x0500): (Action.REMOVE, "PatientState"),
    (0x0040, 0x1001): (Action.REMOVE, "RequestedProcedureID"),
    (0x0040, 0x2016): (Action.REMOVE, "PlacerOrderNumberImagingServiceRequest"),
    (0x0040, 0x2017): (Action.REMOVE, "FillerOrderNumberImagingServiceRequest"),
    # --- Free-text descriptors that routinely carry names and dates ----------
    (0x0008, 0x1030): (Action.REMOVE, "StudyDescription"),
    (0x0008, 0x103E): (Action.REMOVE, "SeriesDescription"),
    (0x0008, 0x2111): (Action.REMOVE, "DerivationDescription"),
    (0x0018, 0x1030): (Action.REMOVE, "ProtocolName"),
    (0x0020, 0x4000): (Action.REMOVE, "ImageComments"),
    (0x0032, 0x1060): (Action.REMOVE, "RequestedProcedureDescription"),
    (0x0032, 0x4000): (Action.REMOVE, "StudyComments"),
    (0x0040, 0x0254): (Action.REMOVE, "PerformedProcedureStepDescription"),
    (0x0008, 0x1080): (Action.REMOVE, "AdmittingDiagnosesDescription"),
    # --- Device identity (option not taken) ----------------------------------
    (0x0008, 0x1010): (Action.REMOVE, "StationName"),
    (0x0018, 0x1000): (Action.REMOVE, "DeviceSerialNumber"),
    (0x0018, 0x1002): (Action.REMOVE, "DeviceUID"),
    (0x0018, 0x1020): (Action.REMOVE, "SoftwareVersions"),
    (0x0018, 0x700A): (Action.REMOVE, "DetectorID"),
    # --- UIDs -----------------------------------------------------------------
    (0x0008, 0x0018): (Action.REMAP_UID, "SOPInstanceUID"),
    (0x0020, 0x000D): (Action.REMAP_UID, "StudyInstanceUID"),
    (0x0020, 0x000E): (Action.REMAP_UID, "SeriesInstanceUID"),
    (0x0020, 0x0052): (Action.REMAP_UID, "FrameOfReferenceUID"),
    (0x0088, 0x0140): (Action.REMAP_UID, "StorageMediaFileSetUID"),
    (0x0008, 0x1155): (Action.REMAP_UID, "ReferencedSOPInstanceUID"),
    # --- Dates, shifted rather than destroyed ---------------------------------
    (0x0008, 0x0012): (Action.SHIFT_DATE, "InstanceCreationDate"),
    (0x0008, 0x0020): (Action.SHIFT_DATE, "StudyDate"),
    (0x0008, 0x0021): (Action.SHIFT_DATE, "SeriesDate"),
    (0x0008, 0x0022): (Action.SHIFT_DATE, "AcquisitionDate"),
    (0x0008, 0x0023): (Action.SHIFT_DATE, "ContentDate"),
    (0x0038, 0x0020): (Action.SHIFT_DATE, "AdmittingDate"),
    (0x0040, 0x0244): (Action.SHIFT_DATE, "PerformedProcedureStepStartDate"),
}

#: Group numbers whose contents are removed wholesale. Overlays (0x60xx) can
#: carry burned-in annotation as a bitmap plane, and curve data is obsolete but
#: still occasionally present.
def is_overlay_or_curve_group(group: int) -> bool:
    return 0x6000 <= group <= 0x60FF or 0x5000 <= group <= 0x50FF


#: Value written where a dummy is required.
DUMMY_STRING = "ANONYMISED"

#: Written into (0012,0062) so a downstream reader can see the object has been
#: de-identified.
PATIENT_IDENTITY_REMOVED = "YES"

#: (0012,0063) DeidentificationMethod is VR LO, VM 1-n: a *list* of values each
#: within the 64-character LO limit, not one long sentence. Writing a single
#: joined string produces a non-conformant object that some readers truncate.
DEIDENTIFICATION_METHOD: tuple[str, ...] = (
    "PS3.15 Annex E Basic Application Confidentiality Profile",
    "Clean Pixel Data Option",
    "Retain Longitudinal Temporal Information Modified Dates",
    "Retain Patient Characteristics Option",
)

#: (0012,0064) DeidentificationMethodCodeSequence, using the DCM code values the
#: standard defines for these options. The coded form is what a downstream tool
#: can act on; the free-text above is for a human reading the header.
DEIDENTIFICATION_METHOD_CODES: tuple[tuple[str, str], ...] = (
    ("113100", "Basic Application Confidentiality Profile"),
    ("113101", "Clean Pixel Data Option"),
    ("113107", "Retain Longitudinal Temporal Information Modified Dates Option"),
    ("113108", "Retain Patient Characteristics Option"),
)


def action_for(group: int, element: int) -> tuple[Action, str] | None:
    """Return the profile's action for a tag, or ``None`` if it is unlisted."""
    return TAG_ACTIONS.get((group, element))
