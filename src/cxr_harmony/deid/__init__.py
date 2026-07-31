"""De-identification: PS3.15 Annex E profile, pseudonymisation, and pixel cleaning."""

from .engine import DeidRecord, DeidResult, deidentify, deidentify_dataset
from .extract import SiteFacts, extract_facts
from .photometric import invert_pixels, is_inverted, normalise_photometric
from .pixels import DetectionParams, TextRegion, clean_pixel_data, detect_text_regions
from .profile import DEIDENTIFICATION_METHOD, TAG_ACTIONS, Action, action_for
from .pseudonym import Pseudonymiser, load_or_create_key, normalise_identifier
from .verify import VerificationReport, verify_store

__all__ = [
    "DEIDENTIFICATION_METHOD",
    "TAG_ACTIONS",
    "Action",
    "DeidRecord",
    "DeidResult",
    "DetectionParams",
    "Pseudonymiser",
    "SiteFacts",
    "TextRegion",
    "VerificationReport",
    "action_for",
    "clean_pixel_data",
    "deidentify",
    "deidentify_dataset",
    "detect_text_regions",
    "extract_facts",
    "invert_pixels",
    "is_inverted",
    "normalise_photometric",
    "load_or_create_key",
    "normalise_identifier",
    "verify_store",
]
