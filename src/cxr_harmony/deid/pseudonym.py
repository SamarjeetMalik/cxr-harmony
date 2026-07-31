"""Keyed pseudonymisation, date shifting and UID remapping.

Every derived value here is a keyed HMAC rather than a plain hash. The
distinction is not pedantry: the space of Indian MRNs, and even of 14-digit
national health identifiers, is small enough to enumerate. An unkeyed
``sha256(mrn)`` is therefore reversible by anyone willing to spend an afternoon
on it, and several published "anonymised" datasets have fallen exactly that way.
With a secret key, re-identification requires the key — and destroying the key
makes the mapping irreversible by construction, which is a thing a data-sharing
agreement can actually require of you.

The linkage rule is the interesting part. Two hospitals cannot be linked through
their local MRNs, which are independent sequences. Where both record a national
health identifier (ABHA), that is the only field on which the same person's two
records can be recognised as one, so it takes precedence. Where it is absent the
pseudonym falls back to being site-scoped, and the two records stay separate —
which is the correct, conservative outcome, and one that QC reports rather than
hides.
"""

from __future__ import annotations

import hmac
import os
import re
import secrets
import warnings
from datetime import date, datetime, timedelta
from hashlib import sha256
from pathlib import Path

#: Derived UIDs are rooted at 2.25, the ISO/IEC 9834-8 arc reserved for values
#: constructed from a UUID-sized integer. It needs no registration, which means a
#: deployment does not have to obtain an organisational root before running this.
DEID_UID_ROOT = "2.25"

#: Bits of HMAC output used per derived UID. 112 keeps the result comfortably
#: inside the 64-character UID limit while leaving collisions implausible.
_UID_BITS = 112

#: Date offsets are drawn from this many days back, per patient.
_MAX_SHIFT_DAYS = 1095

_NON_ALNUM = re.compile(r"[^A-Za-z0-9]")

KEY_BYTES = 32


#: Environment variable holding a hex-encoded key, for Docker or Kubernetes
#: secrets. Preferred over a file: a secret mounted as an environment variable
#: leaves no artefact on disk to be backed up, snapshotted or copied into an image.
KEY_ENV_VAR = "CXR_HARMONY_KEY"


def load_or_create_key(path: Path, *, allow_create: bool = False) -> bytes:
    """Return the pseudonymisation key at ``path``.

    Creating a key is **opt-in**. Silently generating one on first run is how a
    production deployment ends up with its re-identification secret sitting in a
    working directory, inside whatever backs that directory up — and because the
    pipeline would run perfectly, nobody would find out until an audit. An
    explicit ``allow_create`` means the demo works out of the box while a
    deployment has to say so on purpose.

    Creation also warns. The key is the entire basis of the claim that pseudonyms
    are not reversible, so a file-backed one is a fact the operator should be told
    about rather than left to infer from the source.

    In deployment this belongs in a KMS or HSM, or in :data:`KEY_ENV_VAR` via
    :meth:`Pseudonymiser.from_env`.
    """
    path = Path(path)
    if path.exists():
        key = path.read_bytes()
        if len(key) < 16:
            raise ValueError(f"pseudonymisation key at {path} is too short to be usable")
        return key

    if not allow_create:
        raise FileNotFoundError(
            f"no pseudonymisation key at {path}. Set {KEY_ENV_VAR} to a "
            f"KMS-backed secret, or pass allow_create=True to generate a local "
            f"key for development."
        )

    key = secrets.token_bytes(KEY_BYTES)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(key)
    try:
        os.chmod(path, 0o600)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pass

    warnings.warn(
        f"Created a pseudonymisation key on the local filesystem at {path}. "
        "This is adequate for development and not for production: anything that "
        "backs up or snapshots this directory now holds the key that reverses "
        f"every pseudonym in the cohort. Use a KMS, an HSM, or {KEY_ENV_VAR}.",
        RuntimeWarning,
        stacklevel=2,
    )
    return key


def normalise_identifier(value: str) -> str:
    """Strip formatting so ``99-1234-5678-9012`` and ``99123456789012`` agree.

    Sites format the same national identifier differently, and a linkage that
    depends on punctuation is not a linkage.
    """
    return _NON_ALNUM.sub("", value or "").upper()


class Pseudonymiser:
    """Derives stable pseudonyms, date offsets and UIDs from a secret key."""

    def __init__(self, key: bytes) -> None:
        if len(key) < 16:
            raise ValueError("key must be at least 16 bytes")
        self._key = key

    @classmethod
    def from_env(cls, var: str = KEY_ENV_VAR) -> Pseudonymiser:
        """Build from a hex-encoded key in the environment.

        The deployment path: a secret mounted by Docker or Kubernetes never
        touches the filesystem, so it cannot be picked up by a backup, baked into
        an image layer, or left behind on a decommissioned volume.
        """
        raw = os.environ.get(var, "").strip()
        if not raw:
            raise KeyError(f"{var} is not set")
        try:
            key = bytes.fromhex(raw)
        except ValueError as exc:
            raise ValueError(f"{var} must be hex-encoded") from exc
        return cls(key)

    # --- primitives ----------------------------------------------------
    def _mac(self, domain: str, value: str) -> bytes:
        """Domain-separated HMAC, so a pseudonym can never collide with a UID."""
        return hmac.new(self._key, f"{domain}\x00{value}".encode(), sha256).digest()

    # --- patient identity ----------------------------------------------
    def patient_pseudonym(
        self,
        *,
        national_id: str | None,
        site_id: str,
        local_mrn: str,
    ) -> tuple[str, bool]:
        """Return ``(pseudo_id, linked_across_sites)``.

        When a national identifier is present the pseudonym is derived from it
        alone, so the same person collapses to one identity across hospitals.
        Otherwise it is scoped to ``site_id`` and cannot, and must not, match
        anything from another site.
        """
        normalised = normalise_identifier(national_id or "")
        if normalised:
            return self._mac("patient-national", normalised).hex()[:16], True
        scoped = f"{site_id}|{normalise_identifier(local_mrn)}"
        return self._mac("patient-local", scoped).hex()[:16], False

    # --- temporal ------------------------------------------------------
    def date_offset_days(self, pseudo_id: str) -> int:
        """A stable negative offset for one patient.

        The offset is per-patient, not per-study. That is the whole point of the
        Modified Dates option: shift every date belonging to a patient by the
        same amount and the interval between their baseline and follow-up film
        survives intact, while the calendar date does not.
        """
        raw = int.from_bytes(self._mac("date-offset", pseudo_id)[:4], "big")
        return -(raw % _MAX_SHIFT_DAYS) - 1

    def shift_date(self, value: date, pseudo_id: str) -> date:
        return value + timedelta(days=self.date_offset_days(pseudo_id))

    def shift_da(self, value: str, pseudo_id: str) -> str:
        """Shift a DICOM ``DA`` string, returning ``""`` if it is unparseable."""
        text = (value or "").strip()
        if len(text) != 8 or not text.isdigit():
            return ""
        try:
            parsed = datetime.strptime(text, "%Y%m%d").date()
        except ValueError:
            return ""
        return self.shift_date(parsed, pseudo_id).strftime("%Y%m%d")

    # --- UIDs ----------------------------------------------------------
    def remap_uid(self, original: str) -> str:
        """Deterministically map a UID into the 2.25 arc.

        Deterministic rather than random so that a re-run reproduces the same
        catalogue, and so that the same study referenced from two objects maps to
        one value without a lookup table having to be consulted.
        """
        digest = self._mac("uid", original or "")
        value = int.from_bytes(digest, "big") >> (256 - _UID_BITS)
        uid = f"{DEID_UID_ROOT}.{value}"
        assert len(uid) <= 64, "derived UID exceeded the DICOM length limit"
        return uid


__all__ = [
    "DEID_UID_ROOT",
    "KEY_BYTES",
    "KEY_ENV_VAR",
    "Pseudonymiser",
    "load_or_create_key",
    "normalise_identifier",
]
