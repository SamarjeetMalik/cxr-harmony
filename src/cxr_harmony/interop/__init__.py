"""Interoperability exports.

The catalogue is this pipeline's own shape; FHIR is the shape the rest of a
hospital speaks. A cohort that cannot be handed over in a standard form lives
permanently inside one team's tooling.

Scope is deliberately narrow: FHIR R4 resource construction as JSON. DICOM-SR
parsing and an HL7v2 listener are named in the project plan and are not
implemented here — a listener is a server, and shipping half of one would be
worse than shipping none.
"""

from .fhir import BundleStats, build_bundle, write_bundle

__all__ = ["BundleStats", "build_bundle", "write_bundle"]
