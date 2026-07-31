# DPDP Act 2023: controls mapping

**This is an engineering document, not legal advice.** It records which control
in this codebase is intended to support which obligation, so a reviewer can see
the reasoning and disagree with it. Sufficiency is a question for the deploying
institution's Data Protection Officer and ethics committee.

## Scope: this data is pseudonymised, not anonymised

The Act applies to personal data, and genuinely anonymised data falls outside it.
The output of this pipeline is pseudonymised: a keyed HMAC is reversible by
whoever holds the key, so while the key exists the release remains personal data
in the hands of the entity holding it.

That is deliberate — longitudinal linkage and the ability to honour an erasure
request both depend on it — but it means the release must not be treated as out
of scope. Destroying the key is the event that changes the analysis, and it
should be a dated, documented decision rather than a side effect of a machine
being reimaged.

## Mapping

### s.8(4)

> A Data Fiduciary shall implement appropriate technical and organisational measures to give effect to the Act.

**Control.** Identifiers are removed under the DICOM PS3.15 Annex E confidentiality profile, burned-in pixel annotation is detected and destructively redacted, and report prose is scrubbed against the identifiers the header supplied.

**Implemented in.** `cxr_harmony.deid, cxr_harmony.reports.scrub`

### s.8(5)

> A Data Fiduciary shall protect personal data in its possession by taking reasonable security safeguards to prevent a personal data breach.

**Control.** Pseudonyms derive from a keyed HMAC rather than a bare hash, so the mapping is not recoverable by enumerating a small identifier space. The key is held separately from the data and excluded from version control. Role separation confines report text, the highest residual re-identification surface, to the curator role.

**Implemented in.** `cxr_harmony.deid.pseudonym, cxr_harmony.catalog.access`

**Limitation.** Access control here is application-level. It constrains the query helpers, not someone holding the database file; that boundary is infrastructural.

### s.8(7)

> A Data Fiduciary shall erase personal data upon the Data Principal withdrawing consent, unless retention is required by law.

**Control.** Because pseudonyms are deterministic, a withdrawal can be actioned without retaining a name-to-pseudonym table: re-deriving the pseudonym from the identifier the site supplies locates every study for that person across all sites, including ones contributed under a different local MRN.

**Implemented in.** `cxr_harmony.deid.pseudonym.Pseudonymiser.patient_pseudonym`

**Limitation.** Erasure from an already-distributed release is a contractual matter, not a technical one. The release manifest identifies exactly which files a recipient must destroy.

### s.8(3)

> A Data Fiduciary shall ensure the completeness, accuracy and consistency of personal data where it is used to make a decision affecting the Data Principal or is disclosed to another Data Fiduciary.

**Control.** Cross-site harmonisation is validated by QC checks on completeness and consistency; unmapped site values are counted and surfaced rather than silently defaulted, and site-native strings are retained so a mapping can be audited after the fact.

**Implemented in.** `cxr_harmony.qc, cxr_harmony.harmonize`

### s.8(9)

> A Data Fiduciary shall publish the business contact information of a Data Protection Officer or person able to answer questions about processing.

**Control.** Out of scope for this codebase. The deploying institution names the DPO in its own notice.

**Implemented in.** `n/a`

**Limitation.** Not addressed here; recorded so the gap is explicit rather than assumed.

### s.10

> A Significant Data Fiduciary shall appoint a Data Auditor and carry out periodic Data Protection Impact Assessments and audits.

**Control.** A hash-chained, append-only audit log records every stage, its counts and its configuration, and an auditor role can read cohort aggregates without access to patient-level rows.

**Implemented in.** `cxr_harmony.governance.audit, cxr_harmony.catalog.access.Role.AUDITOR`

### s.17(2)(b)

> Provisions of the Act do not apply to processing necessary for research, archiving or statistical purposes where the data is not used to take a decision specific to a Data Principal, subject to prescribed standards.

**Control.** The cohort exists for model development, not for decisions about the individuals in it, and carries no field capable of identifying one. Whether the exemption is available remains a determination for the institution.

**Implemented in.** `cxr_harmony.schema.models`

**Limitation.** The exemption is conditional on standards prescribed by rules under the Act. It is not a substitute for ethics approval or for the sharing agreement.

### s.9

> Processing of a child's personal data requires verifiable parental consent, and tracking or targeted advertising directed at children is prohibited.

**Control.** Paediatric studies are present in a chest radiograph cohort and are not excluded here. Age is retained, so they are identifiable as a stratum and can be filtered or held to a separate consent basis.

**Implemented in.** `cxr_harmony.deid.extract.compute_age`

**Limitation.** Consent capture is upstream of this pipeline. The code cannot verify that parental consent was obtained; it can only make the stratum visible.

## Not addressed by this codebase

- Consent capture, notice, and the consent-manager interface (ss. 5-6).
- Breach notification workflow to the Board and to affected principals (s. 8(6)).
- Grievance redressal and the Consent Manager registration regime.
- Cross-border transfer restrictions (s. 16), which bear directly on any
  collaboration that moves data outside India and must be settled in the
  data-sharing agreement before a release leaves the country.
