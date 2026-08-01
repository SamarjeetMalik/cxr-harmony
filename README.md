# cxr-harmony

A reproducible pipeline for ingesting, de-identifying, harmonising and versioning
chest-radiograph datasets contributed by multiple clinical partner sites with
divergent conventions.

[![CI](https://github.com/SamarjeetMalik/cxr-harmony/actions/workflows/ci.yml/badge.svg)](https://github.com/SamarjeetMalik/cxr-harmony/actions/workflows/ci.yml)

> **The demonstration corpus is synthetic; the evaluation is not.** No patient data
> is distributed by this repository. The generator in `src/cxr_harmony/synth/`
> fabricates DICOM carrying realistic identifiers — names, MRNs, national health
> IDs, burned-in pixel annotation — so the de-identification stage has something
> real to remove, which public corpora cannot provide because they are already
> de-identified.
>
> The pipeline is then **evaluated against two real public corpora**: 3,955
> radiologist reports from Open-i / Indiana University, and 400 real hospital DICOM
> objects from UNIFESP. See **[docs/real-data-evaluation.md](docs/real-data-evaluation.md)**.
> Real data found a bug synthetic data structurally could not.

```bash
pip install -e ".[dev]" && make demo
```

Generates a three-site delivery, runs the whole pipeline, and finishes with an
independent verification pass. About 40 seconds.

![Burned-in PHI removal](docs/figures/redaction_before_after.png)

---

## Results

**➜ [Full results, with plots: `docs/RESULTS.md`](docs/RESULTS.md)**

Measured against the numeric performance targets set in the project proposal this
pipeline serves:

![Targets versus achieved](docs/figures/results_targets.png)

| | |
|---|---|
| Identifiers surviving de-identification (synthetic, ground truth known) | **0 of 145** |
| Burned-in text pixels surviving redaction | **0** |
| Cross-site patients correctly collapsed to one identity | **8 of 8** |
| Label extraction on **real** radiologist prose (held-out, n=1,965) | **micro F1 0.901** |
| Same extractor before real data corrected it | micro F1 0.822 |
| Burned-in text found in **real** hospital images | **71 of 400**, zero-shot across language |
| Real archive mixing greyscale conventions, silently | **372 MONOCHROME1 : 28 MONOCHROME2** |
| Cohen's κ vs radiologist annotation (held-out) | **0.897** — target >0.80 |
| Normal-study detection (strict / vocabulary-adjusted) | **0.854 / 0.932** — was 0.452 / 0.563 |
| Throughput, real archive, end to end | **88,742/hour** median of 5 runs, **60,531** worst case — target >500 |
| Tests | **383** |

---

## What is in here

A reviewer of an earlier revision reported four of these as missing. They were
present and shipped; nothing in this README pointed at them, and work that cannot
be found does not count. So it is now written down.

| | Where | What it does |
|---|---|---|
| **Pipeline versioning** | [`dvc.yaml`](dvc.yaml), [`params.yaml`](params.yaml) | Six stages with declared dependencies and outputs, so a changed parameter reruns exactly the stages it invalidates. `dvc repro` reproduces the results JSONs the docs are read from. |
| **Stratified splitting** | [`release/splits.py`](src/cxr_harmony/release/splits.py) | Patient-level hash-threshold splits with proportional allocation across site × sex × age strata, so a small stratum is not lost to chance. Stable under cohort growth: adding patients does not move existing ones. |
| **Body-part gate** | [`ingest/scanner.py`](src/cxr_harmony/ingest/scanner.py), [`qc/`](src/cxr_harmony/qc/) | `BodyPartExamined` was empty on **all 400** real objects surveyed, so the ingest filter cannot do its job. Studies are flagged `BODY_PART_UNVERIFIED` and a release-blocking QC check refuses to ship a cohort of them. The gate exists; the classifier does not, and [RESULTS.md](docs/RESULTS.md) says why. |
| **Interoperability** | [`interop/fhir.py`](src/cxr_harmony/interop/fhir.py) | FHIR R4 `ImagingStudy`, `DiagnosticReport` and `Observation` resources with SNOMED CT coding, so the cohort is addressable by a hospital system rather than only by this repo. |
| **Key handling** | [`deid/pseudonym.py`](src/cxr_harmony/deid/pseudonym.py) | `load_or_create_key` **raises** rather than silently creating a key, and warns when it does create one. `Pseudonymiser.from_env()` reads `CXR_HARMONY_KEY` so a mounted secret never touches disk. |
| **Security audit** | [`Makefile`](Makefile) | `make audit` runs dependency and static checks as a target, not as a habit someone has to remember. |
| **Deployment gap** | [`docs/deployment.md`](docs/deployment.md) | What separates this demo configuration from a deployment — PostgreSQL with row-level security, object storage, key custody — written out so the gap is explicit rather than assumed closed. |
| **Number traceability** | [`tests/test_docs_consistency.py`](tests/test_docs_consistency.py) | Every headline in this README and in RESULTS.md is pinned to the JSON key it came from, and superseded figures are asserted absent. Added after a stale throughput number survived three revisions of this file. |

---

## The problem

Three hospitals contribute chest radiographs to a shared cohort. They record the
same clinical facts, and no two record them the same way:

| | Site A | Site B | Site C |
|---|---|---|---|
| Projection | `ViewPosition` = `PA` | empty tag; buried in free-text `SeriesDescription` | legacy beam arrow `P->A` |
| Study date | DICOM `DA` | `DD-MM-YYYY` in a private block | DICOM `DA` |
| Sex | `M` / `F` | `MALE` / `FEMALE` | HL7 numerics `1` / `2` |
| Labels | `ImageComments` | sidecar CSV, abbreviated (`CM`, `PE`) | report prose only |
| Burned-in text | none | 85% of images | 35% of images |

Every one of those is a convention real contributed archives exhibit. The HL7
numeric sex codes are the quietest hazard: `1` and `2` are valid strings that
nothing rejects, so an unconfigured reader produces a cohort with no usable sex
variable and no error.

## The pipeline

```mermaid
flowchart LR
    A[Incoming delivery] --> B[ingest]
    B -->|quarantine + reasons| Q[(quarantine.jsonl)]
    B --> C[deid]
    C -->|clinical facts read<br/>before redaction| F[(site_facts)]
    C --> D[reports]
    F --> E[harmonize]
    D --> E
    E --> G[catalog]
    G --> H[qc]
    G --> I[release]
    I --> J[verify]
    H --> J
```

| Stage | What it does |
|---|---|
| `ingest` | Discovers, validates and indexes objects. Rejections go to a quarantine file **with a reason code**, never silently dropped. Copies no pixels. |
| `deid` | DICOM PS3.15 Annex E profile, keyed pseudonymisation, per-patient date shifting, UID remapping, burned-in text redaction. |
| `reports` | Sections report prose, scrubs PHI against the identifiers the header supplied, extracts labels with negation handling. |
| `harmonize` | Maps each site's conventions onto the canonical schema via a YAML adapter. Unmapped values are counted, not defaulted away. |
| `catalog` | Loads into SQLite with foreign keys enforced and three access roles. |
| `qc` | Integrity, completeness and cross-site distribution checks. |
| `release` | Content-addressed, immutable release with patient-level splits and a datasheet. |
| `verify` | Independent check of de-identification, the audit chain, and every release. Exits non-zero on failure. |

## Design decisions worth arguing with

**Clinical values are read before the profile destroys them.** The confidentiality
profile removes `SeriesDescription` and `ImageComments` because both routinely
carry names and dates. But site B encodes the projection in the first and site A
ships its labels in the second — so applying the profile first destroys the
content the cohort exists to hold. Extraction is therefore a separate step that
runs ahead of redaction, and its output carries no direct identifiers.

**Pseudonyms are keyed HMACs, not hashes.** The space of Indian MRNs, and even of
14-digit ABHA numbers, is small enough to enumerate. An unkeyed `sha256(mrn)` is
reversible by anyone willing to spend an afternoon, and several published
"anonymised" datasets have fallen exactly that way. Destroying the key makes the
mapping irreversible by construction — a property a data-sharing agreement can
actually require.

**Cross-site linkage runs on the national ID, and only on it.** Two hospitals
cannot be linked through local MRNs; those are independent sequences. Where both
record an ABHA, that is the only field on which one person's two records can be
recognised as one. Where it is absent, the pseudonym stays site-scoped and the
records stay separate — the conservative outcome, reported in QC rather than
guessed at.

![Detector separation](docs/figures/detection_separation.png)

**Burned-in text is found by edge density, not brightness.** On a chest
radiograph the spine, mediastinum and subdiaphragmatic region saturate to the
same value the text is drawn at, so a threshold finds the anatomy and misses the
name. Measured over 60 images, merged text lines score 0.73–1.00 edge density
against 0.16–0.38 for non-text components — an empty gap of 0.36 to put a
threshold in. Redaction zeroes the region; blurring and pixelation are both
invertible often enough not to be relied on.

It transfers: tuned on synthetic English overlays, it found real Portuguese
annotation on real Brazilian hospital films (`DECUBITO LATERAL DIREITO`,
`OBLIQUA ESQUERDA`) with no retuning, because it keys on the spatial frequency of
glyph strokes rather than on their meaning.

![Split stability](docs/figures/split_stability.png)

**Splits are per patient, and assigned by hash threshold.** A patient with a
baseline and two follow-ups contributes three studies; split them independently
and the same chest — same habitus, same old fracture, same implanted device —
sits in training and in test, inflating the score with nothing in a random-split
evaluation to reveal it. Assignment is by hash rather than shuffle because a
shuffle reassigns everyone when the cohort grows, so patients trained on last
quarter land in this quarter's test set. A threshold depends on the patient
alone.

**The verifier does not import the profile.** It checks output against the
requirement rather than against the implementation. A verifier sharing the
engine's notion of which tags matter will agree with it about a tag they both
forgot.

## What a run looks like

From `make demo` (48 patients, 74 studies, 512×512, seed 20260731):

```
Generated 74 studies for 48 patients across 3 sites
  8 patients were imaged at more than one site
Accepted 74 objects
De-identified 74 objects
  29 had burned-in annotation redacted
  8 patients linked across sites
Processed 74 reports
  1075 redactions applied
Harmonised 74 studies
    SITE_A: 29   SITE_B: 18   SITE_C: 27
All checks passed
Release v1.0.0  digest 09eb0d7e8963496b...
+----------------------------+
| Split | Patients | Studies |
| train |       27 |      40 |
| val   |       10 |      16 |
| test  |       11 |      18 |
+----------------------------+
De-identification: 74 objects checked
  no violations
  searched for 145 known identifier strings
Audit chain: intact
Release v1.0.0: verified
Verification passed
```

The linkage count matching the planted ground truth (8 of 8) is the load-bearing
number: it means the same person imaged at two hospitals under two different MRNs
collapsed to one identity, which is what makes the leakage-free split meaningful
across sites rather than only within one.

## Verification

Correctness claims here are tested, not asserted. 383 tests, and the ones that
matter most break something first — a QC check that has never been observed to
fail is not evidence of anything.

- **No identifier survives.** Every output header is re-read and searched for the
  exact identifier strings that went in. This is only possible because the corpus
  is synthetic and its ground truth is known; against real data you can verify
  structure but not that a particular name is gone, because you are not permitted
  to keep the list of names to search for. That is the argument for validating a
  de-identifier on generated data before pointing it at patients.
- **No pixel of burned-in text survives.** Asserted at pixel level, not by
  counting detections.
- **No patient appears in two splits.**
- **The same seed reproduces the corpus and the release byte for byte.**
- **The verifier can fail** — tampering tests reintroduce an identifier, break the
  audit chain, and corrupt a release manifest, and each is caught.

Coverage on `deid`, the module where a defect is a disclosure, is 94%.

## Governance

`docs/dpdp-controls.md` maps controls to obligations under India's DPDP Act 2023.
It is an engineering document, not legal advice, and it says so.

It also says plainly that **this output is pseudonymised, not anonymised**. A
keyed HMAC is reversible by whoever holds the key, so while the key exists the
release remains personal data and cannot be treated as outside the Act. Claiming
otherwise is the common and expensive error. Gaps the codebase does not address —
consent capture, breach notification, grievance redressal, and cross-border
transfer under s.16 — are listed rather than omitted.

The audit log is hash-chained. That is tamper-*evident*, not tamper-proof: anyone
who can write the file can rewrite the chain. It defends against the realistic
failure, which is one entry quietly edited months later.

![Harmonisation](docs/figures/harmonisation.png)

## Adding a fourth site

Write `configs/sites/site_d.yaml` and run `harmonize`. No code change:

```yaml
site_id: SITE_D
view_position:
  patterns:
    - match: '\bAP\b'
      value: AP
    - match: '\bPA\b'
      value: PA
  default: UNKNOWN
sex:
  map: {"1": M, "2": F}
labels:
  source: sidecar_csv
  sidecar_file: findings.csv
  sidecar_key_column: accession
  sidecar_value_column: codes
  map: {NAD: NO_FINDING, CM: CARDIOMEGALY}
date_formats: ["%d/%m/%Y"]
```

Anything the config does not account for is counted in `unmapped_values.json` and
surfaced in the QC report, so a site that starts sending a new code becomes a
question to ask them rather than silent attrition.

## Limitations

Stated because they bear on whether any of this is usable, not for form's sake.

- **The label extractor scores 0.901 on real prose, not 1.000.** The synthetic
  score is 1.000 because the phrase bank and the report generator share an author.
  Measured against radiologist MeSH annotation on a held-out half of Open-i, it is
  micro F1 0.901. Real prose exposed four outright defects, all now
  regression-tested; see [docs/real-data-evaluation.md](docs/real-data-evaluation.md).
- **The nine-finding vocabulary is not sufficient for real annotation.** One corpus
  contained 244 distinct MeSH terms outside it. Those are now detected as `OTHER`
  rather than silently labelled "no finding", which took normal-study detection
  from 0.452 to 0.854 strict — but `OTHER` is a holding pen, not a trainable label.
- **Burned-in false positives are not fixed.** Two filters were implemented,
  measured, and rejected: one left PHI surviving on 53 of 80 images, the other
  could not be validated against any ground truth. Over-redaction remains, in the
  safe direction, and is reported. See [docs/RESULTS.md](docs/RESULTS.md).
- **Open-i is two Indiana hospital systems.** House style varies, and Indian
  partner-site prose will differ from it. A new archive would need re-scoring.
- **Burned-in text detection transferred, but is not clean.** It found real
  Portuguese annotation on real Brazilian films having been tuned on synthetic
  English, which is genuine zero-shot transfer. It also placed occasional small
  spurious boxes over high-contrast spine anatomy. Over-redaction is the safe error
  direction for a PHI tool, but the false-positive rate on real anatomy is not zero.
- **Anatomical scope can be unverifiable.** `BodyPartExamined` was empty on all 400
  real objects surveyed, so the ingest filter could not do its job and non-chest
  studies entered the cohort. QC now warns rather than pretending otherwise, but a
  production chest cohort needs a body-part classifier as a fallback. Not built here.
- **Access control is application-level.** It constrains the query helpers, not
  someone holding the SQLite file. That boundary is infrastructural.
- **No OCR fallback.** Text detection finds *where* characters are, not what they
  say, which is sufficient for redaction but means there is no audit trail of what
  was removed.
- **The synthetic images are not anatomically faithful.** Nothing here reads them
  diagnostically; they exist to make text detection a genuine problem rather than
  a trivial one.
- **Sequence recursion is shallow-tested.** The profile recurses into nested
  sequences, but the synthetic corpus contains few, so that path has less coverage
  than the top-level one.

## Layout

```
src/cxr_harmony/
  schema/       canonical models, vocabularies, JSON Schema export
  synth/        synthetic three-site corpus generator
  ingest/       discovery, validation, quarantine
  deid/         PS3.15 profile, pseudonymisation, pixel cleaning, verification
  reports/      sectioning, PHI scrubbing, rule-based labels
  harmonize/    YAML site adapters onto the canonical schema
  catalog/      SQLite catalogue, role-based access
  qc/           checks and reporting
  release/      content-addressed releases, leakage-free splits
  governance/   hash-chained audit log, DPDP mapping, verification
  adapters/     readers for real public corpora (Open-i)
  interop/      FHIR R4 export
configs/sites/  one YAML per contributing site
scripts/        figure generation, real-data fetch and evaluation
docs/           governance, real-data evaluation, figures, results
```

Figures are regenerated from the pipeline itself with `make figures`, so a claim
in this README and the picture illustrating it cannot drift apart.

## Licence

MIT.
