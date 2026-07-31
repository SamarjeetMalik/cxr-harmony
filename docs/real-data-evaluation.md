# Evaluation on real data

The synthetic corpus proves the pipeline runs. It cannot prove the pipeline
*works*, because its reports were written by the same hand as the phrase bank and
its images by the same hand as the detector. This document records what happened
when the same code met data nobody here wrote.

Neither corpus is redistributed. Fetch them with:

```bash
python scripts/fetch_real_data.py --openi --unifesp
```

| Corpus | What it is | Licence |
|---|---|---|
| [Open-i / Indiana University](https://openi.nlm.nih.gov/faq) | 3,955 real radiology reports, sectioned, with manually assigned MeSH terms | CC BY-NC-ND 4.0 |
| [UNIFESP X-ray](https://www.kaggle.com/datasets/felipekitamura/unifesp-xray-bodypart-classification) | Real hospital computed-radiography DICOM (Universidade Federal de São Paulo) | CC BY-NC-SA 4.0 |

Both were de-identified by their publishers before release. **Neither can
demonstrate PHI-removal efficacy** — there is nothing left to remove. Evidence for
removal comes from the synthetic corpus, where ground truth exists. What real data
tests is everything else: whether the parser, the negation scope, the label
extractor and the pixel detector survive contact with reality.

---

## 1. Label extraction against radiologist annotation

Scored against Open-i's MeSH terms. The phrase bank was refined by reading
failures on this corpus, so the corpus is split by record id and **the held-out
half is the number that should be quoted**.

| | micro P | micro R | micro F1 | macro F1 | exact match |
|---|---:|---:|---:|---:|---:|
| Synthetic corpus | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Real, before fixes | 0.850 | 0.806 | 0.827 | 0.722 | 0.900 |
| Real, **held-out** | 0.891 | 0.912 | **0.901** | 0.881 | 0.936 |
| Real, dev half | 0.894 | 0.913 | 0.903 | 0.864 | 0.932 |

Dev and held-out agree to within 0.002 F1, so the corrections generalised rather
than fitting the tuning half.

Per finding, held-out (n=1,965 reports):

| Finding | P | R | F1 | support |
|---|---:|---:|---:|---:|
| Atelectasis | 0.981 | 0.962 | 0.971 | 210 |
| Cardiomegaly | 0.861 | 0.856 | 0.859 | 181 |
| Pleural effusion | 0.925 | 0.935 | 0.930 | 92 |
| Pulmonary oedema | 0.803 | 0.860 | 0.831 | 57 |
| Nodule | 0.814 | 0.889 | 0.850 | 54 |
| Consolidation | 0.818 | 0.918 | 0.865 | 49 |
| Fracture | 0.875 | 0.946 | 0.909 | 37 |
| Pneumothorax | 0.800 | 0.923 | 0.857 | 13 |
| Tuberculosis | 0.750 | 1.000 | 0.857 | 3 |

Normal-study detection F1 is **0.674**, the weakest result here and the honest
headline caveat: roughly a third of studies a radiologist called normal are not
recognised as such.

### What real prose taught the extractor

Four defects, none of which the synthetic corpus could have exposed.

**"Clear of" was not a negation.** The sentence *"the lungs are clear of focal
airspace disease, pneumothorax, or pleural effusion"* is boilerplate in real
normal reports, and without that cue it asserts all three findings. It alone
accounted for most false-positive pneumothorax calls — precision 0.256 before,
0.800 after.

**Resolution statements read as assertions.** *"The left apical pneumothorax has
resolved"* places the cue after the finding, so a prefix-only negation scope calls
it positive. Negation now also scans for resolution cues across the sentence.

**Radiologists do not write the label.** The phrase bank expected *"the cardiac
silhouette is enlarged"*; real reports overwhelmingly say *"the heart is mildly
enlarged"*. Similarly, Open-i indexes *Pulmonary Congestion* as oedema, but reports
express it as *"vascular congestion"* or *"vascular prominence"* — oedema recall
was 0.339 before those were added, 0.860 after.

**Descriptor is not diagnosis.** Adding *"airspace disease"*, *"infiltrate"* and
*"patchy opacity"* to consolidation pushed recall to 0.980 and collapsed precision
to 0.306 — 220 false positives. Radiologist annotators index those under a separate
*Opacity* heading, and they are right to: an opacity is a description of increased
density, consolidation is a diagnosis about its cause. Restricting to consolidation
proper gives 0.818/0.918.

That last correction propagated backwards. The synthetic report generator had been
emitting *"patchy air-space opacity"* as a consolidation sentence, teaching a
conflation real annotation does not make, so the generator was corrected too.

**Normal-cue coverage.** The original cues required *"lung fields are clear"*,
which real reports essentially never write — they say *"Lungs are clear"* or close
with *"No acute cardiopulmonary disease"*. Normal-study F1 rose from 0.437 to 0.674.

Every one of these is now a regression test in `tests/test_reports.py`.

---

## 2. Real DICOM

400 objects from the UNIFESP export, run through ingest, de-identification and
verification.

```
SURVEY AS RECEIVED
  Modality                   CR=400
  PhotometricInterpretation  MONOCHROME1=372, MONOCHROME2=28
  BodyPartExamined           <empty>=400
  ViewPosition               <empty>=400
  PatientSex                 <empty>=400
  BitsStored                 15=372, 10=22, 8=6
  Manufacturer               <empty>=400

PIPELINE
  ingest    accepted=400 quarantined=0
  deid      objects=400 pixel_redacted=71 photometric_converted=372
            photometric after: {'MONOCHROME2': 400}
  verify    checked=400 passed=True {}
```

### The finding that mattered: mixed greyscale polarity

**372 of 400 objects were MONOCHROME1 and 28 were MONOCHROME2 — in one archive,
with nothing to warn you.** Under MONOCHROME1 the minimum stored value renders
*white*: the image is photometrically inverted. Pass that cohort to a model
unnormalised and about 7% of it arrives as a photographic negative of the rest.
Nothing errors, thumbnails look plausible, and the model spends capacity learning
that two visually opposite things mean the same.

It also broke redaction. Blacking out a region means writing the value that
*renders* black, which is zero only under MONOCHROME2; on the other 93% a
redaction box would have been painted bright white — still concealing the text, but
introducing a maximal-intensity artefact into every subsequent intensity
normalisation.

`cxr_harmony.deid.photometric` now converts to MONOCHROME2 before redaction, and
a QC check fails a cohort that mixes conventions. This is the single clearest
argument in the project for testing on real archives: the synthetic generator
emitted MONOCHROME2 throughout, because that is what one writes when one is
writing both ends.

### Burned-in text detection transferred

71 of 400 real images carried burned-in annotation, and the detector found it —
having been tuned entirely on synthetic English text. What it found was Portuguese:
`GRANDE ENCHIMENTO`, `OBLIQUA DIREITA`, `OBLIQUA ESQUERDA`,
`DECUBITO LATERAL DIREITO`. The morphological approach keys on the spatial
frequency of glyph strokes, not on their meaning, so an unseen language and unseen
fonts made no difference.

Redacted area was small — median 0.14% of the image, maximum 2.75% — consistent
with annotation stripes rather than eaten anatomy.

It is not clean, though. On inspection at least one image carried two small
spurious boxes over high-contrast spine anatomy. For a PHI tool that error
direction is the safe one, but it is a real false-positive rate on real anatomy
and it is not zero.

### The limitation real data exposed: unverifiable scope

`BodyPartExamined` was **empty on all 400 objects**, and ingest tolerates an empty
value — a populated mismatch is disqualifying, an absent one merely uninformative.
On this archive that tolerance had teeth: the collection is a body-part dataset,
and abdominal and pelvic studies were admitted into what a chest pipeline would
treat as a chest cohort.

No header check can fix this, because the information simply is not there. The
honest response is to make it visible rather than to pretend the filter worked, so
QC now warns when studies carry no body-part attestation and states plainly that
anatomical scope is unverified. A production chest cohort would need a body-part
classifier as a fallback, which is out of scope here and named as future work
rather than quietly assumed.

---

## What this does and does not establish

**Established.** The report parser, negation scope and label extractor work on real
radiologist prose at micro F1 0.901 held-out. The pixel detector transfers
zero-shot across language and font. The pipeline ingests, converts and verifies a
real hospital archive without hand-holding.

**Not established.** That de-identification removes real PHI — no public corpus can
show this, and the synthetic evidence is what stands. That the label extractor
would hold up on a differently-styled archive; Open-i is two Indiana hospital
systems, and Indian partner-site prose will differ. That normal-study detection at
F1 0.674 is good enough for anything; it is not, and it is the first thing to fix.

**Reproduce.**

```bash
python scripts/fetch_real_data.py --openi --unifesp
python scripts/evaluate_openi.py --corpus realdata/ecgen-radiology --fold heldout
python scripts/run_real_dicom.py --src realdata/unifesp/images --work work-real
```
