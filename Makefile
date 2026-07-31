.PHONY: install dev test lint demo clean docs figures realdata evaluate benchmark results

WORK ?= work
SEED ?= 20260731
VERSION ?= v1.0.0

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

docs:
	cxr-harmony schema --out docs/schema
	cxr-harmony docs --out docs

figures:
	python scripts/make_figures.py
	python scripts/make_results_figures.py

benchmark:
	python scripts/benchmark.py --synthetic --real realdata/unifesp/images

# Everything behind docs/RESULTS.md, from corpus fetch to redrawn figures.
results: realdata evaluate benchmark figures

# Real public corpora, fetched on demand. Neither is redistributed here:
# Open-i is CC BY-NC-ND and UNIFESP is CC BY-NC-SA, whose share-alike term
# would conflict with this project's MIT licence.
realdata:
	python scripts/fetch_real_data.py --openi --unifesp

evaluate: realdata
	python scripts/evaluate_openi.py --corpus realdata/ecgen-radiology --fold heldout 		--json-out docs/results/openi_heldout.json
	python scripts/evaluate_openi.py --corpus realdata/ecgen-radiology --fold dev 		--json-out docs/results/openi_dev.json
	python scripts/evaluate_openi.py --corpus realdata/ecgen-radiology --fold heldout --baseline 		--json-out docs/results/openi_heldout_baseline.json
	python scripts/run_real_dicom.py --src realdata/unifesp/images --work work-real

# End-to-end demonstration: synthesise a three-site delivery, then run the full
# pipeline over it, cut a release, and verify the result independently.
demo: clean
	cxr-harmony synth --out $(WORK)/incoming --seed $(SEED) --image-size 512
	cxr-harmony ingest --src $(WORK)/incoming --work $(WORK)
	cxr-harmony deid --work $(WORK)
	cxr-harmony reports --work $(WORK)
	cxr-harmony harmonize --work $(WORK) --configs configs/sites
	cxr-harmony catalog --work $(WORK)
	cxr-harmony qc --work $(WORK)
	cxr-harmony release --work $(WORK) --version $(VERSION)
	cxr-harmony verify --work $(WORK)

clean:
	rm -rf $(WORK)
