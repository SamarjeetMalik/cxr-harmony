.PHONY: install dev test lint demo clean docs

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
