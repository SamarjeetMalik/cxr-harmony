.PHONY: install dev test lint demo clean

WORK ?= work

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

# End-to-end demonstration: synthesise a three-site corpus, then run the full
# pipeline over it and emit a versioned release.
demo: clean
	cxr-harmony synth --out $(WORK)/incoming --seed 20260731
	cxr-harmony ingest --src $(WORK)/incoming --work $(WORK)
	cxr-harmony deid --work $(WORK)
	cxr-harmony reports --work $(WORK)
	cxr-harmony harmonize --work $(WORK) --configs configs/sites
	cxr-harmony catalog --work $(WORK)
	cxr-harmony qc --work $(WORK)
	cxr-harmony release --work $(WORK) --version v1.0.0
	cxr-harmony verify --work $(WORK)

clean:
	rm -rf $(WORK)
