"""Readers for real public corpora used in evaluation.

Separate from :mod:`cxr_harmony.synth` because these read data this project did
not write. That is the whole point of them: a pipeline validated only against a
corpus its own author generated has been tested for self-consistency, not for
correctness.

The corpora themselves are not redistributed; see ``scripts/fetch_real_data.py``.
"""

from .openi import MESH_TO_FINDING, OpeniReport, load_corpus, parse_report

__all__ = ["MESH_TO_FINDING", "OpeniReport", "load_corpus", "parse_report"]
