"""Shared fixtures.

The corpus is generated once per test session at a small image size. Stages that
need a clean workspace get their own, but they all read the same delivery, which
keeps the suite fast enough to run on every commit.
"""

from __future__ import annotations

import pytest

from cxr_harmony.synth import generate_corpus
from cxr_harmony.workspace import Workspace

SEED = 20260731


@pytest.fixture(scope="session")
def delivery(tmp_path_factory):
    """A generated three-site delivery, shared across the session."""
    out = tmp_path_factory.mktemp("delivery")
    truth = generate_corpus(out, seed=SEED, n_patients=20, n_cross_site=5, image_size=128)
    return out, truth


@pytest.fixture
def workspace(tmp_path) -> Workspace:
    return Workspace(tmp_path / "work").ensure()
