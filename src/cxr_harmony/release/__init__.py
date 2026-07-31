"""Immutable, content-addressed dataset releases with leakage-free splits."""

from .builder import (
    ReleaseResult,
    build_manifest,
    build_release,
    digest_manifest,
    render_datasheet,
    verify_release,
)
from .splits import SplitRatios, assign_all, assign_split, realised_proportions

__all__ = [
    "ReleaseResult",
    "SplitRatios",
    "assign_all",
    "assign_split",
    "build_manifest",
    "build_release",
    "digest_manifest",
    "realised_proportions",
    "render_datasheet",
    "verify_release",
]
