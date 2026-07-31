"""Immutable, content-addressed dataset releases with leakage-free splits."""

from .builder import (
    ReleaseResult,
    build_manifest,
    build_release,
    digest_manifest,
    render_datasheet,
    verify_release,
)
from .splits import (
    SplitRatios,
    assign_all,
    assign_split,
    assign_stratified,
    realised_proportions,
    strata_from_dataset,
)

__all__ = [
    "ReleaseResult",
    "SplitRatios",
    "assign_all",
    "assign_split",
    "assign_stratified",
    "build_manifest",
    "build_release",
    "digest_manifest",
    "realised_proportions",
    "strata_from_dataset",
    "render_datasheet",
    "verify_release",
]
