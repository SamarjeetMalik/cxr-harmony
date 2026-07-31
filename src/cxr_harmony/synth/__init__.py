"""Synthetic multi-site corpus generation.

Used to produce the demonstration delivery. Nothing in this subpackage is
imported by the pipeline stages themselves — the pipeline neither knows nor cares
that its input was generated here.
"""

from .generator import generate_corpus
from .pixels import BITS_STORED, IMAGE_SIZE, burn_in_text, synthesise_radiograph
from .sites import SITES, SITES_BY_ID, SiteProfile

__all__ = [
    "BITS_STORED",
    "IMAGE_SIZE",
    "SITES",
    "SITES_BY_ID",
    "SiteProfile",
    "burn_in_text",
    "generate_corpus",
    "synthesise_radiograph",
]
