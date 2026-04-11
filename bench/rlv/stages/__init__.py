"""RLV stages — see docs/phase3_rlv_challenge.md §3.1 for the full architecture.

Stage layering (each stage depends on the previous):
    gist     → produces a structured outline (Stage 1)
    locator  → outline + question → region pointer (Stage 2)
    lookup   → region + question → tentative answer (Stage 3)
    verifier → gist + answer → verdict (Stage 4)
    researcher → retry locator with a different region (Stage 5)
"""
from . import gist
from . import locator
from . import lookup
from . import verifier
from . import researcher

__all__ = ["gist", "locator", "lookup", "verifier", "researcher"]
