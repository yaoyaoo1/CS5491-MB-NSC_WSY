"""Official interface with one import-time, process-fixed model and lambda.

Set MB_NSC_VARIANT to layerwise or global and MB_NSC_LAMBDA to a real in [0,1]
before importing this module. No variant or nonzero lambda is chosen by default.
Changing the environment after import does not change an existing scorer.
This module never reads architecture archives, labels, or experiment results.
"""
import os

from mb_nsc import make_score
from nsc import original_nsc


def _process_score():
    variant = os.environ.get('MB_NSC_VARIANT')
    raw_strength = os.environ.get('MB_NSC_LAMBDA')
    if variant is None and raw_strength is None:
        return None
    if variant is None or raw_strength is None:
        raise ValueError('Set both MB_NSC_VARIANT and MB_NSC_LAMBDA before import')
    try:
        strength = float(raw_strength)
    except ValueError as exc:
        raise ValueError('MB_NSC_LAMBDA must be a finite real in [0,1]') from exc
    return make_score(variant, strength)


_FIXED_SCORE = _process_score()


def score(config):
    """Return the fixed heuristic scalar; larger predicts better quality."""
    if _FIXED_SCORE is None:
        raise ValueError('Set MB_NSC_VARIANT and MB_NSC_LAMBDA before import')
    return _FIXED_SCORE(config)


def ablation(config):
    """Disable only your proposed mechanism; document what changes."""
    return original_nsc(config)
