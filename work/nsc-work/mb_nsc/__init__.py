"""Specification-only quadratic MB-NSC; no data or label access."""

from .common import LAMBDA_GRID, validate_lambda
from .global_model import score as global_score
from .layerwise import score as layerwise_score

__all__ = ["LAMBDA_GRID", "global_score", "layerwise_score", "make_score"]


def make_score(variant, strength):
    """Bind one model and lambda for all configurations in an evaluation."""
    strength = validate_lambda(strength)
    models = {"layerwise": layerwise_score, "global": global_score}
    if variant not in models:
        raise ValueError("MB_NSC_VARIANT must be 'layerwise' or 'global'")
    model = models[variant]

    def fixed_score(config):
        return model(config, strength)

    return fixed_score
