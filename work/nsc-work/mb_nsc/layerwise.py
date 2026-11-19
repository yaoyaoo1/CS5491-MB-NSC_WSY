"""Layer-wise quadratic MB-NSC: S_L = S0 - lambda * sum(C_l D_l)."""

from nsc import original_nsc

from .common import (
    LAMBDA_GRID, capacity_pairs, module_capacities, quadratic_balance,
    stable_sum, validate_capacity, validate_lambda,
)


def penalty(capacities):
    return stable_sum(
        quadratic_balance(c.attention, c.ffn, 0).penalty
        for c in capacity_pairs(capacities)
    )


def score_capacities(capacities, strength):
    """Capacity-level mechanism API, including synthetic zero layers."""
    strength = validate_lambda(strength)
    return stable_sum(
        quadratic_balance(c.attention, c.ffn, strength).score
        for c in capacity_pairs(capacities)
    )


def score(config, strength):
    strength = validate_lambda(strength)
    if strength == 0:
        result = original_nsc(config)
        validate_capacity(result, "Original NSC")
        return result
    return score_capacities(module_capacities(config), strength)
