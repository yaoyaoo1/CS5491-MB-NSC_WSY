"""Global quadratic MB-NSC: aggregate A,F before squaring the difference."""

from nsc import original_nsc

from .common import (
    LAMBDA_GRID, capacity_pairs, module_capacities, quadratic_balance,
    stable_sum, validate_capacity, validate_lambda,
)


def _totals(capacities):
    values = capacity_pairs(capacities)
    return (
        stable_sum(c.attention for c in values),
        stable_sum(c.ffn for c in values),
    )


def penalty(capacities):
    attention, ffn = _totals(capacities)
    return quadratic_balance(attention, ffn, 0).penalty


def score_capacities(capacities, strength):
    strength = validate_lambda(strength)
    attention, ffn = _totals(capacities)
    return quadratic_balance(attention, ffn, strength).score


def score(config, strength):
    strength = validate_lambda(strength)
    if strength == 0:
        result = original_nsc(config)
        validate_capacity(result, "Original NSC")
        return result
    return score_capacities(module_capacities(config), strength)
