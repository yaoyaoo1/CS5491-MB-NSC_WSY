"""Shared Starter grouping and quadratic balance arithmetic.

Equivalent expressions handle balanced, strongly imbalanced, and subnormal
inputs without clipping, epsilon, or changing the quadratic definition.
"""

from dataclasses import dataclass
import math
from numbers import Real
import sys

from nsc import layers, psi_mp


LAMBDA_GRID = tuple(j / 10 for j in range(11))


def _finite_real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_lambda(strength):
    """Lambda is penalty strength, never the balance multiplier."""
    result = _finite_real(strength, "lambda")
    if not 0 <= result <= 1:
        raise ValueError("lambda must be in [0, 1]")
    return result


def validate_capacity(value, name="capacity"):
    result = _finite_real(value, name)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def stable_sum(values):
    """Reject nonfinite inputs/results, including overflowing totals."""
    try:
        result = math.fsum(validate_capacity(v) for v in values)
    except OverflowError as exc:
        raise ValueError("capacity total must be finite") from exc
    return validate_capacity(result, "capacity total")


@dataclass(frozen=True)
class LayerCapacity:
    attention: float
    ffn: float

    def __post_init__(self):
        object.__setattr__(self, "attention", validate_capacity(self.attention))
        object.__setattr__(self, "ffn", validate_capacity(self.ffn))
        stable_sum((self.attention, self.ffn))

    @property
    def total(self):
        return stable_sum((self.attention, self.ffn))


@dataclass(frozen=True)
class Balance:
    total: float
    signed_imbalance: float
    deviation: float
    penalty: float
    multiplier: float
    score: float


def _stable_pair_score(attention, ffn, total, strength):
    """Evaluate the same homogeneous score in a safe floating-point regime."""
    if total < sys.float_info.min:
        # Power-of-two scaling is exact. Do the blend at normal precision,
        # then round once back to the subnormal grid, rather than rounding
        # both weighted terms to that grid before adding them.
        _, exponent = math.frexp(total)
        scaled = _stable_pair_score(
            math.ldexp(attention, -exponent),
            math.ldexp(ffn, -exponent),
            math.ldexp(total, -exponent),
            strength,
        )
        return math.ldexp(scaled, exponent)
    smaller, larger = min(attention, ffn), max(attention, ffn)
    if smaller >= 0.5 * larger:
        # Here |x| <= 1/3: no near-total subtraction. The multiplier is
        # bounded by 1, so multiplication cannot overflow a finite total.
        # The alternative harmonic product can round to inf near MAX_FLOAT.
        signed = (attention - ffn) / total
        return total * (1.0 - strength * (signed * signed))
    # Here 4*a*f/C <= (8/9)*C, away from the overflow boundary. Keep the
    # weak module unscaled, including when the ratio of inputs underflows.
    joint = smaller * (4.0 * (larger / total))
    return stable_sum(((1.0 - strength) * total, strength * joint))


def quadratic_balance(attention, ffn, strength):
    """Return C, signed x, D, P, b, and an equivalent nonnegative score."""
    strength = validate_lambda(strength)
    attention = validate_capacity(attention, "Attention capacity")
    ffn = validate_capacity(ffn, "FFN capacity")
    total = stable_sum((attention, ffn))
    if total == 0:
        return Balance(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    signed = (attention - ffn) / total
    deviation = signed * signed
    penalty = total * deviation
    multiplier = 1.0 - strength * deviation
    score = validate_capacity(
        _stable_pair_score(attention, ffn, total, strength), "balanced score"
    )
    return Balance(total, signed, deviation, penalty, multiplier, score)


def module_capacities(config):
    """Reuse unmodified Starter psi_mp defaults and its two-argument keys."""
    d, heads, inner = layers(config)
    return tuple(
        LayerCapacity(
            3 * h * psi_mp(d, d // h) + psi_mp(d, d),
            2 * psi_mp(d, f),
        )
        for h, f in zip(heads, inner)
    )


def capacity_pairs(capacities):
    """Validate synthetic/module pairs without relaxing config validation."""
    return tuple(
        item if isinstance(item, LayerCapacity) else LayerCapacity(*item)
        for item in capacities
    )
