"""Shared arithmetic, zero rules, and independent numerical references."""

from decimal import Decimal, localcontext
import math

import pytest

import nsc
from mb_nsc import common


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)


class TestCommonCapacities:
    def test_shared_starter_function_and_defaults(self):
        assert common.psi_mp is nsc.psi_mp
        assert common.layers is nsc.layers
        assert nsc.psi_mp.__wrapped__.__defaults__ == (0.02, 512)

    def test_capacity_records_and_stable_sum(self):
        value = common.LayerCapacity(8, 2)
        assert value.total == 10
        assert common.capacity_pairs([(8, 2), value]) == (value, value)
        assert common.stable_sum([1, 2, 3]) == 6

    def test_cache_keys_match_starter(self):
        cfg = dict(d_model=64, n_layer=2, n_head=[2, 4], d_inner=[128, 256])
        nsc.original_nsc(cfg)
        misses = nsc.psi_mp.cache_info().misses
        capacities = common.module_capacities(cfg)
        assert nsc.psi_mp.cache_info().misses == misses
        assert common.stable_sum(c.total for c in capacities) == nsc.original_nsc(cfg)

    def test_sum_overflow_is_rejected(self):
        with pytest.raises(ValueError):
            common.stable_sum((1e308, 1e308))


class TestQuadraticFunctions:
    @pytest.mark.parametrize("a,f,x,d,p", [(5, 5, 0, 0, 0), (8, 2, .6, .36, 3.6), (2, 8, -.6, .36, 3.6)])
    def test_02_signed_deviation_penalty(self, a, f, x, d, p):
        actual = common.quadratic_balance(a, f, .5)
        exchanged = common.quadratic_balance(f, a, .5)
        close(actual.signed_imbalance, x)
        close(actual.deviation, d)
        close(actual.penalty, p)
        assert actual.deviation == exchanged.deviation
        assert actual.penalty == exchanged.penalty
        assert actual.signed_imbalance == -exchanged.signed_imbalance

    @pytest.mark.parametrize("a,f,p", [(8, 2, 3.6), (16, 4, 7.2), (24, 6, 10.8)])
    def test_04_nonbalanced_scaling(self, a, f, p):
        actual = common.quadratic_balance(a, f, .5)
        close(actual.deviation, .36)
        close(actual.penalty, p)

    @pytest.mark.parametrize("a,f,d,p,b,score", [(0, 10, 1, 10, .5, 5), (10, 0, 1, 10, .5, 5), (0, 0, 0, 0, 1, 0)])
    def test_05_zero_capacity(self, a, f, d, p, b, score):
        actual = common.quadratic_balance(a, f, .5)
        assert (actual.deviation, actual.penalty, actual.multiplier, actual.score) == (d, p, b, score)

    @pytest.mark.parametrize("invalid", [-1, math.nan, math.inf, -math.inf, True, "5", None])
    def test_05_invalid_capacity(self, invalid):
        for pair in [(invalid, 1), (1, invalid)]:
            with pytest.raises(ValueError):
                common.quadratic_balance(*pair, .5)
            with pytest.raises(ValueError):
                common.LayerCapacity(*pair)

    @pytest.mark.parametrize("invalid", [-.1, 1.1, math.nan, math.inf, -math.inf, True, "0.5", None])
    def test_09_invalid_lambda(self, invalid):
        with pytest.raises(ValueError):
            common.validate_lambda(invalid)
        with pytest.raises(ValueError):
            common.quadratic_balance(0, 0, invalid)

    def test_09_multiplier_and_endpoints(self):
        for strength in common.LAMBDA_GRID:
            for a, f in [(0, 0), (0, 10), (10, 0), (5, 5), (8, 2)]:
                result = common.quadratic_balance(a, f, strength)
                close(result.multiplier, 1 - strength * result.deviation)
                assert 1 - strength <= result.multiplier <= 1
        assert common.quadratic_balance(8, 2, 0).score == 10
        close(common.quadratic_balance(8, 2, 1).score, 6.4)
        assert common.quadratic_balance(0, 10, 1).score == 0


class TestNumericalStability:
    def test_penalty_crosscheck_well_conditioned(self):
        for strength in common.LAMBDA_GRID:
            for a, f in [(5, 5), (8, 2), (2, 8), (9, 1), (1, 3), (0, 0), (0, 10), (1e10, 1e8)]:
                result = common.quadratic_balance(a, f, strength)
                close(result.score, result.total - strength * result.penalty)
                close(result.score, result.total * result.multiplier)

    def test_cancellation_against_independent_decimal(self):
        # Binary64 subtraction loses the weak module; do NOT loosen tolerance.
        actual = common.quadratic_balance(1e16, 1, 1)
        naive = actual.total - actual.penalty
        with localcontext() as context:
            context.prec = 80
            a, f = Decimal(10) ** 16, Decimal(1)
            reference = (a + f) - (a - f) ** 2 / (a + f)
        assert naive == 0
        close(actual.score, float(reference))
        close(actual.score, 4)
        assert actual.score > naive

    @pytest.mark.parametrize("a,f", [(1e300, 1e-300), (1e-300, 1e300), (1e-300, 1e-300), (1e300, 1e300)])
    def test_extreme_finite_pair(self, a, f):
        with localcontext() as context:
            context.prec = 80
            da, df = Decimal.from_float(a), Decimal.from_float(f)
            reference = 4 * da * df / (da + df)
        actual = common.quadratic_balance(a, f, 1).score
        assert math.isfinite(actual) and actual > 0
        assert math.isclose(actual, float(reference), rel_tol=1e-12, abs_tol=0)
