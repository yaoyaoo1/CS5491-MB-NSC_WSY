"""Independent full-archive capacity audit and binary64 boundary regressions.

Only architecture configuration values are consumed; identities below are
reported by their one-based value index and structural fields, never by ID.
"""

from fractions import Fraction
import math
import sys

import pytest

import nsc
from mb_nsc import common, global_model, layerwise
from conftest import AUDIT31_SUMMARY


def exact_score(a, f, strength):
    """Exact rational evaluation of binary64 inputs; no shared model algebra."""
    a, f, strength = map(Fraction.from_float, map(float, (a, f, strength)))
    total = a + f
    if not total:
        return 0.0
    return float(total - strength * (a - f) ** 2 / total)


def test_maxfinite_valid_balanced_pair():
    maximum = sys.float_info.max
    a = math.nextafter(maximum / 2, 0.0)
    f = math.nextafter(maximum / 2, math.inf)
    assert math.fsum((a, f)) == maximum
    actual = common.quadratic_balance(a, f, 1).score
    assert math.isfinite(actual)
    assert math.isclose(actual, exact_score(a, f, 1), rel_tol=1e-12, abs_tol=0)


def test_subnormal_balanced_range():
    unit = math.ulp(0.0)
    a, f = 2 * unit, unit
    actual = common.quadratic_balance(a, f, .5).score
    assert actual <= a + f
    assert actual == exact_score(a, f, .5) == 3 * unit


@pytest.mark.parametrize("strength", common.LAMBDA_GRID)
def test_near_maxfinite_grid_and_models(strength):
    maximum = sys.float_info.max
    half = maximum / 2
    smaller_values = [half]
    for _ in range(8):
        smaller_values.append(math.nextafter(smaller_values[-1], 0.0))
    for smaller in smaller_values:
        # Round the residual DOWN: a nearest-rounded MAX-smaller can make
        # the exact sum exceed the finite range for some adjacent floats.
        # Invalid overflowing totals are tested separately, not skipped.
        larger = math.nextafter(maximum - smaller, 0.0)
        for a, f in [(smaller, larger), (larger, smaller)]:
            total = math.fsum((a, f))
            expected = exact_score(a, f, strength)
            result = common.quadratic_balance(a, f, strength)
            assert math.isfinite(result.score) and 0 <= result.score <= total
            assert math.isclose(result.score, expected, rel_tol=1e-12, abs_tol=0)
            for model in (layerwise, global_model):
                actual = model.score_capacities([(a, f)], strength)
                assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0)
                assert math.isfinite(model.penalty([(a, f)]))


def test_subnormal_sweep_exact_reference_and_bounds():
    unit = math.ulp(0.0)
    max_units_error = 0
    count = 0
    for i in range(9):
        for j in range(9):
            a, f = i * unit, j * unit
            for strength in common.LAMBDA_GRID:
                expected = exact_score(a, f, strength)
                result = common.quadratic_balance(a, f, strength)
                assert 0 <= result.score <= result.total
                # At subnormal resolution use an explicit one-ULP budget,
                # not abs_tol=1e-12 which would conceal every small error.
                units_error = abs(result.score - expected) / unit
                assert units_error <= 1
                max_units_error = max(max_units_error, units_error)
                count += 1
    AUDIT31_SUMMARY["subnormal"] = {
        "checks": count, "max_error_in_min_subnormal_units": max_units_error
    }


def test_single_sided_large_small_and_lambda_one():
    for a in [math.ulp(0.0), sys.float_info.min, 1e-300, 1, 1e300, sys.float_info.max]:
        for pair in [(a, 0.0), (0.0, a)]:
            for strength in common.LAMBDA_GRID:
                result = common.quadratic_balance(*pair, strength)
                assert result.deviation == 1
                assert result.penalty == a
                assert result.multiplier == 1 - strength
                expected = exact_score(*pair, strength)
                assert abs(result.score - expected) <= math.ulp(expected)
            assert common.quadratic_balance(*pair, 1).score == 0


def test_finite_harmonic_despite_naive_product_overflow():
    for pair in [(1e308, 1e307), (1e307, 1e308), (1e300, 1e-300), (1e-300, 1e300)]:
        a, f = pair
        for strength in common.LAMBDA_GRID:
            expected = exact_score(a, f, strength)
            result = common.quadratic_balance(a, f, strength)
            assert math.isfinite(result.score) and result.score > 0
            assert math.isclose(result.score, expected, rel_tol=1e-12, abs_tol=0)
    assert math.isinf(4 * 1e308 * 1e307)


def test_nonfinite_true_total_still_rejected():
    with pytest.raises(ValueError):
        common.quadratic_balance(sys.float_info.max, sys.float_info.max, 1)


def test_adjacent_endpoint_lambdas():
    # Numerical stress points, not additions to the formal search grid.
    strengths = [math.nextafter(0.0, 1.0), math.nextafter(1.0, 0.0), 1.0]
    maximum = sys.float_info.max
    pairs = [(1e300, 1e-300), (1e-300, 1e300), (maximum, 0.0),
             (.75 * maximum, math.nextafter(.25 * maximum, 0.0))]
    for a, f in pairs:
        for strength in strengths:
            expected = exact_score(a, f, strength)
            actual = common.quadratic_balance(a, f, strength).score
            assert math.isfinite(actual) and 0 <= actual <= math.fsum((a, f))
            assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=math.ulp(0.0))


def test_numerical_branch_boundary_is_same_formula():
    for scale in (1e-300, 1.0, 1e300):
        a = 2 * scale
        for f in (math.nextafter(scale, 0.0), scale, math.nextafter(scale, math.inf)):
            for strength in common.LAMBDA_GRID:
                expected = exact_score(a, f, strength)
                actual = common.quadratic_balance(a, f, strength).score
                assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0)


def independent_capacities(config):
    """Separate schema expansion and explicit per-projection/head summation.

    Reuse the required Starter MP integral, but NOT common.module_capacities,
    nsc.layer_capacities, or their compact capacity grouping expression.
    """
    d, length = config["d_model"], config["n_layer"]
    heads, inner = config["n_head"], config["d_inner"]
    heads = [heads] * length if isinstance(heads, int) else list(heads)
    inner = [inner] * length if isinstance(inner, int) else list(inner)
    assert len(heads) == len(inner) == length
    result = []
    for h, f in zip(heads, inner):
        assert d % h == 0
        projections = [nsc.psi_mp(d, d // h) for _ in range(h) for _ in ("Q", "K", "V")]
        projections.append(nsc.psi_mp(d, d))  # One full O, once per layer.
        attention = math.fsum(projections)
        ffn = math.fsum((nsc.psi_mp(d, f), nsc.psi_mp(f, d)))
        result.append((attention, ffn))
    return result


def test_all_200_independent_capacities(architecture_configs):
    def identity(index, config):
        return {"value_index_1based": index, "config": {
            key: config[key] for key in ("d_model", "n_layer", "n_head", "d_inner")
        }}

    metrics = {}
    for name in ("shared_vs_original", "independent_vs_original", "shared_vs_independent_attention", "shared_vs_independent_ffn"):
        metrics[name] = {"max_absolute_error": -1, "max_relative_error": -1,
                         "absolute_tie_count": 0, "relative_tie_count": 0}

    def record(name, actual, expected, ident):
        error = abs(actual - expected)
        relative = error / abs(expected) if expected else error
        row = metrics[name]
        for label, value in (("absolute", error), ("relative", relative)):
            key = "max_" + label + "_error"
            if value > row[key]:
                row[key] = value
                row[label + "_architecture"] = ident
                row[label + "_tie_count"] = 1
            elif value == row[key]:
                row[label + "_tie_count"] += 1
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)

    shared_exact_count = 0
    for index, config in enumerate(architecture_configs, 1):
        independent = independent_capacities(config)
        shared = common.module_capacities(config)
        assert len(independent) == len(shared) == config["n_layer"]
        baseline = nsc.original_nsc(config)
        shared_total = common.stable_sum(item.total for item in shared)
        independent_total = math.fsum(math.fsum(pair) for pair in independent)
        ident = identity(index, config)
        record("shared_vs_original", shared_total, baseline, ident)
        record("independent_vs_original", independent_total, baseline, ident)
        shared_exact_count += shared_total == baseline
        for layer, (actual, expected) in enumerate(zip(shared, independent), 1):
            layer_ident = dict(ident, layer_1based=layer)
            record("shared_vs_independent_attention", actual.attention, expected[0], layer_ident)
            record("shared_vs_independent_ffn", actual.ffn, expected[1], layer_ident)
    assert shared_exact_count == 200
    AUDIT31_SUMMARY["capacity_reproduction"] = {
        "architectures": len(architecture_configs),
        "shared_exact_matches": shared_exact_count,
        "metrics": metrics,
    }
