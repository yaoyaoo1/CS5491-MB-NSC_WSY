"""Synthetic mechanisms, independent oracles, and official import compatibility."""

import builtins
from copy import deepcopy
from fractions import Fraction
import inspect
import math
from unittest.mock import Mock

import pytest

import nsc
from mb_nsc import common, global_model, layerwise
from conftest import NUMERIC_ERRORS


REL_TOL = ABS_TOL = 1e-12


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL), (actual, expected)


class TestLayerwiseSynthetic:
    @pytest.mark.parametrize("pair,expected", [((5, 5), 10), ((8, 2), 8.2), ((2, 8), 8.2)])
    def test_03_fixed_total(self, pair, expected):
        close(layerwise.score_capacities([pair], .5), expected)

    def test_04_fixed_proportion(self):
        for pair, expected in [((5, 5), 10), ((10, 10), 20), ((15, 15), 30)]:
            for strength in common.LAMBDA_GRID:
                assert layerwise.score_capacities([pair], strength) == expected
        base = layerwise.score_capacities([(8, 2)], .5)
        for k in [1, 2, 3]:
            close(layerwise.score_capacities([(8*k, 2*k)], .5), k * base)

    def test_05_append_zero(self):
        for strength in common.LAMBDA_GRID:
            assert layerwise.score_capacities([(9, 1), (1, 9)], strength) == layerwise.score_capacities([(9, 1), (0, 0), (1, 9)], strength)
            assert layerwise.score_capacities([(0, 0)], strength) == 0

    def test_06_local_penalty(self):
        close(layerwise.penalty([(9, 1), (1, 9)]), 12.8)
        close(layerwise.score_capacities([(9, 1), (1, 9)], .5), 13.6)


class TestGlobalSynthetic:
    @pytest.mark.parametrize("pair,expected", [((5, 5), 10), ((8, 2), 8.2), ((2, 8), 8.2)])
    def test_03_fixed_total_and_single_layer_agreement(self, pair, expected):
        close(global_model.score_capacities([pair], .5), expected)
        assert global_model.score_capacities([pair], .5) == layerwise.score_capacities([pair], .5)

    def test_04_fixed_proportion(self):
        for pair, expected in [((5, 5), 10), ((10, 10), 20), ((15, 15), 30)]:
            for strength in common.LAMBDA_GRID:
                assert global_model.score_capacities([pair], strength) == expected
        base = global_model.score_capacities([(8, 2)], .5)
        for k in [1, 2, 3]:
            close(global_model.score_capacities([(8*k, 2*k)], .5), k * base)

    def test_05_append_zero_and_total_zero(self):
        for strength in common.LAMBDA_GRID:
            assert global_model.score_capacities([(9, 1), (1, 9)], strength) == global_model.score_capacities([(9, 1), (0, 0), (1, 9)], strength)
            assert global_model.score_capacities([(0, 0), (0, 0)], strength) == 0

    def test_06_signed_cancellation_before_square(self):
        assert global_model.penalty([(9, 1), (1, 9)]) == 0
        assert global_model.score_capacities([(9, 1), (1, 9)], .5) == 20


class TestMathematicalIdentities:
    @pytest.mark.parametrize("pairs,weights,xs,mean,variance,pl,pg", [
        ([(9, 1), (1, 9)], [Fraction(1, 2)]*2, [.8, -.8], 0, .64, 12.8, 0),
        ([(8, 2), (16, 4)], [Fraction(1, 3), Fraction(2, 3)], [.6, .6], .6, 0, 10.8, 10.8),
        ([(9, 1), (1, 3)], [Fraction(5, 7), Fraction(2, 7)], [.8, -.5], Fraction(3, 7), Fraction(169, 490), Fraction(37, 5), Fraction(18, 7)),
    ])
    def test_06_07_weighted_variance(self, pairs, weights, xs, mean, variance, pl, pg):
        totals = [a + f for a, f in pairs]
        total = sum(totals)
        actual_weights = [c / total for c in totals]
        signed = [common.quadratic_balance(a, f, 0).signed_imbalance for a, f in pairs]
        mu = math.fsum(w*x for w, x in zip(actual_weights, signed))
        var = math.fsum(w*(x-mu)**2 for w, x in zip(actual_weights, signed))
        for value, expected in zip(actual_weights, weights):
            close(value, float(expected))
        for value, expected in zip(signed, xs):
            close(value, expected)
        close(mu, float(mean))
        close(var, float(variance))
        close(layerwise.penalty(pairs), float(pl))
        close(global_model.penalty(pairs), float(pg))
        close(layerwise.penalty(pairs) - global_model.penalty(pairs), total * var)
        if variance == 0:
            for strength in common.LAMBDA_GRID:
                close(layerwise.score_capacities(pairs, strength), global_model.score_capacities(pairs, strength))

    def test_08_pairing_and_layer_permutation(self):
        original, repaired = [(9, 1), (1, 9)], [(9, 9), (1, 1)]
        assert sum(a+f for a, f in original) == sum(a+f for a, f in repaired) == 20
        close(layerwise.penalty(original), 12.8)
        assert layerwise.penalty(repaired) == 0
        for strength in common.LAMBDA_GRID:
            assert global_model.score_capacities(original, strength) == global_model.score_capacities(repaired, strength) == 20
            close(layerwise.score_capacities(original, strength), 20 - 12.8 * strength)
            for model in (layerwise, global_model):
                assert model.score_capacities(original, strength) == model.score_capacities(list(reversed(original)), strength)

    def test_09_ranges_monotonicity_and_fraction_oracle(self):
        datasets = [[(0, 0)], [(0, 10)], [(5, 5)], [(8, 2)], [(9, 1), (1, 9)], [(9, 1), (1, 3)], [(0, 0), (2, 8), (4, 1)]]
        for pairs in datasets:
            total = sum(a+f for a, f in pairs)
            pl = layerwise.penalty(pairs)
            pg = global_model.penalty(pairs)
            assert 0 <= pg <= pl + ABS_TOL <= total + 2 * ABS_TOL
            exact_pl = sum((Fraction((a-f)**2, a+f) for a, f in pairs if a+f), Fraction())
            a_total, f_total = map(sum, zip(*pairs))
            exact_pg = Fraction((a_total-f_total)**2, total) if total else Fraction()
            previous_l = previous_g = math.inf
            for j, strength in enumerate(common.LAMBDA_GRID):
                sl = layerwise.score_capacities(pairs, strength)
                sg = global_model.score_capacities(pairs, strength)
                close(sl, float(Fraction(total) - Fraction(j, 10)*exact_pl))
                close(sg, float(Fraction(total) - Fraction(j, 10)*exact_pg))
                assert (1-strength)*total <= sl + ABS_TOL
                assert sl <= sg + ABS_TOL <= total + 2*ABS_TOL
                assert sl <= previous_l + ABS_TOL and sg <= previous_g + ABS_TOL
                previous_l, previous_g = sl, sg
                for index in range(len(pairs)):
                    for module in (0, 1):
                        increased = [list(pair) for pair in pairs]
                        increased[index][module] += 1
                        assert layerwise.score_capacities(increased, strength) + ABS_TOL >= sl
                        assert global_model.score_capacities(increased, strength) + ABS_TOL >= sg

    def test_09_search_grid(self):
        expected = (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1)
        assert common.LAMBDA_GRID == expected
        assert len(set(common.LAMBDA_GRID)) == 11
        assert common.LAMBDA_GRID[0] == 0 and common.LAMBDA_GRID[-1] == 1
        assert layerwise.LAMBDA_GRID is global_model.LAMBDA_GRID is common.LAMBDA_GRID


class TestSpectralGrouping:
    @pytest.mark.parametrize("model", [layerwise, global_model])
    def test_real_config_stable_score_against_original_deduction(self, model):
        # Exercise the real Starter summation order, not only synthetic pairs.
        configurations = [
            dict(d_model=64, n_layer=2, n_head=[2, 4], d_inner=[128, 256]),
            dict(d_model=128, n_layer=2, n_head=4, d_inner=[256, 512]),
            dict(d_model=512, n_layer=3, n_head=[4, 8, 16], d_inner=[512, 1024, 1536]),
        ]
        error = 0.0
        for cfg in configurations:
            baseline = nsc.original_nsc(cfg)
            penalty = model.penalty(common.module_capacities(cfg))
            for strength in common.LAMBDA_GRID:
                stable = model.score(cfg, strength)
                deduction = baseline - strength * penalty
                close(stable, deduction)
                error = max(error, abs(stable - deduction))
        NUMERIC_ERRORS[model.__name__.rsplit(".", 1)[-1]] = error

    def test_08_independent_projection_sentinels(self, monkeypatch):
        # Independent hand expectations: H=2 -> 6*11+100=166;
        # H=4 -> 12*7+100=184. O contributes 100 ONCE, FFN is 26.
        psi = Mock(side_effect=lambda m, n: {(64, 32): 11, (64, 16): 7, (64, 64): 100, (64, 256): 13}[(m, n)])
        monkeypatch.setattr(common, "psi_mp", psi)
        cfg = dict(d_model=64, n_layer=2, n_head=[2, 4], d_inner=256)
        result = common.module_capacities(cfg)
        assert [(c.attention, c.ffn, c.total) for c in result] == [(166, 26, 192), (184, 26, 210)]
        assert [call.args for call in psi.call_args_list].count((64, 64)) == 2
        assert all(not call.kwargs and len(call.args) == 2 for call in psi.call_args_list)

    def test_08_real_spectral_terms_independent_head_loop(self):
        for h in [1, 2, 4, 8]:
            cfg = dict(d_model=64, n_layer=1, n_head=h, d_inner=128)
            record, = common.module_capacities(cfg)
            qkv = math.fsum(nsc.psi_mp(64, 64//h) for _ in range(h) for _ in ("Q", "K", "V"))
            close(record.attention, qkv + nsc.psi_mp(64, 64))
            close(record.ffn, nsc.psi_mp(64, 128) + nsc.psi_mp(128, 64))
            close(record.total, nsc.original_nsc(cfg))

    def test_08_official_reorder_and_joint_permutation(self):
        cfg = dict(d_model=128, n_layer=2, n_head=[4, 4], d_inner=[256, 512])
        reverse = dict(cfg, d_inner=[512, 256])
        mixed = dict(d_model=64, n_layer=2, n_head=[2, 4], d_inner=[128, 256])
        permuted = dict(mixed, n_head=[4, 2], d_inner=[256, 128])
        for strength in common.LAMBDA_GRID:
            for model in (layerwise, global_model):
                close(model.score(cfg, strength), model.score(reverse, strength))
                close(model.score(mixed, strength), model.score(permuted, strength))


class TestOfficialInterface:
    @pytest.mark.parametrize("model", [layerwise, global_model])
    def test_invalid_lambda_rejected_by_both_public_apis(self, model):
        cfg = dict(d_model=64, n_layer=1, n_head=4, d_inner=128)
        for invalid in [-.1, 1.1, math.nan, math.inf, -math.inf, True, None]:
            with pytest.raises(ValueError):
                model.score(cfg, invalid)
            with pytest.raises(ValueError):
                model.score_capacities([(5, 5)], invalid)

    @pytest.mark.parametrize("variant,model", [("layerwise", layerwise), ("global", global_model)])
    def test_evaluator_style_import_fixed_process_and_no_input_mutation(self, variant, model, student_loader, monkeypatch):
        module = student_loader(variant, .5)
        cfg = dict(d_model=64, n_layer=2, n_head=[2, 4], d_inner=[128, 256])
        snapshot = deepcopy(cfg)
        assert list(inspect.signature(module.score).parameters) == ["config"]
        assert list(inspect.signature(module.ablation).parameters) == ["config"]
        expected = model.score(cfg, .5)
        monkeypatch.setenv("MB_NSC_VARIANT", "global" if variant == "layerwise" else "layerwise")
        monkeypatch.setenv("MB_NSC_LAMBDA", "1")
        assert module.score(cfg) == expected
        assert module.ablation(cfg) == nsc.original_nsc(cfg)
        assert cfg == snapshot
        assert isinstance(expected, float) and math.isfinite(expected)

    def test_missing_or_invalid_process_settings(self, student_loader):
        module = student_loader()
        with pytest.raises(ValueError):
            module.score({})
        for variant, strength in [("unknown", .5), ("layerwise", -1), ("global", "nan"), ("global", "inf"), ("global", "nonsense"), (None, .5), ("global", None)]:
            with pytest.raises(ValueError):
                student_loader(variant, strength)

    @pytest.mark.parametrize("model", [layerwise, global_model])
    def test_05_starter_validation_unchanged(self, model):
        cfg = dict(d_model=64, n_layer=2, n_head=4, d_inner=128)
        invalid = [dict(d_model=0), dict(n_layer=0), dict(d_inner=0), dict(n_head=0), dict(n_head=3), dict(n_head=[2]), dict(d_inner=[128]), dict(d_model=True)]
        for change in invalid:
            for strength in [0, .5, 1]:
                with pytest.raises(ValueError):
                    model.score(dict(cfg, **change), strength)

    def test_score_and_ablation_do_not_open_files(self, student_loader, monkeypatch):
        students = [student_loader("layerwise", .5), student_loader("global", .5)]
        cfg = dict(d_model=64, n_layer=1, n_head=4, d_inner=128)
        def forbidden(*args, **kwargs):
            raise AssertionError("Scoring must not open files")
        monkeypatch.setattr(builtins, "open", forbidden)
        from pathlib import Path
        monkeypatch.setattr(Path, "open", forbidden)
        for student in students:
            assert math.isfinite(student.score(cfg))
            assert math.isfinite(student.ablation(cfg))

    @pytest.mark.parametrize("model", [layerwise, global_model])
    def test_zero_lambda_uses_direct_original_function(self, model, monkeypatch):
        original = Mock(return_value=123.25)
        monkeypatch.setattr(model, "original_nsc", original)
        monkeypatch.setattr(model, "module_capacities", Mock(side_effect=AssertionError("Must use direct original path")))
        config = object()
        assert model.score(config, 0) == 123.25
        original.assert_called_once_with(config)
