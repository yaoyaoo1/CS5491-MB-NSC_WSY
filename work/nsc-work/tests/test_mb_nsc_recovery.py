"""Only architecture specification values are loaded, never labels/results."""

from copy import deepcopy
import math

import pytest

import nsc
from mb_nsc import global_model, layerwise
from conftest import RECOVERY_COUNTS


@pytest.mark.parametrize("variant,model", [("layerwise", layerwise), ("global", global_model)])
def test_01_all_200_exact_original_and_ablation(variant, model, architecture_configs, student_loader):
    student = student_loader(variant, 0)
    passed = 0
    for config in architecture_configs:
        snapshot = deepcopy(config)
        nsc.layers(config)  # A malformed archive entry fails, never skips.
        baseline = nsc.original_nsc(config)
        outputs = (model.score(config, 0), student.score(config), student.ablation(config))
        assert all(isinstance(value, float) and math.isfinite(value) for value in outputs)
        assert all(value == baseline for value in outputs)  # No rounding/tolerance.
        assert config == snapshot
        passed += 1
    assert passed == 200
    RECOVERY_COUNTS[variant] = passed
