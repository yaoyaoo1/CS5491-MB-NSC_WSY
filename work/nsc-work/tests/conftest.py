"""Test-only data-access guard; no evaluator/provenance imports."""

import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


WORK = Path(__file__).resolve().parents[1]
PROJECT = WORK.parents[1]
ORIGINAL = PROJECT / "normal" / "nsc-starter"
ARCHITECTURES = WORK / "data" / "architectures.json"
JSON_OPENS = set()
RECOVERY_COUNTS = {}
NUMERIC_ERRORS = {}
AUDIT31_SUMMARY = {}
PROTECTED_NAMES = {"nsc.py", "evaluate.py", "check_starter.py", "README.md"}


def _access_guard(event, args):
    if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    flags = args[2] or 0
    writing = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
    if PROJECT not in path.parents:
        return
    if path.suffix.lower() == ".json":
        if path != ARCHITECTURES or writing:
            raise PermissionError(f"Forbidden project JSON access: {path.name}")
        JSON_OPENS.add(path.name)
    if writing and (
        ORIGINAL in path.parents
        or WORK / "data" in path.parents
        or (path.parent == WORK and path.name in PROTECTED_NAMES)
    ):
        raise PermissionError(f"Protected file write: {path}")


sys.addaudithook(_access_guard)


@pytest.fixture
def student_loader(monkeypatch):
    """Use exactly the evaluator's import mechanism, without importing it."""
    def load(variant=None, strength=None):
        with monkeypatch.context() as env:
            env.delenv("MB_NSC_VARIANT", raising=False)
            env.delenv("MB_NSC_LAMBDA", raising=False)
            if variant is not None:
                env.setenv("MB_NSC_VARIANT", variant)
            if strength is not None:
                env.setenv("MB_NSC_LAMBDA", str(strength))
            spec = importlib.util.spec_from_file_location(
                "student_submission", WORK / "student_score.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    return load


@pytest.fixture(scope="session")
def architecture_configs():
    # Values only: never pass dictionary keys/architecture IDs to the scorer.
    values = list(json.loads(ARCHITECTURES.read_text(encoding="utf-8")).values())
    assert len(values) == 200, f"Expected 200 configurations, found {len(values)}"
    assert all(isinstance(config, dict) for config in values)
    return values


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line(
        "Project JSON contents opened: " + (", ".join(sorted(JSON_OPENS)) or "none")
    )
    if RECOVERY_COUNTS:
        terminalreporter.write_line("Strict recovery configurations: " + str(RECOVERY_COUNTS))
    if NUMERIC_ERRORS:
        terminalreporter.write_line("Max stable/deduction absolute error: " + str(NUMERIC_ERRORS))
    if AUDIT31_SUMMARY:
        terminalreporter.write_line("Phase 3.1 audit: " + json.dumps(AUDIT31_SUMMARY, sort_keys=True))


def pytest_collection_modifyitems(items):
    """Keep the required phase order in a single repeatable suite command."""
    stages = {
        "TestCommonCapacities": 2,
        "TestQuadraticFunctions": 3,
        "TestLayerwiseSynthetic": 4,
        "TestGlobalSynthetic": 5,
        "TestMathematicalIdentities": 6,
        "TestNumericalStability": 6,
        "TestSpectralGrouping": 7,
        "TestOfficialInterface": 7,
    }
    items.sort(key=lambda item: (
        stages.get(item.cls.__name__ if item.cls else "", 8), item.nodeid
    ))
