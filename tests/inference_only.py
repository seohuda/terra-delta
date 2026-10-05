"""Run the complete collection without the user-forbidden legacy training calls.

Usage: PYTHONPATH=tests:src pytest -p inference_only
Skipped legacy tests call backward/fit_verifier or construct a Trainer optimizer.
All other tests, including every v2.3 test, retain the runtime prohibition.
"""

import ast
import inspect
import textwrap

import pytest

TRAINER_TESTS = {
    "tests/test_training.py::test_checkpoint_resume_restores_state_without_training",
    "tests/test_training.py::test_early_stopping_observation_no_training",
}


def forbidden(*args, **kwargs):
    raise AssertionError("Inference-only validation forbids optimizer, backward and fitting")


def pytest_configure(config):
    import torch
    import terradelta.training.verifier
    torch.optim.Optimizer.__init__ = forbidden
    torch.Tensor.backward = forbidden
    torch.autograd.backward = forbidden
    terradelta.training.verifier.fit_verifier = forbidden


def pytest_collection_modifyitems(config, items):
    excluded = []
    for item in items:
        source = ast.parse(textwrap.dedent(inspect.getsource(item.obj)))
        prohibited = any(isinstance(node, ast.Call) and (
            isinstance(node.func, ast.Attribute) and node.func.attr == "backward"
            or isinstance(node.func, ast.Name) and node.func.id == "fit_verifier"
        ) for node in ast.walk(source))
        if prohibited or item.nodeid in TRAINER_TESTS:
            item.add_marker(pytest.mark.skip(reason="User forbids optimizer/backward/fitting in this task"))
            excluded.append(item.nodeid)
    config._inference_only_excluded = excluded


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    terminalreporter.write_line("Inference-only legacy exclusions: "
                               + str(len(config._inference_only_excluded)))
