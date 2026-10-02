"""Shared test fixtures: isolate on-disk persistence to a temp dir so tests
never read or write the real ~/.modelshift/ files."""
import os
import tempfile

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolate_persistence():
    d = tempfile.mkdtemp(prefix="modelshift-test-")
    os.environ["MODELSHIFT_SETTINGS_PATH"] = os.path.join(d, "settings.json")
    os.environ["MODELSHIFT_RUNS_DIR"] = os.path.join(d, "runs")
    yield
