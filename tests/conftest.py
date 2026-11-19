"""Shared test setup: keep every test off the real ~/.modelshift/ files.

This must happen when conftest is imported, NOT in a fixture: test modules import modelshift.config
at collection time, which loads the saved settings, and fixtures only run after collection. A
fixture-based version let a developer's own saved settings (e.g. a changed quality gate) leak into
tests that assume the defaults."""
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="modelshift-test-")
os.environ["MODELSHIFT_SETTINGS_PATH"] = os.path.join(_TMP, "settings.json")
os.environ["MODELSHIFT_RUNS_DIR"] = os.path.join(_TMP, "runs")
