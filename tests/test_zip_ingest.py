"""Zip ingestion: a folder tree of log files consolidated into ONE run."""
import io
import json
import zipfile
from pathlib import Path

import pytest

from modelshift.ingest import ingest
from modelshift.schemas import Shape
from modelshift.sources import ZipArchiveError, is_zip, load_zip_bytes

FIXTURES = Path(__file__).parent / "fixtures"


def _capture(request_id: str, n: int) -> bytes:
    row = json.loads((FIXTURES / "prompt_capture_single.json").read_bytes())
    row.update(request_id=request_id, user_prompt=f"{row['user_prompt']}\ncall {n}")
    return json.dumps(row, indent=2).encode()  # pretty-printed, like the real captures


def _zip(entries) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries:
            zf.writestr(name, data)
    return buf.getvalue()


def _tree_zip(extra=()):
    """root/<request_id>/<call>.json with Windows '\\' separators, as built on Windows."""
    entries = []
    for rid, calls in (("REQ-A", 3), ("REQ-B", 1), ("REQ-C", 2)):
        for i in range(calls):
            entries.append((f"export\\{rid}\\2026090{i}_call.json", _capture(rid, i)))
    return _zip(entries + list(extra))


def test_is_zip_sniffs_magic_bytes():
    assert is_zip(_tree_zip())
    assert not is_zip(b'{"a": 1}')


def test_zip_tree_consolidates_into_one_run_with_unique_ids():
    objects, skipped = load_zip_bytes(_tree_zip(), "upload:t.zip")
    assert skipped == 0 and len(objects) == 6
    # backslashes normalised; member path kept in the object uri
    assert all("\\" not in uri for _, uri in objects)
    assert objects[0][1].startswith("upload:t.zip!export/REQ-A/")
    res = ingest(objects, skipped_objects=skipped)
    assert res.report.found == 6 and res.report.evaluable == 6
    assert res.report.reconciles()
    ids = [s.id for s in res.samples]
    assert len(set(ids)) == 6  # no collisions even though request_ids repeat
    assert sorted(ids) == sorted(["REQ-A", "REQ-A#2", "REQ-A#3", "REQ-B", "REQ-C", "REQ-C#2"])
    dup = next(s for s in res.samples if s.id == "REQ-A#2")
    assert dup.tags["request_id"] == "REQ-A"  # original grouping key preserved
    assert {s.source.shape for s in res.samples} == {Shape.PROMPT_CAPTURE}


def test_zip_skips_non_logs_counts_them_and_ignores_os_junk():
    extra = [
        ("export/README.txt", b"notes"),               # non-log -> skipped + counted
        ("export/inner.zip", _zip([("x.json", b"{}")])),  # nested zip -> skipped, not expanded
        ("__MACOSX/export/._a.json", b"junk"),          # OS noise -> ignored entirely
        ("export/.DS_Store", b"junk"),
    ]
    objects, skipped = load_zip_bytes(_tree_zip(extra), "z")
    assert len(objects) == 6 and skipped == 2
    res = ingest(objects, skipped_objects=skipped)
    assert res.report.found == 8 and res.report.skipped_objects == 2
    assert res.report.reconciles()
    assert res.report.shape_mix == {Shape.PROMPT_CAPTURE: 1.0}  # ratios stay per-row


def test_zip_mixed_formats_in_one_archive():
    jsonl = "\n".join(json.dumps(json.loads(_capture(f"L{i}", i))) for i in range(2)).encode()
    objects, _ = load_zip_bytes(_zip([
        ("a/one.json", _capture("P1", 0)),
        ("b/many.jsonl", jsonl),
        ("c/spend.json", (FIXTURES / "litellm_spendlogs.json").read_bytes()),
    ]), "z")
    res = ingest(objects)
    assert res.report.reconciles()
    assert res.report.evaluable >= 3 + 2  # 1 capture + 2 jsonl + spendlogs evaluable rows


def test_zip_rejects_bad_archive():
    with pytest.raises(ZipArchiveError):
        load_zip_bytes(b"PK\x03\x04 not really a zip", "z")


def test_zip_entry_limit(monkeypatch):
    monkeypatch.setenv("MODELSHIFT_ZIP_MAX_ENTRIES", "3")
    with pytest.raises(ZipArchiveError, match="limit is 3"):
        load_zip_bytes(_tree_zip(), "z")


def test_zip_uncompressed_size_limit(monkeypatch):
    monkeypatch.setenv("MODELSHIFT_ZIP_MAX_UNCOMPRESSED_MB", "1")
    bomb = _zip([("big.json", b"0" * (2 * 1024 * 1024))])  # compresses tiny, expands 2 MB
    with pytest.raises(ZipArchiveError, match="limit is 1 MB"):
        load_zip_bytes(bomb, "z")


def test_api_zip_upload_ingests_as_one_run():
    from fastapi.testclient import TestClient
    from modelshift import api as api_mod

    c = TestClient(api_mod.app)
    run_id = c.post("/api/v1/runs", json={"name": "zip"}).json()["run_id"]
    up = c.post("/api/v1/uploads",
                files={"file": ("logs.zip", _tree_zip([("export/notes.txt", b"x")]), "application/zip")}).json()
    r = c.post(f"/api/v1/runs/{run_id}/ingest", json={"upload_id": up["upload_id"]})
    assert r.status_code == 202
    ing = r.json()["ingestion"]
    assert ing["found"] == 7 and ing["evaluable"] == 6 and ing["skipped_objects"] == 1


def test_api_bad_zip_is_400():
    from fastapi.testclient import TestClient
    from modelshift import api as api_mod

    c = TestClient(api_mod.app)
    run_id = c.post("/api/v1/runs", json={"name": "badzip"}).json()["run_id"]
    up = c.post("/api/v1/uploads", files={"file": ("x.zip", b"PK\x03\x04garbage", "application/zip")}).json()
    r = c.post(f"/api/v1/runs/{run_id}/ingest", json={"upload_id": up["upload_id"]})
    assert r.status_code == 400
    assert r.json()["detail"]["error"]["code"] == "invalid_zip"
