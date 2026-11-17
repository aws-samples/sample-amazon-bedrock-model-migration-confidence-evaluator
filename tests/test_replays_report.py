"""PDF with replays: consistency section only when replays were on; D7 row in the dimension table."""
import io

from modelshift import api as api_mod
from modelshift.report import build_run_report
from modelshift.report_pdf import render_pdf
from test_reasoning_efforts import _client
from test_replays import BODY, _run, _varying


def _text(pdf: bytes) -> str:
    from pypdf import PdfReader
    return " ".join(" ".join((p.extract_text() or "").split()) for p in PdfReader(io.BytesIO(pdf)).pages)


def _report(tmp_path, monkeypatch, replays):
    c, rid = _client(tmp_path, monkeypatch)
    transport, _ = _varying(["q", "an entirely different reply"])
    api_mod.configure_transports(candidate_transport=transport, judge_transport=None)
    _run(c, rid, {**BODY, "replays": replays})
    return build_run_report(api_mod._RUNS[rid])


def test_json_report_carries_replays_and_d7(tmp_path, monkeypatch):
    rep = _report(tmp_path, monkeypatch, 2)
    c = rep["candidates"][0]
    assert rep["replays"] == 2 and c["replays"] == 2 and c["consistency"] is not None
    assert c["least_consistent"] and c["least_consistent"][0]["labels"]
    rows = c["evaluations"]["rows"]
    assert [r["replay_index"] for r in rows] == [0, 1] and rows[0]["consistency"] == c["least_consistent"][0]["d7"]


def test_pdf_consistency_section_only_with_replays(tmp_path, monkeypatch):
    on = _text(render_pdf(_report(tmp_path / "a", monkeypatch, 2)))
    assert "Consistency across replays" in on and "D7 Consistency" in on and "replay 2 of 2" in on
    off = _text(render_pdf(_report(tmp_path / "b", monkeypatch, 1)))
    assert "Consistency across replays" not in off and "replay 1 of" not in off
