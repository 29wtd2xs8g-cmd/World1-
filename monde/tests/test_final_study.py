"""Guards of the final study: the test seeds open once, a relaunch resumes the same study, nothing partial is read."""
import json

import pytest

from toyworld import seeds
from toyworld.compare import D10_MARKER, require_complete, require_d10_recorded, study_purpose


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(seeds, "TEST_SEED_LEDGER", path)
    return path


def lines(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_first_call_opens_and_records_the_purpose(ledger):
    assert seeds.test_seeds_for_study("study A") == seeds.TEST_SEEDS
    assert [e["purpose"] for e in lines(ledger)] == ["study A"]


def test_a_relaunch_of_the_same_study_resumes_without_a_second_line(ledger):
    seeds.test_seeds_for_study("study A")
    assert seeds.test_seeds_for_study("study A") == seeds.TEST_SEEDS
    assert seeds.resume_test_seeds("study A") == seeds.TEST_SEEDS
    assert len(lines(ledger)) == 1


def test_another_purpose_cannot_use_the_test_seeds(ledger):
    seeds.test_seeds_for_study("study A")
    with pytest.raises(PermissionError):
        seeds.test_seeds_for_study("study B")
    with pytest.raises(PermissionError):
        seeds.resume_test_seeds("study B")
    assert len(lines(ledger)) == 1


def test_resume_needs_an_opened_ledger(ledger):
    with pytest.raises(PermissionError):
        seeds.resume_test_seeds("study A")
    assert not ledger.exists()                       # resuming never writes


def test_open_test_seeds_stays_one_shot(ledger):
    seeds.open_test_seeds("study A")
    with pytest.raises(PermissionError):
        seeds.open_test_seeds("study A")


def test_purpose_names_the_model_thresholds_and_reference(tmp_path):
    m, t, r = tmp_path / "m.pt", tmp_path / "t.json", tmp_path / "r.json"
    for p, x in ((m, b"model"), (t, b'{"a": 1}'), (r, b'{"b": 2}')):
        p.write_bytes(x)
    base = study_purpose(m, t, r)
    assert base == study_purpose(m, t, r)
    t.write_bytes(b'{"a": 2}')
    assert study_purpose(m, t, r) != base            # a changed threshold cannot resume the study
    t.write_bytes(b'{"a": 1}')
    m.write_bytes(b"other model")
    assert study_purpose(m, t, r) != base
    m.write_bytes(b"model")
    r.write_bytes(b'{"b": 3}')
    assert study_purpose(m, t, r) != base


def test_the_study_waits_for_d10(tmp_path):
    p = tmp_path / "research.md"
    with pytest.raises(RuntimeError):
        require_d10_recorded(p)
    p.write_text("D9 only\n")
    with pytest.raises(RuntimeError):
        require_d10_recorded(p)
    p.write_text(f"{D10_MARKER} (2026-11-23).**\n")
    require_d10_recorded(p)


def test_no_result_is_printed_before_every_seed_is_complete():
    require_complete(list(range(60)), 60)
    with pytest.raises(RuntimeError):
        require_complete(list(range(59)), 60)
    with pytest.raises(RuntimeError):
        require_complete([], 60)


def test_the_real_research_file_records_d10():
    from toyworld.compare import RESEARCH
    require_d10_recorded(RESEARCH)
