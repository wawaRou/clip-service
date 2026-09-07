from pathlib import Path

import pytest

from clip_service import storage
from clip_service.detector import CandidateDecision
from clip_service.frame_buffer import EncodedFrame
from clip_service.storage import CandidateStore


def decision() -> CandidateDecision:
    return CandidateDecision(
        started_at=10.0,
        confirmed_at=10.2,
        similarity=0.5,
        minimum_similarity=0.4,
        frames=(EncodedFrame(10.0, b"first"), EncodedFrame(10.2, b"second")),
    )


def test_published_candidate_exposes_its_original_frames(tmp_path: Path) -> None:
    store = CandidateStore(tmp_path, retention_seconds=60)

    record = store.create("front_door", "episode-1", decision())

    assert store.get(record.candidate_id) == record
    assert store.get_frame(record.candidate_id, 0) == b"first"
    assert store.get_frame(record.candidate_id, 1) == b"second"
    assert store.get_frame(record.candidate_id, 2) is None
    assert store.get_frame(record.candidate_id, -1) is None
    assert store.get_frame("missing", 0) is None
    assert record.to_api()["frames"] == [
        {
            "index": 0,
            "timestamp": 10.0,
            "url": f"/api/v1/candidates/{record.candidate_id}/frames/0",
        },
        {
            "index": 1,
            "timestamp": 10.2,
            "url": f"/api/v1/candidates/{record.candidate_id}/frames/1",
        },
    ]


def test_acknowledgement_and_frames_survive_reopening_store(tmp_path: Path) -> None:
    store = CandidateStore(tmp_path, retention_seconds=60)
    record = store.create("front_door", "episode-1", decision())

    acknowledged = store.update_status(record.candidate_id, status="acked", triggered_vlm=True)
    reopened = CandidateStore(tmp_path, retention_seconds=60)

    assert acknowledged is not None
    assert acknowledged.status == "acked"
    assert acknowledged.triggered_vlm is True
    assert record.status == "pending"
    assert reopened.get(record.candidate_id) == acknowledged
    assert reopened.get_frame(record.candidate_id, 1) == b"second"


def test_expiration_hides_candidate_and_cleans_only_owned_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(storage.time, "time", lambda: 100.0)
    store = CandidateStore(tmp_path, retention_seconds=60)
    record = store.create("front_door", "episode-1", decision())
    unrelated = tmp_path / "my-files"
    unrelated.mkdir()
    (unrelated / "keep.txt").write_text("keep", encoding="utf-8")

    monkeypatch.setattr(storage.time, "time", lambda: 160.0)

    assert store.get(record.candidate_id) is None
    assert store.get_frame(record.candidate_id, 0) is None
    assert store.update_status(record.candidate_id, status="acked", triggered_vlm=True) is None
    assert store.cleanup() == 1
    assert not (tmp_path / record.candidate_id).exists()
    assert (unrelated / "keep.txt").read_text("utf-8") == "keep"
    assert store.cleanup() == 0


def test_reopening_discards_interrupted_candidate_staging_only(tmp_path: Path) -> None:
    interrupted = tmp_path / f".candidate-{'a' * 32}.tmp"
    interrupted.mkdir()
    (interrupted / "frame-0.jpg").write_bytes(b"unfinished")
    unrelated = tmp_path / ".candidate-not-ours.tmp"
    unrelated.mkdir()

    CandidateStore(tmp_path, retention_seconds=60)

    assert not interrupted.exists()
    assert unrelated.exists()


def test_failed_manifest_update_preserves_previous_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = CandidateStore(tmp_path, retention_seconds=60)
    record = store.create("front_door", "episode-1", decision())

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("disk write failed")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_replace)
        with pytest.raises(OSError, match="disk write failed"):
            store.update_status(record.candidate_id, status="acked", triggered_vlm=True)

    assert store.get(record.candidate_id) == record
    assert CandidateStore(tmp_path, retention_seconds=60).get(record.candidate_id) == record


def test_failed_candidate_publish_leaves_no_visible_or_partial_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = CandidateStore(tmp_path, retention_seconds=60)

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("disk write failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="disk write failed"):
        store.create("front_door", "episode-1", decision())

    assert list(tmp_path.iterdir()) == []
