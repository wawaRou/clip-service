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


def test_ack_deadline_is_separate_from_frame_retention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(storage.time, "time", lambda: 100.0)
    store = CandidateStore(tmp_path, retention_seconds=300)

    record = store.create("front_door", "episode-1", decision(), ack_timeout_seconds=20)

    assert record.ack_deadline_at == 120.0
    assert record.expires_at == 400.0
    assert record.to_api()["ack_deadline_at"] == 120.0
    assert CandidateStore(tmp_path, retention_seconds=300).get(record.candidate_id) == record

    monkeypatch.setattr(storage.time, "time", lambda: 130.0)
    expired = store.update_status(record.candidate_id, status="expired", triggered_vlm=None)
    assert expired is not None
    assert expired.status == "expired"
    assert store.get_frame(record.candidate_id, 0) == b"first"


def test_retention_hides_frames_but_preserves_pending_until_service_expires_it(
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
    assert store.cleanup() == 0
    assert (tmp_path / record.candidate_id).exists()
    expired = store.update_status(record.candidate_id, status="expired", triggered_vlm=None)
    assert expired is not None
    assert expired.status == "expired"
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


def test_startup_cancels_restored_pending_candidates_without_changing_acked_ones(
    tmp_path: Path,
) -> None:
    store = CandidateStore(tmp_path, retention_seconds=60)
    pending = store.create("front_door", "episode-1", decision())
    completed = store.create("living_room", "episode-2", decision())
    acknowledged = store.update_status(completed.candidate_id, status="acked", triggered_vlm=True)
    restarted = CandidateStore(tmp_path, retention_seconds=60)

    assert restarted.cancel_pending() == 1
    reopened = CandidateStore(tmp_path, retention_seconds=60)
    cancelled = reopened.get(pending.candidate_id)
    assert cancelled is not None
    assert cancelled.status == "cancelled"
    assert reopened.get_frame(pending.candidate_id, 0) == b"first"
    assert reopened.get(completed.candidate_id) == acknowledged
    assert reopened.cancel_pending() == 0


@pytest.mark.parametrize("error_type", [OSError, FileNotFoundError])
def test_failed_manifest_update_preserves_previous_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_type: type[OSError]
) -> None:
    store = CandidateStore(tmp_path, retention_seconds=60)
    record = store.create("front_door", "episode-1", decision())

    def fail_replace(self: Path, target: Path) -> Path:
        raise error_type("disk write failed")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_replace)
        with pytest.raises(OSError, match="disk write failed"):
            store.update_status(record.candidate_id, status="acked", triggered_vlm=True)

    assert store.get(record.candidate_id) == record
    assert CandidateStore(tmp_path, retention_seconds=60).get(record.candidate_id) == record
    acknowledged = store.update_status(record.candidate_id, status="acked", triggered_vlm=True)
    assert acknowledged is not None
    assert CandidateStore(tmp_path, retention_seconds=60).get(record.candidate_id) == acknowledged


def test_failed_startup_cancellation_can_be_retried_without_changing_failed_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = CandidateStore(tmp_path, retention_seconds=60)
    first = store.create("front_door", "episode-1", decision())
    second = store.create("living_room", "episode-2", decision())
    replace_path = Path.replace

    def fail_second_replace(self: Path, target: Path) -> Path:
        if self.parent.name == second.candidate_id:
            raise OSError("disk write failed")
        return replace_path(self, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_second_replace)
        with pytest.raises(OSError, match="disk write failed"):
            store.cancel_pending()

    cancelled = store.get(first.candidate_id)
    assert cancelled is not None
    assert cancelled.status == "cancelled"
    assert store.get(second.candidate_id) == second
    restarted = CandidateStore(tmp_path, retention_seconds=60)
    assert restarted.get(first.candidate_id) == cancelled
    assert restarted.get(second.candidate_id) == second
    assert store.cancel_pending() == 1
    assert CandidateStore(tmp_path, retention_seconds=60).cancel_pending() == 0


def test_cleanup_tolerates_candidate_directory_removed_externally(tmp_path: Path) -> None:
    store = CandidateStore(tmp_path, retention_seconds=60)
    record = store.create("front_door", "episode-1", decision())
    store.update_status(record.candidate_id, status="expired", triggered_vlm=None)
    storage.shutil.rmtree(tmp_path / record.candidate_id)

    assert store.cleanup(now=record.expires_at) == 1
    assert store.cleanup(now=record.expires_at) == 0


def test_startup_cancellation_clears_missing_candidates_and_continues(tmp_path: Path) -> None:
    store = CandidateStore(tmp_path, retention_seconds=60)
    removed = store.create("front_door", "episode-1", decision())
    surviving = store.create("living_room", "episode-2", decision())
    storage.shutil.rmtree(tmp_path / removed.candidate_id)

    assert store.cancel_pending() == 2

    assert store.get(removed.candidate_id) is None
    cancelled = store.get(surviving.candidate_id)
    assert cancelled is not None
    assert cancelled.status == "cancelled"
    assert CandidateStore(tmp_path, retention_seconds=60).get(surviving.candidate_id) == cancelled


def test_failed_cleanup_can_be_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = CandidateStore(tmp_path, retention_seconds=60)
    record = store.create("front_door", "episode-1", decision())
    store.update_status(record.candidate_id, status="expired", triggered_vlm=None)

    def fail_rmtree(path: Path) -> None:
        raise PermissionError("directory is busy")

    with monkeypatch.context() as patch:
        patch.setattr(storage.shutil, "rmtree", fail_rmtree)
        with pytest.raises(PermissionError, match="directory is busy"):
            store.cleanup(now=record.expires_at)

    assert store.cleanup(now=record.expires_at) == 1
    assert not (tmp_path / record.candidate_id).exists()
    assert store.cleanup(now=record.expires_at) == 0


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
