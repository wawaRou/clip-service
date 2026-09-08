from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import ServiceError

if TYPE_CHECKING:
    from .detector import CandidateDecision


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    camera: str
    episode_id: str
    status: str
    created_at: float
    expires_at: float
    started_at: float
    confirmed_at: float
    similarity: float
    minimum_similarity: float
    frame_timestamps: tuple[float, ...]
    triggered_vlm: bool | None = None
    ack_deadline_at: float | None = None

    def to_api(self) -> dict:
        result = asdict(self)
        result["frames"] = [
            {
                "index": index,
                "timestamp": timestamp,
                "url": f"/api/v1/candidates/{self.candidate_id}/frames/{index}",
            }
            for index, timestamp in enumerate(self.frame_timestamps)
        ]
        del result["frame_timestamps"]
        return result


class CandidateStore:
    """Publish a candidate only after its manifest and all frames are on disk."""

    def __init__(self, root: Path, retention_seconds: float) -> None:
        self.root = root
        self.retention_seconds = retention_seconds
        self._records: dict[str, CandidateRecord] = {}
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._load_existing()

    def _load_existing(self) -> None:
        for directory in self.root.iterdir():
            if directory.is_symlink() or not directory.is_dir():
                continue
            if re.fullmatch(r"\.candidate-[0-9a-f]{32}\.tmp", directory.name):
                shutil.rmtree(directory)
                continue
            if not re.fullmatch(r"[0-9a-f]{32}", directory.name):
                continue
            try:
                data = json.loads((directory / "manifest.json").read_text("utf-8"))
                data["frame_timestamps"] = tuple(data["frame_timestamps"])
                record = CandidateRecord(**data)
            except (OSError, ValueError, TypeError, KeyError):
                continue
            if record.candidate_id == directory.name:
                self._records[record.candidate_id] = record
        self.cleanup()

    def create(
        self,
        camera: str,
        episode_id: str,
        decision: CandidateDecision,
        *,
        ack_timeout_seconds: float = 60,
    ) -> CandidateRecord:
        candidate_id = uuid.uuid4().hex
        now = time.time()
        record = CandidateRecord(
            candidate_id=candidate_id,
            camera=camera,
            episode_id=episode_id,
            status="pending",
            created_at=now,
            expires_at=now + self.retention_seconds,
            ack_deadline_at=now + ack_timeout_seconds,
            started_at=decision.started_at,
            confirmed_at=decision.confirmed_at,
            similarity=decision.similarity,
            minimum_similarity=decision.minimum_similarity,
            frame_timestamps=tuple(frame.timestamp for frame in decision.frames),
        )
        with self._lock:
            candidate_dir = self.root / candidate_id
            temporary_dir = self.root / f".candidate-{candidate_id}.tmp"
            temporary_dir.mkdir(mode=0o700)
            try:
                for index, frame in enumerate(decision.frames):
                    descriptor = os.open(
                        temporary_dir / f"frame-{index}.jpg",
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                    )
                    with os.fdopen(descriptor, "wb") as output:
                        output.write(frame.jpeg)
                self._write_manifest(record, temporary_dir)
                temporary_dir.replace(candidate_dir)
            except Exception:
                shutil.rmtree(temporary_dir)
                raise
            self._records[candidate_id] = record
        return record

    def get(self, candidate_id: str) -> CandidateRecord | None:
        with self._lock:
            record = self._records.get(candidate_id)
            if record is None or record.expires_at <= time.time():
                return None
            return record

    def get_frame(self, candidate_id: str, index: int) -> bytes | None:
        with self._lock:
            record = self.get(candidate_id)
            if record is None or index < 0 or index >= len(record.frame_timestamps):
                return None
            try:
                return (self.root / candidate_id / f"frame-{index}.jpg").read_bytes()
            except FileNotFoundError:
                return None

    def get_acknowledged(
        self, candidate_id: str, episode_id: str, triggered_vlm: bool
    ) -> CandidateRecord | None:
        """Resolve retries from persisted state, even when the camera no longer exists."""
        record = self.get(candidate_id)
        if record is None:
            raise ServiceError("candidate not found", 404, "candidate_not_found")
        if record.status != "acknowledged":
            return None
        if record.episode_id != episode_id or record.triggered_vlm != triggered_vlm:
            raise ServiceError(
                "acknowledgement conflicts with the completed result", 409, "ack_conflict"
            )
        return record

    def update_status(
        self, candidate_id: str, *, status: str, triggered_vlm: bool | None
    ) -> CandidateRecord | None:
        with self._lock:
            record = self._records.get(candidate_id)
            if record is None:
                return None
            updated = replace(record, status=status, triggered_vlm=triggered_vlm)
            candidate_dir = self.root / candidate_id
            try:
                self._write_manifest(updated, candidate_dir)
            except FileNotFoundError:
                if candidate_dir.exists():
                    raise
                del self._records[candidate_id]
                return None
            self._records[candidate_id] = updated
            return updated

    def _write_manifest(self, record: CandidateRecord, candidate_dir: Path) -> None:
        temporary = candidate_dir / "manifest.tmp"
        temporary.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(candidate_dir / "manifest.json")

    def cancel_pending(self) -> int:
        """Cancel candidates whose in-memory episodes did not survive a restart."""
        with self._lock:
            cancelled = 0
            for candidate_id, record in list(self._records.items()):
                if record.status == "pending":
                    self.update_status(candidate_id, status="cancelled", triggered_vlm=None)
                    cancelled += 1
            return cancelled

    def cleanup(self, now: float | None = None) -> int:
        now = time.time() if now is None else now
        with self._lock:
            expired = [
                candidate_id
                for candidate_id, record in self._records.items()
                if record.expires_at <= now and record.status != "pending"
            ]
            for candidate_id in expired:
                try:
                    shutil.rmtree(self.root / candidate_id)
                except FileNotFoundError:
                    pass
                del self._records[candidate_id]
            return len(expired)
