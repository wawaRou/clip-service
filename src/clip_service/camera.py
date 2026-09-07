"""One camera's detection session and candidate lifecycle."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Any

from .detector import CandidateDetector
from .errors import ServiceError
from .events import EventBroker
from .frame_buffer import EncodedFrame
from .model import ImageEncoder
from .storage import CandidateRecord, CandidateStore
from .video import CameraReader


class Camera:
    def __init__(
        self,
        reader: CameraReader,
        encoder: ImageEncoder,
        store: CandidateStore,
        events: EventBroker,
    ) -> None:
        self.name = reader.name
        self.reader = reader
        self.settings = reader.settings
        self.encoder = encoder
        self.store = store
        self.events = events
        self.episode_id: str | None = None
        self._pending: CandidateRecord | None = None
        self.detector = CandidateDetector(
            similarity_threshold=self.settings.similarity_threshold,
            stable_seconds=self.settings.stable_seconds,
            frame_interval=self.settings.candidate_frame_interval,
            frame_count=self.settings.candidate_frame_count,
        )
        self._lock = threading.RLock()
        self._revision = 0
        self._last_sequence = 0
        self._stream_generation = 0
        self._inference_error: str | None = None
        self._storage_error: str | None = None
        self._measurement_started_at = time.monotonic()
        self._completed_at: deque[float] = deque()
        self._frames_processed = 0
        self._last_processing_seconds: float | None = None
        self._last_frame_latency_seconds: float | None = None

    @property
    def pending_candidate_id(self) -> str | None:
        return self._pending.candidate_id if self._pending else None

    def pending_candidate(self) -> CandidateRecord | None:
        with self._lock:
            self.expire_candidate()
            return self._pending

    def expire_candidate(self, now: float | None = None) -> None:
        """Release a timed-out candidate without changing the comparison baseline."""
        now = time.time() if now is None else now
        with self._lock:
            pending = self._pending
            if pending is None:
                return
            deadline = min(pending.ack_deadline_at or pending.expires_at, pending.expires_at)
            if now < deadline:
                return
            self.store.update_status(pending.candidate_id, status="expired", triggered_vlm=None)
            self._pending = None
            self.detector.acknowledge()
            self._revision += 1

    def start_episode(self, episode_id: str) -> None:
        if not episode_id:
            raise ServiceError("episode_id is required")
        with self._lock:
            if self.episode_id == episode_id:
                return
            if self.episode_id is not None:
                raise ServiceError("camera already has an active episode", 409, "episode_active")
            self.episode_id = episode_id
            self.detector.clear_baseline()
            self._revision += 1

    def stop_episode(self, episode_id: str) -> None:
        with self._lock:
            self._require_episode(episode_id)
            if self.pending_candidate_id is not None:
                self.store.update_status(
                    self.pending_candidate_id, status="cancelled", triggered_vlm=None
                )
            self.episode_id = None
            self._pending = None
            self.detector.clear_baseline()
            self._revision += 1

    def set_baseline(self, episode_id: str, jpeg: bytes) -> None:
        with self._lock:
            self._require_episode(episode_id)
            revision = self._revision
        embedding = self._encode(jpeg)
        with self._lock:
            self._require_revision(episode_id, revision)
            self.detector.set_baseline(embedding)
            self._revision += 1

    def latest_jpeg(self) -> bytes:
        frame = self.reader.latest_frame()
        if frame is None:
            raise ServiceError("camera has no fresh frame", 503, "frame_unavailable")
        return frame.jpeg

    def frame_window(
        self, episode_id: str, start: float, end: float, count: int
    ) -> list[EncodedFrame]:
        with self._lock:
            self._require_episode(episode_id)
        if not 1 <= count <= 9 or not math.isfinite(start) or not math.isfinite(end) or start > end:
            raise ServiceError("invalid frame window")
        targets = (
            [(start + end) / 2]
            if count == 1
            else [start + i * (end - start) / (count - 1) for i in range(count)]
        )
        frames = self.reader.ring.nearest_many(
            targets, max_distance=self.settings.frame_window_max_distance_seconds
        )
        if frames is None:
            raise ServiceError(
                "camera frame window is unavailable", 503, "frame_window_unavailable"
            )
        return frames

    def acknowledge(
        self,
        candidate_id: str,
        *,
        episode_id: str,
        triggered_vlm: bool,
        baseline_jpeg: bytes | None,
    ) -> CandidateRecord:
        with self._lock:
            completed = self._acknowledged(candidate_id, episode_id, triggered_vlm)
            if completed is not None:
                return completed
            self.expire_candidate()
            self._require_pending(episode_id, candidate_id)
            revision = self._revision
        embedding = self._encode(baseline_jpeg) if baseline_jpeg is not None else None
        with self._lock:
            completed = self._acknowledged(candidate_id, episode_id, triggered_vlm)
            if completed is not None:
                return completed
            self.expire_candidate()
            self._require_revision(episode_id, revision)
            self._require_pending(episode_id, candidate_id)
            record = self.store.update_status(
                candidate_id, status="acknowledged", triggered_vlm=triggered_vlm
            )
            if record is None:
                raise ServiceError("candidate not found", 404, "candidate_not_found")
            if embedding is not None:
                self.detector.set_baseline(embedding)
            self.detector.acknowledge()
            self._pending = None
            self._revision += 1
            return record

    def process_latest(self) -> None:
        with self._lock:
            if (
                self.episode_id is None
                or self.detector.baseline is None
                or self.pending_candidate_id
            ):
                return
            frame = self.reader.latest_frame()
            if frame is None:
                self.detector.reset_tracking()
                return
            if frame.stream_generation != self._stream_generation:
                self.detector.reset_tracking()
                self._stream_generation = frame.stream_generation
            if frame.sequence == self._last_sequence:
                return
            self._last_sequence = frame.sequence
            revision = self._revision
            episode_id = self.episode_id
        started = time.monotonic()
        embedding = self._encode(frame.jpeg)
        finished = time.monotonic()
        with self._lock:
            self._completed_at.append(finished)
            self._prune_measurements(finished)
            self._frames_processed += 1
            self._last_processing_seconds = finished - started
            self._last_frame_latency_seconds = finished - frame.received_monotonic
            latest = self.reader.latest_frame()
            if (
                self._revision != revision
                or latest is None
                or latest.stream_generation != frame.stream_generation
            ):
                return
            decision = self.detector.observe(frame.timestamp, embedding, self.reader.ring)
            if decision is None:
                return
            try:
                record = self.store.create(
                    self.name,
                    episode_id,
                    decision,
                    ack_timeout_seconds=self.settings.ack_timeout_seconds,
                )
            except OSError:
                self._storage_error = "candidate storage write failed"
                self.detector.acknowledge()
                raise
            self._storage_error = None
            self._pending = record
            self._revision += 1
            self.events.publish("candidate_change", record.to_api())

    def health(self) -> dict[str, Any]:
        with self._lock:
            status = self.reader.health()
            now = time.monotonic()
            self._prune_measurements(now)
            measurement_seconds = min(5.0, now - self._measurement_started_at)
            return status | {
                "inference_error": self._inference_error,
                "storage_error": self._storage_error,
                "inference": {
                    "target_fps": self.settings.inference_fps,
                    "actual_fps": len(self._completed_at) / measurement_seconds
                    if measurement_seconds > 0
                    else 0.0,
                    "measurement_seconds": measurement_seconds,
                    "frames_processed": self._frames_processed,
                    "last_processing_seconds": self._last_processing_seconds,
                    "last_frame_latency_seconds": self._last_frame_latency_seconds,
                },
                "episode": {
                    "active": self.episode_id is not None,
                    "episode_id": self.episode_id,
                    "baseline_ready": self.detector.baseline is not None,
                    "inference_active": self.episode_id is not None
                    and self.detector.baseline is not None
                    and self.pending_candidate_id is None
                    and not status["stale"],
                    "pending_candidate_id": self.pending_candidate_id,
                },
            }

    def _prune_measurements(self, now: float) -> None:
        while self._completed_at and self._completed_at[0] < now - 5:
            self._completed_at.popleft()

    def _encode(self, jpeg: bytes):
        try:
            embedding = self.encoder.encode_jpeg(jpeg)
        except (OSError, ValueError, RuntimeError, ImportError) as error:
            with self._lock:
                self._inference_error = f"CLIP encoding failed ({type(error).__name__})"
            raise ServiceError(self._inference_error, 503, "clip_unavailable") from error
        with self._lock:
            self._inference_error = None
        return embedding

    def _require_episode(self, episode_id: str) -> None:
        if self.episode_id != episode_id:
            raise ServiceError("episode is not active", 404, "episode_not_found")

    def _require_revision(self, episode_id: str, revision: int) -> None:
        self._require_episode(episode_id)
        if self._revision != revision:
            raise ServiceError(
                "camera state changed during encoding; retry request", 409, "state_changed"
            )

    def _require_pending(self, episode_id: str, candidate_id: str) -> None:
        self._require_episode(episode_id)
        if self.pending_candidate_id != candidate_id:
            raise ServiceError("candidate is not pending", 409, "candidate_not_pending")

    def _acknowledged(
        self, candidate_id: str, episode_id: str, triggered_vlm: bool
    ) -> CandidateRecord | None:
        record = self.store.get(candidate_id)
        if record is None:
            raise ServiceError("candidate not found", 404, "candidate_not_found")
        if record.status != "acknowledged":
            return None
        if record.episode_id != episode_id or record.triggered_vlm != triggered_vlm:
            raise ServiceError(
                "acknowledgement conflicts with the completed result", 409, "ack_conflict"
            )
        return record
