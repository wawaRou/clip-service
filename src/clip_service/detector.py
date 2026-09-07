from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .frame_buffer import EncodedFrame, FrameRingBuffer


@dataclass(frozen=True)
class CandidateDecision:
    started_at: float
    confirmed_at: float
    similarity: float
    minimum_similarity: float
    frames: tuple[EncodedFrame, ...]


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        raise ValueError("embedding must have non-zero norm")
    return float(np.dot(left, right) / denominator)


class CandidateDetector:
    """Detects sustained visual divergence from a fixed baseline."""

    def __init__(
        self,
        *,
        similarity_threshold: float,
        stable_seconds: float,
        frame_interval: float,
        frame_count: int,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.stable_seconds = stable_seconds
        self.frame_interval = frame_interval
        self.frame_count = frame_count
        self.baseline: np.ndarray | None = None
        self.pending = False
        self._run_started_at: float | None = None
        self._minimum_similarity = 1.0

    def set_baseline(self, embedding: Sequence[float] | np.ndarray) -> None:
        value = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(value))
        if norm == 0.0:
            raise ValueError("baseline embedding must have non-zero norm")
        self.baseline = value / norm
        self.reset_tracking()

    def clear_baseline(self) -> None:
        self.baseline = None
        self.pending = False
        self.reset_tracking()

    def reset_tracking(self) -> None:
        self._run_started_at = None
        self._minimum_similarity = 1.0

    def acknowledge(self) -> None:
        self.pending = False
        self.reset_tracking()

    def observe(
        self,
        timestamp: float,
        embedding: np.ndarray,
        ring: FrameRingBuffer,
    ) -> CandidateDecision | None:
        if self.baseline is None or self.pending:
            return None
        similarity = cosine_similarity(self.baseline, embedding)
        if similarity >= self.similarity_threshold:
            self.reset_tracking()
            return None
        if self._run_started_at is None:
            self._run_started_at = timestamp
            self._minimum_similarity = similarity
            return None
        self._minimum_similarity = min(self._minimum_similarity, similarity)
        if timestamp - self._run_started_at < self.stable_seconds:
            return None

        targets = [
            self._run_started_at + index * self.frame_interval for index in range(self.frame_count)
        ]
        tolerance = max(self.frame_interval / 2.0, 0.075)
        if timestamp < targets[-1]:
            return None
        frames = ring.nearest_many(targets, max_distance=tolerance)
        if frames is None or len({frame.timestamp for frame in frames}) != self.frame_count:
            # This historical window can no longer be completed after a gap or eviction.
            if timestamp >= targets[-1] + tolerance:
                self._run_started_at = timestamp
                self._minimum_similarity = similarity
            return None
        self.pending = True
        return CandidateDecision(
            started_at=self._run_started_at,
            confirmed_at=timestamp,
            similarity=similarity,
            minimum_similarity=self._minimum_similarity,
            frames=tuple(frames),
        )
