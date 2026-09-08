from __future__ import annotations

import bisect
import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EncodedFrame:
    timestamp: float
    jpeg: bytes
    sequence: int = 0
    stream_generation: int = 0
    received_monotonic: float = field(default_factory=time.monotonic)


class FrameRingBuffer:
    """Thread-safe time-bounded JPEG frame buffer."""

    def __init__(self, duration_seconds: float, max_fps: float = 30.0) -> None:
        self._duration = duration_seconds
        self._frames: deque[EncodedFrame] = deque(
            maxlen=max(2, int(duration_seconds * max_fps) + 2)
        )
        self._lock = threading.RLock()

    def append(self, frame: EncodedFrame) -> None:
        with self._lock:
            self._frames.append(frame)
            cutoff = frame.timestamp - self._duration
            while self._frames and self._frames[0].timestamp < cutoff:
                self._frames.popleft()

    def update_settings(self, duration_seconds: float, max_fps: float) -> None:
        """Keep recent history within the new time and frame-count limits."""
        with self._lock:
            self._duration = duration_seconds
            self._frames = deque(self._frames, maxlen=max(2, int(duration_seconds * max_fps) + 2))
            if self._frames:
                cutoff = self._frames[-1].timestamp - duration_seconds
                while self._frames and self._frames[0].timestamp < cutoff:
                    self._frames.popleft()

    def latest(self) -> EncodedFrame | None:
        with self._lock:
            return self._frames[-1] if self._frames else None

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()

    def snapshot(self) -> tuple[EncodedFrame, ...]:
        with self._lock:
            return tuple(self._frames)

    def nearest_many(
        self,
        timestamps: list[float],
        *,
        max_distance: float | None = None,
        window: tuple[float, float] | None = None,
    ) -> list[EncodedFrame] | None:
        with self._lock:
            frames = list(self._frames)
        if window is not None:
            frames = [frame for frame in frames if window[0] <= frame.timestamp <= window[1]]
        if not frames:
            return None
        frame_times = [frame.timestamp for frame in frames]
        selected: list[EncodedFrame] = []
        for target in timestamps:
            index = bisect.bisect_left(frame_times, target)
            choices = []
            if index < len(frames):
                choices.append(frames[index])
            if index:
                choices.append(frames[index - 1])
            chosen = min(choices, key=lambda frame: abs(frame.timestamp - target))
            if max_distance is not None and abs(chosen.timestamp - target) > max_distance:
                return None
            selected.append(chosen)
        return selected

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)
