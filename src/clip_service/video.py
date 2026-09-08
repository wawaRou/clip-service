"""Frigate video ingestion: one bounded reader and JPEG buffer per camera."""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

# Native FFmpeg diagnostics can contain credential-bearing stream URLs.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2  # noqa: E402

from .config import DetectionConfig  # noqa: E402
from .frame_buffer import EncodedFrame, FrameRingBuffer  # noqa: E402

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecodedFrame:
    timestamp: float
    pixels: Any  # OpenCV BGR pixels; retained only for the latest frame.
    sequence: int
    stream_generation: int
    received_monotonic: float
    jpeg_quality: int

    @property
    def jpeg(self) -> bytes:
        """Encode only for history sampling or an explicit latest-image request."""
        ok, jpeg = cv2.imencode(".jpg", self.pixels, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise OSError("unable to encode camera frame")
        return jpeg.tobytes()


class Capture(Protocol):
    """The external video decoder boundary, also used by camera test feeds."""

    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, Any]: ...
    def release(self) -> None: ...


CaptureFactory = Callable[[str, DetectionConfig], Capture]


def open_capture(url: str, settings: DetectionConfig) -> Capture:
    return cv2.VideoCapture(
        url,
        cv2.CAP_FFMPEG,
        [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            math.ceil(settings.open_timeout_seconds * 1000),
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            math.ceil(settings.read_timeout_seconds * 1000),
        ],
    )


class CameraReader:
    def __init__(
        self,
        name: str,
        url: str,
        settings: DetectionConfig,
        capture_factory: CaptureFactory = open_capture,
    ) -> None:
        self.name = name
        self.settings = settings
        self.ring = FrameRingBuffer(settings.ring_seconds, settings.ring_max_fps)
        self._url = url
        self._capture_factory = capture_factory
        self._stop = threading.Event()
        self._wake_reconnect = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._connected = False
        self._frames_received = 0
        self._last_error: str | None = None
        self._last_received_monotonic: float | None = None
        self._latest: DecodedFrame | None = None
        self._capture_settings: DetectionConfig | None = None

    @property
    def source_url(self) -> str:
        return self._url

    @property
    def open_timeout_budget(self) -> float:
        """Include the timeout already given to a decoder that is still open."""
        with self._lock:
            return (self._capture_settings or self.settings).open_timeout_seconds

    @property
    def read_timeout_budget(self) -> float:
        with self._lock:
            return (self._capture_settings or self.settings).read_timeout_seconds

    def update_settings(self, settings: DetectionConfig) -> None:
        """Apply live settings; decoder timeouts apply on its next connection."""
        with self._lock:
            self.ring.update_settings(settings.ring_seconds, settings.ring_max_fps)
            if settings.reconnect_delay_seconds != self.settings.reconnect_delay_seconds:
                self._wake_reconnect.set()
            self.settings = settings

    def start(self) -> None:
        self._thread = threading.Thread(target=self._read, name=f"video-{self.name}", daemon=True)
        self._thread.start()

    def request_stop(self) -> None:
        self._stop.set()
        self._wake_reconnect.set()

    def close(self, timeout: float | None = None) -> None:
        self.request_stop()
        if self._thread is not None:
            if timeout is None:
                timeout = self.open_timeout_budget + self.read_timeout_budget + 1
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError(
                    f"camera {self.name}: video decoder did not stop within {timeout}s"
                )

    def health(self) -> dict[str, Any]:
        with self._lock:
            age = (
                None
                if self._last_received_monotonic is None
                else max(0.0, time.monotonic() - self._last_received_monotonic)
            )
            latest = self._latest
            return {
                "connected": self._connected,
                "frames_received": self._frames_received,
                "buffered_frames": len(self.ring),
                "last_frame_at": latest.timestamp if latest else None,
                "frame_age_seconds": age,
                "stale": not self._connected
                or age is None
                or age > self.settings.frame_max_age_seconds,
                "last_error": self._last_error,
            }

    def latest_frame(self) -> DecodedFrame | None:
        with self._lock:
            if (
                not self._connected
                or self._last_received_monotonic is None
                or time.monotonic() - self._last_received_monotonic
                > self.settings.frame_max_age_seconds
            ):
                return None
            return self._latest

    def _read(self) -> None:
        sequence = 0
        generation = 0
        while not self._stop.is_set():
            capture = None
            try:
                with self._lock:
                    self._capture_settings = self.settings
                    capture_settings = self._capture_settings
                capture = self._capture_factory(self._url, capture_settings)
                if not capture.isOpened():
                    raise OSError("unable to open Frigate stream")
                generation += 1
                buffer_started_at = time.monotonic()
                next_buffer_at = buffer_started_at
                buffer_rate = self.settings.ring_max_fps
                while not self._stop.is_set():
                    ok, pixels = capture.read()
                    if not ok:
                        raise OSError("Frigate stream stopped providing frames")
                    now = time.monotonic()
                    received_at = time.time()
                    with self._lock:
                        self._connected = True
                        self._frames_received += 1
                        self._last_error = None
                    sequence += 1
                    with self._lock:
                        frame = DecodedFrame(
                            received_at,
                            pixels,
                            sequence,
                            generation,
                            now,
                            self.settings.jpeg_quality,
                        )
                        self._latest = frame
                        self._last_received_monotonic = now
                        if buffer_rate != self.settings.ring_max_fps:
                            buffer_rate = self.settings.ring_max_fps
                            buffer_started_at = next_buffer_at = now
                        if now >= next_buffer_at:
                            self.ring.append(
                                EncodedFrame(received_at, frame.jpeg, sequence, generation, now)
                            )
                            sample = math.floor((now - buffer_started_at) * buffer_rate) + 1
                            next_buffer_at = buffer_started_at + sample / buffer_rate
            except (OSError, cv2.error):
                # Decoder exception text may contain authentication data.
                with self._lock:
                    self._last_error = "video unavailable; reconnecting"
                LOGGER.warning("camera %s: video unavailable; reconnecting", self.name)
            finally:
                if capture is not None:
                    capture.release()
                with self._lock:
                    self._connected = False
                    self._last_received_monotonic = None
                    self._latest = None
                    self._capture_settings = None
                    self.ring.clear()
            while not self._stop.is_set():
                with self._lock:
                    self._wake_reconnect.clear()
                    reconnect_delay = self.settings.reconnect_delay_seconds
                if self._stop.is_set() or not self._wake_reconnect.wait(reconnect_delay):
                    break
