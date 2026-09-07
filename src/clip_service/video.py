"""Frigate video ingestion: one bounded reader and JPEG buffer per camera."""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

# Native FFmpeg diagnostics can contain credential-bearing stream URLs.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2  # noqa: E402

from .config import DetectionConfig  # noqa: E402
from .frame_buffer import EncodedFrame, FrameRingBuffer  # noqa: E402

LOGGER = logging.getLogger(__name__)


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
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._connected = False
        self._frames_received = 0
        self._last_error: str | None = None
        self._last_received_monotonic: float | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._read, name=f"video-{self.name}", daemon=True)
        self._thread.start()

    def request_stop(self) -> None:
        self._stop.set()

    def close(self, timeout: float | None = None) -> None:
        self.request_stop()
        if self._thread is not None:
            if timeout is None:
                timeout = (
                    self.settings.open_timeout_seconds + self.settings.read_timeout_seconds + 1
                )
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
            latest = self.ring.latest()
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

    def _read(self) -> None:
        sequence = 0
        next_buffer_at = 0.0
        while not self._stop.is_set():
            capture = None
            try:
                capture = self._capture_factory(self._url, self.settings)
                if not capture.isOpened():
                    raise OSError("unable to open Frigate stream")
                while not self._stop.is_set():
                    ok, pixels = capture.read()
                    if not ok:
                        raise OSError("Frigate stream stopped providing frames")
                    now = time.monotonic()
                    with self._lock:
                        self._connected = True
                        self._frames_received += 1
                        self._last_error = None
                    if now < next_buffer_at:
                        continue
                    ok, jpeg = cv2.imencode(
                        ".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, self.settings.jpeg_quality]
                    )
                    if not ok:
                        raise OSError("unable to encode camera frame")
                    sequence += 1
                    with self._lock:
                        self.ring.append(EncodedFrame(time.time(), jpeg.tobytes(), sequence))
                        self._last_received_monotonic = now
                    next_buffer_at = now + 1 / self.settings.ring_max_fps
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
                    self.ring.clear()
            self._stop.wait(self.settings.reconnect_delay_seconds)
