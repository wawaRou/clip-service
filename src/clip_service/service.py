"""Own the configured cameras and the service lifecycle."""

import time
from typing import Any

from .config import ServiceConfig
from .video import CameraReader, CaptureFactory, open_capture


class ClipService:
    def __init__(
        self, config: ServiceConfig, *, capture_factory: CaptureFactory = open_capture
    ) -> None:
        self.config = config
        self.started_at = time.monotonic()
        self.cameras = {
            name: CameraReader(
                name, config.camera_url(name), config.camera_settings(name), capture_factory
            )
            for name, camera in config.cameras.items()
            if camera.enabled
        }

    def start(self) -> None:
        for camera in self.cameras.values():
            camera.start()

    def close(self) -> None:
        for camera in self.cameras.values():
            camera.request_stop()
        # FFmpeg may serialize concurrent opens; include every in-flight open in the budget.
        timeout = sum(camera.settings.open_timeout_seconds for camera in self.cameras.values())
        timeout += (
            max((c.settings.read_timeout_seconds for c in self.cameras.values()), default=0) + 1
        )
        deadline = time.monotonic() + timeout
        for camera in self.cameras.values():
            camera.close(timeout=max(0, deadline - time.monotonic()))

    def health(self) -> dict[str, Any]:
        cameras = {name: camera.health() for name, camera in self.cameras.items()}
        return {
            "schema_version": "1.0",
            "status": "degraded" if any(camera["stale"] for camera in cameras.values()) else "ok",
            "uptime_seconds": time.monotonic() - self.started_at,
            "cameras": cameras,
        }
