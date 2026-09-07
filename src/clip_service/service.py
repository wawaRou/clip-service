"""Own the configured cameras and the service lifecycle."""

import logging
import threading
import time
from typing import Any

from .camera import Camera
from .config import ServiceConfig
from .errors import ServiceError
from .events import EventBroker
from .model import ClipEncoder, ImageEncoder
from .scheduler import FrameScheduler
from .storage import CandidateStore
from .video import CameraReader, CaptureFactory, open_capture


class ClipService:
    def __init__(
        self,
        config: ServiceConfig,
        *,
        capture_factory: CaptureFactory = open_capture,
        encoder: ImageEncoder | None = None,
    ) -> None:
        self.config = config
        self.started_at = time.monotonic()
        self.encoder = encoder or ClipEncoder(config.model.path, config.model.device)
        self.store = CandidateStore(config.data_dir / "candidates", config.retention_hours * 3600)
        self.store.cancel_pending()
        self.store.cleanup()
        self.events = EventBroker()
        self._stop = threading.Event()
        self._inference: threading.Thread | None = None
        self._next_cleanup_at = time.monotonic()
        self._maintenance_error: str | None = None
        self.cameras = {
            name: Camera(
                CameraReader(
                    name, config.camera_url(name), config.camera_settings(name), capture_factory
                ),
                self.encoder,
                self.store,
                self.events,
            )
            for name, camera in config.cameras.items()
            if camera.enabled
        }
        self._scheduler = FrameScheduler(
            {name: camera.settings.inference_fps for name, camera in self.cameras.items()},
            now=time.monotonic(),
        )

    def start(self) -> None:
        for camera in self.cameras.values():
            camera.reader.start()
        self._inference = threading.Thread(target=self._detect, name="clip-inference", daemon=True)
        self._inference.start()

    def close(self) -> None:
        self._stop.set()
        self.events.close()
        for camera in self.cameras.values():
            camera.reader.request_stop()
        # FFmpeg may serialize concurrent opens; include every in-flight open in the budget.
        timeout = sum(camera.settings.open_timeout_seconds for camera in self.cameras.values())
        timeout += (
            max((c.settings.read_timeout_seconds for c in self.cameras.values()), default=0) + 1
        )
        deadline = time.monotonic() + timeout
        for camera in self.cameras.values():
            camera.reader.close(timeout=max(0, deadline - time.monotonic()))
        if self._inference is not None:
            self._inference.join(timeout=10)
            if self._inference.is_alive():
                raise TimeoutError("CLIP inference did not finish during shutdown")

    def camera(self, name: str) -> Camera:
        try:
            return self.cameras[name]
        except KeyError:
            raise ServiceError("camera not found", 404, "camera_not_found") from None

    def _detect(self) -> None:
        while not self._stop.is_set():
            self._maintain_candidates()
            for name in self._scheduler.due(time.monotonic()):
                if self._stop.is_set():
                    return
                camera = self.cameras[name]
                try:
                    camera.process_latest()
                except (ServiceError, OSError) as error:
                    logging.getLogger(__name__).warning("camera %s: %s", camera.name, error)
                finally:
                    self._scheduler.complete(name, time.monotonic())
            self._stop.wait(min(0.1, self._scheduler.delay(time.monotonic())))

    def pending_candidates(self) -> list[dict[str, Any]]:
        candidates = (camera.pending_candidate() for camera in self.cameras.values())
        return [candidate.to_api() for candidate in candidates if candidate is not None]

    def _maintain_candidates(self) -> None:
        failed = False
        now = time.time()
        for camera in self.cameras.values():
            try:
                camera.expire_candidate(now)
            except OSError:
                failed = True
                logging.getLogger(__name__).exception(
                    "camera %s: candidate expiry failed", camera.name
                )
        try:
            if time.monotonic() >= self._next_cleanup_at:
                self.store.cleanup()
                self._next_cleanup_at = time.monotonic() + 1
        except OSError:
            failed = True
            logging.getLogger(__name__).exception("candidate cleanup failed")
        self._maintenance_error = "candidate storage update failed; retrying" if failed else None

    def health(self) -> dict[str, Any]:
        cameras = {name: camera.health() for name, camera in self.cameras.items()}
        return {
            "schema_version": "1.0",
            "status": "degraded"
            if self._maintenance_error
            or any(
                camera["stale"] or camera["inference_error"] or camera["storage_error"]
                for camera in cameras.values()
            )
            else "ok",
            "uptime_seconds": time.monotonic() - self.started_at,
            "maintenance_error": self._maintenance_error,
            "ring_seconds": self.config.detection.ring_seconds,
            "frame_window_max_distance_seconds": (
                self.config.detection.frame_window_max_distance_seconds
            ),
            "model": {
                "path": str(self.config.model.path),
                "available_offline": self.config.model.path.is_dir(),
                "loaded": self.encoder.loaded,
                "device": self.encoder.device,
            },
            "cameras": cameras,
        }
