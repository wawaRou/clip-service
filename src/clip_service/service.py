"""Own the configured cameras, shared inference loop and configuration reloads."""

import logging
import threading
import time
from pathlib import Path
from typing import Any

from .camera import Camera
from .config import ConfigError, ServiceConfig, load_config
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
        config_path: Path | None = None,
        capture_factory: CaptureFactory = open_capture,
        encoder: ImageEncoder | None = None,
    ) -> None:
        self.config = config
        self._config_path = config_path.resolve() if config_path is not None else None
        self._capture_factory = capture_factory
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
        self._reload_error: str | None = None
        # Registry operations are short; encoding and reader shutdown run outside this lock.
        self._registry_lock = threading.RLock()
        self._reload_lock = threading.Lock()
        self._retired_readers: list[CameraReader] = []
        self.cameras = {
            name: self._new_camera(name, config)
            for name, camera in config.cameras.items()
            if camera.enabled
        }
        self._scheduler = FrameScheduler(self._rates(), now=time.monotonic())

    def _new_camera(self, name: str, config: ServiceConfig) -> Camera:
        reader = CameraReader(
            name, config.camera_url(name), config.camera_settings(name), self._capture_factory
        )
        return Camera(reader, self.encoder, self.store, self.events)

    def _rates(self) -> dict[str, float]:
        return {name: camera.settings.inference_fps for name, camera in self.cameras.items()}

    def _camera_snapshot(self) -> dict[str, Camera]:
        with self._registry_lock:
            return self.cameras.copy()

    def start(self) -> None:
        for camera in self.cameras.values():
            camera.reader.start()
        self._inference = threading.Thread(target=self._detect, name="clip-inference", daemon=True)
        self._inference.start()

    @staticmethod
    def _close_readers(readers: list[CameraReader]) -> None:
        for reader in readers:
            reader.request_stop()
        # FFmpeg can serialize opens; budget all opens and the longest current read.
        timeout = sum(reader.open_timeout_budget for reader in readers)
        timeout += max((reader.read_timeout_budget for reader in readers), default=0) + 1
        deadline = time.monotonic() + timeout
        for reader in readers:
            reader.close(timeout=max(0, deadline - time.monotonic()))

    def close(self) -> None:
        with self._reload_lock:
            self._stop.set()
            self.events.close()
            self._close_readers(
                [camera.reader for camera in self._camera_snapshot().values()]
                + self._retired_readers
            )
            self._retired_readers.clear()
        if self._inference is not None:
            self._inference.join(timeout=10)
            if self._inference.is_alive():
                raise TimeoutError("CLIP inference did not finish during shutdown")

    def camera(self, name: str) -> Camera:
        with self._registry_lock:
            try:
                return self.cameras[name]
            except KeyError:
                raise ServiceError("camera not found", 404, "camera_not_found") from None

    def reload_config(self) -> dict[str, Any]:
        """Read the startup TOML again, validate it fully, then apply camera differences."""
        with self._reload_lock:
            if self._config_path is None:
                raise ServiceError(
                    "reload requires a startup config path", 409, "reload_unavailable"
                )
            if self._stop.is_set():
                raise ServiceError("service is stopping", 409, "service_stopping")
            try:
                config = load_config(self._config_path)
            except ConfigError as error:
                raise ServiceError(str(error), 400, "invalid_config") from None
            restart_fields = [
                field
                for field in ("server", "model", "data_dir", "retention_hours")
                if getattr(config, field) != getattr(self.config, field)
            ]
            if restart_fields:
                raise ServiceError(
                    "settings require restart: " + ", ".join(restart_fields),
                    409,
                    "restart_required",
                )
            try:
                with self._registry_lock:
                    try:
                        changes = self._apply_cameras(config, self._retired_readers)
                        self.config = config
                    finally:
                        self._scheduler.set_rates(self._rates(), now=time.monotonic())
                self._close_readers(self._retired_readers)
                self._retired_readers.clear()
            except (OSError, RuntimeError) as error:
                self._reload_error = (
                    "configuration reload stopped during application; "
                    "some cameras may have changed; inspect health and retry"
                )
                raise ServiceError(self._reload_error, 503, "reload_failed") from error
            self._reload_error = None
            return changes | {"cameras": self.health()["cameras"]}

    def _apply_cameras(
        self, config: ServiceConfig, retired: list[CameraReader]
    ) -> dict[str, list[str]]:
        changes: dict[str, list[str]] = {"added": [], "updated": [], "removed": [], "restarted": []}
        for name, camera in list(self.cameras.items()):
            configured = config.cameras.get(name)
            reason = None
            if configured is None:
                reason = "removed"
            elif not configured.enabled:
                reason = "disabled"
            elif camera.reader.source_url != config.camera_url(name):
                reason = "stream_changed"
            if reason is not None:
                camera.retire(reason)
                retired.append(camera.reader)
                del self.cameras[name]
                changes["restarted" if reason == "stream_changed" else "removed"].append(name)
        for name, configured in config.cameras.items():
            if not configured.enabled:
                continue
            if name not in self.cameras:
                camera = self._new_camera(name, config)
                if self._inference is not None:
                    camera.reader.start()
                self.cameras[name] = camera
                if name not in changes["restarted"]:
                    changes["added"].append(name)
            else:
                camera = self.cameras[name]
                settings = config.camera_settings(name)
                if settings != camera.settings:
                    camera.update_settings(settings)
                    changes["updated"].append(name)
        return changes

    def _detect(self) -> None:
        while not self._stop.is_set():
            self._maintain_candidates()
            with self._registry_lock:
                due = [
                    (name, self.cameras[name], self.cameras[name].settings.inference_fps)
                    for name in self._scheduler.due(time.monotonic())
                ]
            for name, camera, rate in due:
                if self._stop.is_set():
                    return
                try:
                    camera.process_latest()
                except (ServiceError, OSError) as error:
                    logging.getLogger(__name__).warning("camera %s: %s", camera.name, error)
                finally:
                    with self._registry_lock:
                        if (
                            self.cameras.get(name) is camera
                            and camera.settings.inference_fps == rate
                        ):
                            self._scheduler.complete(name, time.monotonic())
            with self._registry_lock:
                delay = min(0.1, self._scheduler.delay(time.monotonic()))
            self._stop.wait(delay)

    def pending_candidates(self) -> list[dict[str, Any]]:
        candidates = (camera.pending_candidate() for camera in self._camera_snapshot().values())
        return [candidate.to_api() for candidate in candidates if candidate is not None]

    def _maintain_candidates(self) -> None:
        failed = False
        now = time.time()
        for camera in self._camera_snapshot().values():
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
        cameras = {name: camera.health() for name, camera in self._camera_snapshot().items()}
        return {
            "schema_version": "1.0",
            "status": "degraded"
            if self._maintenance_error
            or self._reload_error
            or any(
                camera["stale"] or camera["inference_error"] or camera["storage_error"]
                for camera in cameras.values()
            )
            else "ok",
            "uptime_seconds": time.monotonic() - self.started_at,
            "maintenance_error": self._maintenance_error,
            "reload_error": self._reload_error,
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
