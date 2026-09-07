"""HTTP commands, candidate images and SSE notifications for Agent Server."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import math
import queue
import re
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlsplit

from .errors import ServiceError
from .service import ClipService

CAMERA_EPISODES = re.compile(r"^/api/v1/cameras/([^/]+)/episodes$")
CAMERA_EPISODE = re.compile(r"^/api/v1/cameras/([^/]+)/episodes/([^/]+)$")
CAMERA_BASELINE = re.compile(r"^/api/v1/cameras/([^/]+)/baseline$")
CAMERA_FRAME_WINDOWS = re.compile(r"^/api/v1/cameras/([^/]+)/frame-windows$")
CANDIDATE = re.compile(r"^/api/v1/candidates/([a-f0-9]+)$")
CANDIDATE_FRAME = re.compile(r"^/api/v1/candidates/([a-f0-9]+)/frames/(\d+)$")
CANDIDATE_ACK = re.compile(r"^/api/v1/candidates/([a-f0-9]+)/ack$")
MAX_JSON_BODY = 10 * 1024 * 1024
MAX_FRAME_WINDOW_BYTES = 8 * 1024 * 1024


def _text(body: dict[str, Any], name: str) -> str:
    value = body.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ServiceError(f"{name} must be a nonempty string")
    return value


def _integer(body: dict[str, Any], name: str) -> int:
    value = body.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ServiceError(f"{name} must be an integer")
    return value


def _number(body: dict[str, Any], name: str) -> float:
    value = body.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ServiceError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError:
        raise ServiceError(f"{name} must be a finite number") from None
    if not math.isfinite(number):
        raise ServiceError(f"{name} must be a finite number")
    return number


class ServiceHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], service: ClipService) -> None:
        self.service = service
        super().__init__(address, RequestHandler)


class RequestHandler(BaseHTTPRequestHandler):
    server: ServiceHTTPServer  # pyright: ignore[reportIncompatibleVariableOverride]
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self._dispatch(self._get)

    def do_POST(self) -> None:
        self._dispatch(self._post)

    def do_PUT(self) -> None:
        self._dispatch(self._put)

    def do_DELETE(self) -> None:
        self._dispatch(self._delete)

    def _dispatch(self, operation: Callable[[str], None]) -> None:
        try:
            operation(urlsplit(self.path).path)
        except ServiceError as err:
            self.close_connection = True
            self._json(err.status, {"error": {"code": err.code, "message": str(err)}})
        except (TimeoutError, BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            logging.getLogger(__name__).exception("HTTP request failed")
            self.close_connection = True
            self._json(
                500,
                {"error": {"code": "internal_error", "message": "internal error"}},
            )

    def _get(self, path: str) -> None:
        if path == "/api/v1/health":
            self._json(200, self.server.service.health())
            return
        if path == "/api/v1/events":
            self._events()
            return
        match = CANDIDATE.fullmatch(path)
        if match:
            record = self.server.service.store.get(match.group(1))
            if record is None:
                raise ServiceError("candidate not found", 404, "candidate_not_found")
            self._json(200, record.to_api())
            return
        match = CANDIDATE_FRAME.fullmatch(path)
        if match:
            content = self.server.service.store.get_frame(match.group(1), int(match.group(2)))
            if content is None:
                raise ServiceError("candidate frame not found", 404, "frame_not_found")
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "private, max-age=86400")
            self.end_headers()
            self.wfile.write(content)
            return
        raise ServiceError("endpoint not found", 404, "not_found")

    def _post(self, path: str) -> None:
        match = CAMERA_FRAME_WINDOWS.fullmatch(path)
        if match:
            body = self._read_json()
            camera = self.server.service.camera(unquote(match.group(1)))
            episode_id = _text(body, "episode_id")
            window_start = _number(body, "window_start")
            window_end = _number(body, "window_end")
            frame_count = _integer(body, "frame_count")
            frames = camera.frame_window(
                episode_id,
                window_start,
                window_end,
                frame_count,
            )
            if sum(len(frame.jpeg) for frame in frames) > MAX_FRAME_WINDOW_BYTES:
                raise ServiceError(
                    "camera frame window exceeds response limit",
                    503,
                    "frame_window_too_large",
                )
            self._json(
                200,
                {
                    "schema_version": "1.0",
                    "camera": camera.name,
                    "episode_id": episode_id,
                    "window_start": window_start,
                    "window_end": window_end,
                    "frames": [
                        {
                            "index": index,
                            "timestamp": frame.timestamp,
                            "content_type": "image/jpeg",
                            "jpeg_base64": base64.b64encode(frame.jpeg).decode("ascii"),
                        }
                        for index, frame in enumerate(frames)
                    ],
                },
            )
            return
        match = CAMERA_EPISODES.fullmatch(path)
        if match:
            body = self._read_json()
            episode_id = _text(body, "episode_id")
            camera = self.server.service.camera(unquote(match.group(1)))
            camera.start_episode(episode_id)
            self._json(
                201,
                {
                    "camera": camera.name,
                    "episode_id": episode_id,
                    "state": "active",
                },
            )
            return
        match = CANDIDATE_ACK.fullmatch(path)
        if match:
            body = self._read_json()
            episode_id = _text(body, "episode_id")
            triggered_vlm = body.get("triggered_vlm")
            if not isinstance(triggered_vlm, bool):
                raise ServiceError("triggered_vlm must be a boolean")
            candidate_id = match.group(1)
            record = self.server.service.store.get(candidate_id)
            if record is None:
                raise ServiceError("candidate not found", 404, "candidate_not_found")
            camera = self.server.service.camera(record.camera)
            baseline_jpeg = None
            if body.get("baseline") is not None:
                baseline_jpeg = self._resolve_baseline(camera.name, episode_id, body["baseline"])
            updated = camera.acknowledge(
                candidate_id,
                episode_id=episode_id,
                triggered_vlm=triggered_vlm,
                baseline_jpeg=baseline_jpeg,
            )
            self._json(200, updated.to_api())
            return
        raise ServiceError("endpoint not found", 404, "not_found")

    def _put(self, path: str) -> None:
        match = CAMERA_BASELINE.fullmatch(path)
        if not match:
            raise ServiceError("endpoint not found", 404, "not_found")
        body = self._read_json()
        camera_name = unquote(match.group(1))
        episode_id = _text(body, "episode_id")
        jpeg = self._resolve_baseline(camera_name, episode_id, body.get("source"))
        camera = self.server.service.camera(camera_name)
        camera.set_baseline(episode_id, jpeg)
        self._json(
            200,
            {
                "camera": camera_name,
                "episode_id": episode_id,
                "baseline_ready": True,
                "device": self.server.service.encoder.device,
            },
        )

    def _delete(self, path: str) -> None:
        match = CAMERA_EPISODE.fullmatch(path)
        if not match:
            raise ServiceError("endpoint not found", 404, "not_found")
        camera = self.server.service.camera(unquote(match.group(1)))
        episode_id = unquote(match.group(2))
        camera.stop_episode(episode_id)
        self._json(200, {"camera": camera.name, "episode_id": episode_id, "state": "stopped"})

    def _resolve_baseline(self, camera_name: str, episode_id: str, source: Any) -> bytes:
        if not isinstance(source, dict):
            raise ServiceError("baseline source must be an object")
        camera = self.server.service.camera(camera_name)
        source_type = source.get("type")
        if source_type == "latest":
            return camera.latest_jpeg()
        if source_type == "candidate":
            candidate_id = _text(source, "candidate_id")
            frame_index = _integer(source, "frame_index")
            record = self.server.service.store.get(candidate_id)
            if record is None:
                raise ServiceError("candidate not found", 404, "candidate_not_found")
            if record.camera != camera_name or record.episode_id != episode_id:
                raise ServiceError(
                    "candidate does not belong to episode", 409, "candidate_mismatch"
                )
            frame = self.server.service.store.get_frame(candidate_id, frame_index)
            if frame is None:
                raise ServiceError("candidate frame not found", 404, "frame_not_found")
            return frame
        if source_type == "jpeg_base64":
            try:
                content = base64.b64decode(_text(source, "data"), validate=True)
            except (binascii.Error, ValueError) as err:
                raise ServiceError("invalid base64 JPEG") from err
            if not content:
                raise ServiceError("JPEG must not be empty")
            return content
        raise ServiceError("baseline source type must be latest, candidate, or jpeg_base64")

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as err:
            raise ServiceError("invalid Content-Length") from err
        if length <= 0 or length > MAX_JSON_BODY:
            raise ServiceError(
                "JSON body is required and must be at most 10 MiB", 413, "invalid_body"
            )
        try:
            value = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError) as err:
            raise ServiceError("invalid JSON") from err
        if not isinstance(value, dict):
            raise ServiceError("JSON body must be an object")
        return value

    def _json(self, status: int | HTTPStatus, value: Any) -> None:
        content = json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _events(self) -> None:
        # Subscribe before returning headers, so the first client command cannot race us.
        with self.server.service.events.subscribe() as subscriber:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.flush()
            self.close_connection = True
            while True:
                try:
                    event = subscriber.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")
                else:
                    if event is None:
                        return
                    self.wfile.write(event.as_sse())
                self.wfile.flush()

    def log_message(self, format: str, *args: object) -> None:
        # Keep the service quiet by default; application logging reports failures.
        return
