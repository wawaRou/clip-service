"""Validate client input through the actual HTTP server."""

import json
import threading
import time
from http.client import HTTPConnection

import numpy as np
import pytest

from clip_service.config import load_config
from clip_service.http_api import ServiceHTTPServer
from clip_service.service import ClipService
from tests.helpers import RunningService


@pytest.fixture
def http_api(tmp_path):
    path = tmp_path / "service.toml"
    path.write_text(
        '[frigate]\nrtsp_base_url="rtsp://localhost:8554"\n[cameras.front]\nstream="front_sub"\n'
    )
    service = ClipService(load_config(path))
    server = ServiceHTTPServer(("127.0.0.1", 0), service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, service
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        service.close()


def request(server, method, path, body):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    try:
        connection.request(method, path, body, {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def test_episode_id_requires_a_nonempty_string(http_api):
    server, _ = http_api
    for value in (True, 3, None, "", "  "):
        status, body = request(
            server, "POST", "/api/v1/cameras/front/episodes", json.dumps({"episode_id": value})
        )
        assert status == 400
        assert body["error"]["code"] == "bad_request"
        assert "episode_id" in body["error"]["message"]


def test_malformed_json_is_a_client_error(http_api):
    server, _ = http_api
    for content in (b"\xff", b'{"episode_id":', b"[]"):
        status, body = request(server, "POST", "/api/v1/cameras/front/episodes", content)
        assert status == 400
        assert body["error"]["code"] == "bad_request"


def test_frame_window_rejects_invalid_counts_and_timestamps(tmp_path):
    with RunningService(tmp_path) as app:
        app.start_episode()
        valid = {
            "episode_id": "motion-1",
            "window_start": time.time() - 0.01,
            "window_end": time.time(),
            "frame_count": 1,
        }
        for field, value in (
            ("window_start", True),
            ("window_end", float("nan")),
            ("window_end", float("inf")),
            ("frame_count", True),
            ("frame_count", 1.5),
            ("frame_count", 0),
            ("frame_count", 10),
        ):
            status, body = app.request(
                "POST", "/api/v1/cameras/room/frame-windows", valid | {field: value}
            )
            assert status == 400, (field, value, body)
            assert body["error"]["code"] == "bad_request"


def test_frame_window_returns_one_jpeg_at_the_midpoint(tmp_path):
    with RunningService(tmp_path) as app:
        app.start_episode()
        now = time.time()
        status, body = app.request(
            "POST",
            "/api/v1/cameras/room/frame-windows",
            {
                "episode_id": "motion-1",
                "window_start": now - 0.01,
                "window_end": now,
                "frame_count": 1,
            },
        )
        assert status == 200
        assert body["camera"] == "room"
        assert body["episode_id"] == "motion-1"
        assert len(body["frames"]) == 1
        frame = body["frames"][0]
        assert frame["index"] == 0
        assert frame["content_type"] == "image/jpeg"
        assert abs(frame["timestamp"] - now) < 0.5
        assert frame["jpeg_base64"].startswith("/9j/")


def test_sse_connection_ends_when_service_closes(http_api):
    server, service = http_api
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    try:
        connection.request("GET", "/api/v1/events")
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type") == "text/event-stream"
        service.close()
        assert response.read() == b""
    finally:
        connection.close()


def test_frame_window_limits_large_image_responses(tmp_path):
    app = RunningService(tmp_path)
    pixels = np.random.default_rng(0).integers(0, 256, (1536, 1536, 3), dtype=np.uint8)

    def read_large_frame():
        time.sleep(0.05)
        return True, pixels

    app.feed.read = read_large_frame
    with app:
        assert (
            app.request("POST", "/api/v1/cameras/room/episodes", {"episode_id": "motion-1"})[0]
            == 201
        )
        timestamp = app.request("GET", "/api/v1/health")[1]["cameras"]["room"]["last_frame_at"]
        status, body = app.request(
            "POST",
            "/api/v1/cameras/room/frame-windows",
            {
                "episode_id": "motion-1",
                "window_start": timestamp,
                "window_end": timestamp,
                "frame_count": 9,
            },
        )
        assert status == 503
        assert body["error"]["code"] == "frame_window_too_large"
