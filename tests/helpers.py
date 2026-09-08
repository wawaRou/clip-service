"""External video and image-encoder substitutes for service integration tests."""

import http.client
import io
import json
import threading
import time

import numpy as np
from PIL import Image

from clip_service.config import load_config
from clip_service.http_api import ServiceHTTPServer
from clip_service.service import ClipService


class VideoFeed:
    value = 0
    connected = True
    paused = False

    def isOpened(self):
        return self.connected

    def read(self):
        while self.paused:
            time.sleep(0.01)
        time.sleep(0.01)
        return self.connected, np.full((24, 32, 3), self.value, dtype=np.uint8)

    def release(self):
        pass


class PixelEncoder:
    device = "cpu"
    loaded = True

    def __init__(self):
        self.values = []

    def encode_jpeg(self, jpeg):
        with Image.open(io.BytesIO(jpeg)) as image:
            value = image.getpixel((0, 0))[0]
        return self._encode_value(value)

    def encode_bgr(self, pixels):
        return self._encode_value(int(pixels[0, 0, 2]))

    def _encode_value(self, value):
        self.values.append(value)
        return np.array([1, 0] if value < 128 else [0, 1], dtype=np.float32)


class RunningService:
    def __init__(self, directory, settings=""):
        path = directory / "service.toml"
        path.write_text(
            '[frigate]\nrtsp_base_url="rtsp://localhost:8554"\n'
            "[detection]\nstable_seconds=0.2\ncandidate_frame_interval=0.05\n"
            "inference_fps=20\nring_max_fps=40\nreconnect_delay_seconds=0.05\n"
            + settings
            + '\n[cameras.room]\nstream="room_sub"\n'
        )
        self.feed = VideoFeed()
        self.encoder = PixelEncoder()
        self.service = ClipService(
            load_config(path), capture_factory=lambda url, settings: self.feed, encoder=self.encoder
        )
        self.server = ServiceHTTPServer(("127.0.0.1", 0), self.service)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.service.start()
        self.thread.start()
        self.wait_for(
            lambda: not self.request("GET", "/api/v1/health")[1]["cameras"]["room"]["stale"]
        )
        return self

    def __exit__(self, *exc):
        self.feed.paused = False
        self.service.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, method, path, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        connection.request(
            method,
            path,
            None if body is None else json.dumps(body),
            {} if body is None else {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        data = response.read()
        result = json.loads(data) if "json" in response.getheader("Content-Type", "") else data
        connection.close()
        return response.status, result

    def subscribe(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        connection.request("GET", "/api/v1/events")
        return connection, connection.getresponse()

    @staticmethod
    def wait_for(predicate):
        deadline = time.monotonic() + 3
        while not predicate():
            assert time.monotonic() < deadline, "service did not reach expected state"
            time.sleep(0.01)

    def start_episode(self):
        assert (
            self.request("POST", "/api/v1/cameras/room/episodes", {"episode_id": "motion-1"})[0]
            == 201
        )
        assert (
            self.request(
                "PUT",
                "/api/v1/cameras/room/baseline",
                {
                    "episode_id": "motion-1",
                    "source": {"type": "latest"},
                },
            )[0]
            == 200
        )


def next_event(response):
    event = {}
    while True:
        line = response.readline().decode().strip()
        if not line and event:
            return event
        if line.startswith("event: "):
            event["type"] = line[7:]
        if line.startswith("data: "):
            event["data"] = json.loads(line[6:])
