import json
import threading
import time
from urllib.request import urlopen

import numpy as np

from clip_service.config import load_config
from clip_service.http_api import ServiceHTTPServer
from clip_service.service import ClipService


class CameraFeed:
    """Stand-in for an external camera connection, producing real image pixels."""

    def __init__(self):
        self.closed = False

    def isOpened(self):
        return True

    def read(self):
        time.sleep(0.01)
        return True, np.zeros((24, 32, 3), dtype=np.uint8)

    def release(self):
        self.closed = True


def test_configured_camera_becomes_visible_through_http(tmp_path):
    config_file = tmp_path / "service.toml"
    config_file.write_text(
        '[frigate]\nrtsp_base_url="rtsp://localhost:8554"\n'
        '[cameras.living_room]\nstream="living_room_sub"\n'
    )
    feeds = []

    def connect(url, settings):
        assert url == "rtsp://localhost:8554/living_room_sub"
        feed = CameraFeed()
        feeds.append(feed)
        return feed

    service = ClipService(load_config(config_file), capture_factory=connect)
    server = ServiceHTTPServer(("127.0.0.1", 0), service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    service.start()
    thread.start()
    try:
        deadline = time.monotonic() + 2
        while True:
            with urlopen(f"http://127.0.0.1:{server.server_port}/api/v1/health") as response:
                health = json.load(response)
            camera = health["cameras"]["living_room"]
            if camera["buffered_frames"]:
                break
            assert time.monotonic() < deadline, health
            time.sleep(0.01)
        assert camera["connected"] is True
        assert camera["frame_age_seconds"] < 1
        assert camera["stale"] is False
        assert health["status"] == "ok"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        service.close()
    assert feeds and all(feed.closed for feed in feeds)


def test_empty_configuration_starts_without_sample_camera(tmp_path):
    path = tmp_path / "empty.toml"
    path.write_text('[frigate]\nrtsp_base_url="rtsp://localhost:8554"\n')
    service = ClipService(load_config(path))
    service.start()
    try:
        assert service.health()["cameras"] == {}
    finally:
        service.close()


def test_disconnected_camera_is_stale_then_reconnects_without_stopping_other_camera(tmp_path):
    path = tmp_path / "reconnect.toml"
    path.write_text(
        '[frigate]\nrtsp_base_url="rtsp://localhost:8554"\n'
        "[detection]\nreconnect_delay_seconds=0.2\n"
        '[cameras.front]\nstream="front"\n[cameras.back]\nstream="back"\n'
    )
    disconnected = threading.Event()
    recovered = threading.Event()

    class IntermittentFeed(CameraFeed):
        def read(self):
            if disconnected.is_set() and not recovered.is_set():
                time.sleep(0.01)
                return False, None
            return super().read()

    def connect(url, settings):
        return IntermittentFeed() if url.endswith("/front") else CameraFeed()

    service = ClipService(load_config(path), capture_factory=connect)
    service.start()

    def wait_until(predicate):
        deadline = time.monotonic() + 2
        while not predicate(service.health()["cameras"]):
            assert time.monotonic() < deadline, service.health()
            time.sleep(0.01)

    try:
        wait_until(lambda cameras: all(not c["stale"] for c in cameras.values()))
        disconnected.set()
        wait_until(lambda cameras: cameras["front"]["stale"])
        assert service.health()["cameras"]["front"]["buffered_frames"] == 0
        assert service.health()["cameras"]["back"]["stale"] is False
        recovered.set()
        wait_until(lambda cameras: not cameras["front"]["stale"])
    finally:
        service.close()
