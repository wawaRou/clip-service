"""Exercise frame delivery through the real reader, scheduler and episode lifecycle."""

import io
import threading
import time
from contextlib import contextmanager

import numpy as np
import pytest
from PIL import Image

from clip_service.config import ServiceConfig
from clip_service.service import ClipService
from tests.helpers import PixelEncoder
from tests.test_reader_settings import ControlledFeed, wait_for


class GatedEncoder(PixelEncoder):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def encode_bgr(self, pixels):
        self.entered.set()
        assert self.release.wait(3), "test did not release inference"
        return super().encode_bgr(pixels)


@contextmanager
def driven_service(tmp_path, fps=100, names=("room",)):
    feeds = {name: ControlledFeed() for name in names}
    encoder = GatedEncoder()
    config = ServiceConfig.model_validate(
        {
            "frigate": {"rtsp_base_url": "rtsp://localhost:8554"},
            "data_dir": tmp_path,
            "detection": {"inference_fps": fps, "similarity_threshold": -1},
            "cameras": {name: {"stream": name} for name in names},
        }
    )
    service = ClipService(
        config,
        capture_factory=lambda url, settings: feeds[url.rsplit("/", 1)[1]].open(url, settings),
        encoder=encoder,
    )
    jpeg = io.BytesIO()
    Image.new("RGB", (32, 24), "black").save(jpeg, format="JPEG")
    for camera in service.cameras.values():
        camera.start_episode("episode")
        camera.set_baseline("episode", jpeg.getvalue())
    encoder.values.clear()
    service.start()
    try:
        yield service, feeds, encoder, jpeg.getvalue()
    finally:
        encoder.release.set()
        for camera in service.cameras.values():
            camera.reader.request_stop()
        for feed in feeds.values():
            feed.frames.put(None)
        service.close()


def send(feed, reader, value):
    feed.send_frame(reader, np.full((24, 32, 3), value, dtype=np.uint8))


def test_first_frame_after_idle_wakes_inference_without_consuming_empty_turns(tmp_path):
    with driven_service(tmp_path, fps=1) as (service, feeds, encoder, _):
        time.sleep(0.15)
        send(feeds["room"], service.camera("room").reader, 10)
        assert encoder.entered.wait(0.2)
        wait_for(lambda: service.camera("room").health()["inference"]["frames_processed"] == 1)


def test_frames_arriving_during_inference_are_processed_once_in_order(tmp_path):
    with driven_service(tmp_path) as (service, feeds, encoder, _):
        reader = service.camera("room").reader
        encoder.release.clear()
        send(feeds["room"], reader, 10)
        assert encoder.entered.wait(1)
        send(feeds["room"], reader, 11)
        send(feeds["room"], reader, 12)
        encoder.release.set()
        wait_for(lambda: len(encoder.values) == 3)
        assert encoder.values == [10, 11, 12]
        time.sleep(0.05)
        assert service.camera("room").health()["inference"]["frames_processed"] == 3


@pytest.mark.parametrize("change", ["baseline", "episode", "settings"])
def test_session_changes_discard_frames_waiting_for_previous_state(tmp_path, change):
    with driven_service(tmp_path, fps=5) as (service, feeds, encoder, jpeg):
        camera = service.camera("room")
        send(feeds["room"], camera.reader, 10)
        wait_for(lambda: camera.health()["inference"]["frames_processed"] == 1)
        send(feeds["room"], camera.reader, 11)
        if change == "baseline":
            camera.set_baseline("episode", jpeg)
        elif change == "episode":
            camera.stop_episode("episode")
            camera.start_episode("new-episode")
            camera.set_baseline("new-episode", jpeg)
        else:
            camera.update_settings(camera.settings.model_copy(update={"stable_seconds": 0.4}))
        time.sleep(0.25)
        assert camera.health()["inference"]["frames_processed"] == 1
        send(feeds["room"], camera.reader, 12)
        wait_for(lambda: camera.health()["inference"]["frames_processed"] == 2)
        assert 11 not in encoder.values


def test_busy_camera_yields_to_another_camera_with_queued_frames(tmp_path):
    with driven_service(tmp_path, names=("room", "door")) as (service, feeds, encoder, _):
        encoder.release.clear()
        send(feeds["room"], service.camera("room").reader, 10)
        assert encoder.entered.wait(1)
        send(feeds["room"], service.camera("room").reader, 11)
        send(feeds["room"], service.camera("room").reader, 12)
        send(feeds["door"], service.camera("door").reader, 20)
        send(feeds["door"], service.camera("door").reader, 21)
        encoder.release.set()
        wait_for(lambda: len(encoder.values) == 5)
        assert encoder.values == [10, 20, 11, 21, 12]


def test_stale_stream_cannot_confirm_a_change_using_tracking_from_before_the_gap(tmp_path):
    with driven_service(tmp_path) as (service, feeds, encoder, _):
        camera = service.camera("room")
        camera.update_settings(
            camera.settings.model_copy(
                update={
                    "similarity_threshold": 0.9,
                    "stable_seconds": 0.4,
                    "frame_max_age_seconds": 0.1,
                }
            )
        )
        for count in range(1, 6):
            send(feeds["room"], camera.reader, 200)
            wait_for(lambda: camera.health()["inference"]["frames_processed"] == count)
            time.sleep(0.05)
        time.sleep(0.3)
        resumed_at = time.time()
        send(feeds["room"], camera.reader, 200)
        wait_for(lambda: camera.health()["inference"]["frames_processed"] == 6)
        assert camera.pending_candidate() is None
        for _ in range(10):
            time.sleep(0.05)
            send(feeds["room"], camera.reader, 200)
        wait_for(lambda: camera.pending_candidate() is not None)
        assert camera.pending_candidate().started_at >= resumed_at
