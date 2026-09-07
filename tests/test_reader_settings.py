"""Hot settings changes observed through a reader and its external video feed."""

import queue
import threading
import time
from contextlib import contextmanager

import numpy as np

from clip_service.config import DetectionConfig
from clip_service.video import CameraReader


class ControlledFeed:
    def __init__(self):
        self.frames = queue.Queue()
        self.opens = []
        self.releases = 0

    def open(self, url, settings):
        self.opens.append(settings)
        return self

    def isOpened(self):
        return True

    def read(self):
        pixels = self.frames.get(timeout=5)
        return pixels is not None, pixels

    def release(self):
        self.releases += 1

    def send_frame(self, reader, pixels=None):
        count = reader.health()["frames_received"]
        if pixels is None:
            pixels = np.zeros((24, 32, 3), dtype=np.uint8)
        self.frames.put(pixels)
        wait_for(lambda: (frame := reader.latest_frame()) is not None and frame.sequence > count)


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError("reader did not reach the expected state")


@contextmanager
def running_reader(settings):
    feed = ControlledFeed()
    reader = CameraReader("room", "rtsp://localhost:8554/room_sub", settings, feed.open)
    reader.start()
    try:
        yield reader, feed
    finally:
        reader.request_stop()
        feed.frames.put(None)
        reader.close()


def test_smaller_history_preserves_recent_frames_without_reopening_stream():
    with running_reader(DetectionConfig(ring_seconds=2, ring_max_fps=100)) as (reader, feed):
        for _ in range(15):
            feed.send_frame(reader)
            time.sleep(0.012)
        before = reader.ring.snapshot()
        latest = reader.latest_frame()

        reader.update_settings(
            DetectionConfig(ring_seconds=0.5, ring_max_fps=10, candidate_frame_interval=0.1)
        )

        assert reader.ring.snapshot() == before[-7:]
        assert reader.latest_frame() == latest
        assert reader.source_url == "rtsp://localhost:8554/room_sub"
        assert len(feed.opens) == 1
        assert feed.releases == 0


def test_faster_history_sampling_starts_on_the_next_frame():
    settings = DetectionConfig(ring_max_fps=10, candidate_frame_interval=0.1)
    with running_reader(settings) as (reader, feed):
        feed.send_frame(reader)
        reader.update_settings(settings.model_copy(update={"ring_max_fps": 100}))
        time.sleep(0.02)
        feed.send_frame(reader)

        assert len(reader.ring) == 2
        assert len(feed.opens) == 1


def test_close_honors_the_active_decoder_timeout_after_it_is_reduced():
    reading = threading.Event()

    class SlowCapture(ControlledFeed):
        def read(self):
            reading.set()
            time.sleep(1.2)
            return False, None

    feed = SlowCapture()
    settings = DetectionConfig(open_timeout_seconds=0.1, read_timeout_seconds=2)
    reader = CameraReader("room", "rtsp://localhost/room", settings, feed.open)
    reader.start()
    try:
        assert reading.wait(2)
        reader.update_settings(
            settings.model_copy(
                update={"open_timeout_seconds": 0.001, "read_timeout_seconds": 0.001}
            )
        )
        reader.close()
        assert feed.releases == 1
    finally:
        reader.close(timeout=3)


def test_shorter_retry_delay_takes_effect_while_waiting_to_reconnect():
    class InitiallyUnavailableFeed(ControlledFeed):
        def isOpened(self):
            return len(self.opens) > 1

    feed = InitiallyUnavailableFeed()
    settings = DetectionConfig(reconnect_delay_seconds=10)
    reader = CameraReader("room", "rtsp://localhost/room", settings, feed.open)
    reader.start()
    try:
        wait_for(lambda: feed.releases == 1)
        time.sleep(0.02)
        reader.update_settings(settings.model_copy(update={"reconnect_delay_seconds": 0.01}))
        wait_for(lambda: len(feed.opens) == 2)
    finally:
        reader.request_stop()
        feed.frames.put(None)
        reader.close()


def test_longer_retry_delay_does_not_trigger_an_immediate_reconnect():
    class InitiallyUnavailableFeed(ControlledFeed):
        def isOpened(self):
            return len(self.opens) > 1

    feed = InitiallyUnavailableFeed()
    settings = DetectionConfig(reconnect_delay_seconds=0.1)
    reader = CameraReader("room", "rtsp://localhost/room", settings, feed.open)
    reader.start()
    try:
        wait_for(lambda: feed.releases == 1)
        time.sleep(0.02)
        reader.update_settings(settings.model_copy(update={"reconnect_delay_seconds": 0.3}))
        time.sleep(0.15)
        assert len(feed.opens) == 1
        wait_for(lambda: len(feed.opens) == 2)
    finally:
        reader.request_stop()
        feed.frames.put(None)
        reader.close()


def test_new_decoder_timeouts_apply_at_reconnect_and_replace_shutdown_budgets():
    settings = DetectionConfig(
        open_timeout_seconds=3, read_timeout_seconds=4, reconnect_delay_seconds=0.01
    )
    with running_reader(settings) as (reader, feed):
        feed.send_frame(reader)
        updated = settings.model_copy(
            update={"open_timeout_seconds": 0.2, "read_timeout_seconds": 0.3}
        )
        reader.update_settings(updated)
        assert reader.open_timeout_budget == 3
        assert reader.read_timeout_budget == 4

        feed.frames.put(None)
        wait_for(lambda: len(feed.opens) == 2)
        assert feed.opens[1] == updated
        assert reader.open_timeout_budget == 0.2
        assert reader.read_timeout_budget == 0.3


def test_new_frame_quality_and_freshness_limits_apply_without_reconnecting():
    settings = DetectionConfig(jpeg_quality=95)
    pixels = np.random.default_rng(42).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    with running_reader(settings) as (reader, feed):
        feed.send_frame(reader, pixels)
        original = reader.latest_frame()
        reader.update_settings(settings.model_copy(update={"jpeg_quality": 10}))
        feed.send_frame(reader, pixels)
        updated = reader.latest_frame()
        assert len(updated.jpeg) < len(original.jpeg) / 2
        assert updated.stream_generation == original.stream_generation

        time.sleep(0.02)
        reader.update_settings(settings.model_copy(update={"frame_max_age_seconds": 0.001}))
        assert reader.latest_frame() is None
        assert reader.health()["stale"]
        reader.update_settings(settings)
        assert reader.latest_frame() == updated
        assert not reader.health()["stale"]
        assert len(feed.opens) == 1


def test_shorter_history_duration_discards_only_frames_outside_the_new_window():
    settings = DetectionConfig(ring_seconds=2, ring_max_fps=100)
    with running_reader(settings) as (reader, feed):
        for _ in range(6):
            feed.send_frame(reader)
            time.sleep(0.1)
        original = reader.ring.snapshot()
        reader.update_settings(settings.model_copy(update={"ring_seconds": 0.31}))
        retained = reader.ring.snapshot()

        assert original[0] not in retained
        assert retained[-1] == original[-1]
        assert all(frame.timestamp >= original[-1].timestamp - 0.31 for frame in retained)
