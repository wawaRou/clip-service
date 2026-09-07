"""Exercise shared-model scheduling through independent live video feeds."""

import io
import time

from PIL import Image

from clip_service.config import load_config
from clip_service.service import ClipService
from tests.helpers import PixelEncoder, RunningService, VideoFeed


class SlowPixelEncoder(PixelEncoder):
    def encode_jpeg(self, jpeg):
        time.sleep(0.06)
        return super().encode_jpeg(jpeg)


class RunningCameras:
    def __init__(self, directory, rates, *, fps=5, ring_fps=40, encoder=None):
        config_path = directory / "service.toml"
        config_path.write_text(
            '[frigate]\nrtsp_base_url="rtsp://localhost:8554"\n'
            f"[detection]\ninference_fps={fps}\nring_max_fps={ring_fps}\n"
            "stable_seconds=0.2\ncandidate_frame_interval=0.05\n"
            + "".join(
                f'[cameras.{name}]\nstream="{name}_sub"\n'
                + (f"inference_fps={rate}\n" if rate is not None else "")
                for name, rate in rates.items()
            )
        )
        self.feeds = {name: VideoFeed() for name in rates}
        self.encoder = encoder or PixelEncoder()
        self.service = ClipService(
            load_config(config_path, environ={}),
            capture_factory=lambda url, settings: self.feeds[url.rsplit("/", 1)[1][:-4]],
            encoder=self.encoder,
        )
        for index, (name, feed) in enumerate(self.feeds.items()):
            feed.value = (index + 1) * 20
            camera = self.service.camera(name)
            camera.start_episode(f"episode-{name}")
            jpeg = io.BytesIO()
            Image.new("RGB", (32, 24), (feed.value,) * 3).save(jpeg, format="JPEG")
            camera.set_baseline(f"episode-{name}", jpeg.getvalue())

    def __enter__(self):
        self.service.start()
        self.wait_for(lambda: all(not camera["stale"] for camera in self.health().values()))
        return self

    def __exit__(self, *exc):
        for feed in self.feeds.values():
            feed.paused = False
        self.service.close()

    def health(self):
        return self.service.health()["cameras"]

    def counts(self):
        return {
            name: camera["inference"]["frames_processed"] for name, camera in self.health().items()
        }

    def latest_values(self):
        values = {}
        for name in self.feeds:
            frame = self.service.camera(name).reader.latest_frame()
            if frame is not None:
                with Image.open(io.BytesIO(frame.jpeg)) as image:
                    values[name] = image.getpixel((0, 0))[0]
        return values

    wait_for = staticmethod(RunningService.wait_for)


def test_four_cameras_share_one_encoder_and_keep_the_default_detection_rate(tmp_path):
    with RunningCameras(tmp_path, dict.fromkeys(["door", "room", "yard", "garage"])) as app:
        app.wait_for(lambda: min(app.counts().values()) >= 6)
        counts = app.counts().values()
        assert max(counts) - min(counts) <= 2
        for name, health in app.health().items():
            assert app.service.camera(name).encoder is app.encoder
            inference = health["inference"]
            assert inference["target_fps"] == 5
            assert 4 <= inference["actual_fps"] <= 6
            assert inference["measurement_seconds"] > 0
            assert inference["last_processing_seconds"] >= 0
            assert inference["last_frame_latency_seconds"] >= inference["last_processing_seconds"]
            assert health["episode"]["episode_id"] == f"episode-{name}"


def test_a_camera_can_override_the_global_detection_rate(tmp_path):
    with RunningCameras(tmp_path, {"room": None, "door": 10}) as app:
        app.wait_for(lambda: app.counts()["room"] >= 6)
        counts = app.counts()
        assert 1.5 * counts["room"] <= counts["door"] <= 2.5 * counts["room"]
        health = app.health()
        assert health["room"]["inference"]["target_fps"] == 5
        assert health["door"]["inference"]["target_fps"] == 10


def test_detection_frequency_is_not_limited_by_cache_sampling_phase(tmp_path, monkeypatch):
    original_read = VideoFeed.read

    def source_at_40_fps(feed):
        time.sleep(0.015)  # Combined with VideoFeed's 10ms, this is about 40 FPS.
        return original_read(feed)

    monkeypatch.setattr(VideoFeed, "read", source_at_40_fps)
    with RunningCameras(tmp_path, {"room": None}, fps=25, ring_fps=30) as app:
        app.wait_for(lambda: app.counts()["room"] >= 30)
        assert 22 <= app.health()["room"]["inference"]["actual_fps"] <= 27


def test_overloaded_encoder_serves_every_camera_and_skips_old_frames(tmp_path):
    with RunningCameras(
        tmp_path, dict.fromkeys(["door", "room", "yard"]), fps=20, encoder=SlowPixelEncoder()
    ) as app:
        app.wait_for(lambda: min(app.counts().values()) >= 4)
        counts = app.counts()
        assert max(counts.values()) - min(counts.values()) <= 2
        assert all(camera["inference"]["actual_fps"] < 20 for camera in app.health().values())

        # The feeds keep producing frames while the shared encoder is busy.
        # New values still match the original baseline, so no candidate pauses detection.
        for feed in app.feeds.values():
            feed.value += 1
        latest = {name: feed.value for name, feed in app.feeds.items()}
        app.wait_for(lambda: app.latest_values() == latest)
        boundary = len(app.encoder.values)
        app.wait_for(lambda: all(app.counts()[name] >= count + 3 for name, count in counts.items()))
        processed = app.encoder.values[boundary:]
        assert set(latest.values()) <= set(processed)
        # Only the single inference already in flight may finish with an older frame.
        assert sum(value not in latest.values() for value in processed) <= 1


def test_one_subscription_receives_independent_candidates_from_multiple_cameras(tmp_path):
    with RunningCameras(tmp_path, dict.fromkeys(["door", "room", "yard"]), fps=10) as app:
        with app.service.events.subscribe() as events:
            app.feeds["door"].value = 180
            app.feeds["room"].value = 220
            received = [events.get(timeout=3), events.get(timeout=3)]
            assert all(event.event_type == "candidate_change" for event in received)
            candidates = {event.data["camera"]: event.data for event in received}
            assert set(candidates) == {"door", "room"}
            assert len({candidate["candidate_id"] for candidate in candidates.values()}) == 2
            for name, candidate in candidates.items():
                assert candidate["episode_id"] == f"episode-{name}"
                assert len(candidate["frames"]) == 5

            health = app.health()
            assert health["yard"]["episode"]["pending_candidate_id"] is None
            before = health["yard"]["inference"]["frames_processed"]
            app.wait_for(lambda: app.counts()["yard"] >= before + 2)
            app.service.camera("door").acknowledge(
                candidates["door"]["candidate_id"],
                episode_id="episode-door",
                triggered_vlm=False,
                baseline_jpeg=app.service.camera("door").latest_jpeg(),
            )
            health = app.health()
            assert health["door"]["episode"]["pending_candidate_id"] is None
            assert (
                health["room"]["episode"]["pending_candidate_id"]
                == candidates["room"]["candidate_id"]
            )
