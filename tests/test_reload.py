"""Exercise explicit TOML reloads through the running HTTP service."""

import json
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from clip_service.config import load_config
from clip_service.errors import ServiceError
from clip_service.http_api import ServiceHTTPServer
from clip_service.service import ClipService
from tests.helpers import PixelEncoder, RunningService, VideoFeed, next_event
from tests.test_recovery import block_next_encode, jpeg_source


def configuration(cameras=None, *, detection=None, root="", extra="", frigate=""):
    if cameras is None:
        cameras = {"room": {"stream": "room_sub"}}
    settings = {
        "stable_seconds": 0.2,
        "candidate_frame_interval": 0.05,
        "inference_fps": 20,
        "ring_max_fps": 40,
        "reconnect_delay_seconds": 0.05,
    } | (detection or {})
    content = root + '\n[frigate]\nrtsp_base_url="rtsp://localhost:8554"\n' + frigate
    content += "\n[detection]\n"
    content += "".join(f"{key}={json.dumps(value)}\n" for key, value in settings.items())
    content += extra
    for name, values in cameras.items():
        content += f"\n[cameras.{name}]\n"
        content += "".join(f"{key}={json.dumps(value)}\n" for key, value in values.items())
    return content


class RunningReload(RunningService):
    def __init__(self, directory, content=None):
        self.path = directory / "service.toml"
        self.path.write_text(configuration() if content is None else content)
        self.feeds = defaultdict(VideoFeed)
        self.opened_urls = []
        self.encoder = PixelEncoder()

        def open_feed(url, settings):
            self.opened_urls.append(url)
            return self.feeds[urlsplit(url).path.rsplit("/", 1)[1]]

        self.service = ClipService(
            load_config(self.path, environ={}),
            config_path=self.path,
            capture_factory=open_feed,
            encoder=self.encoder,
        )
        self.server = ServiceHTTPServer(("127.0.0.1", 0), self.service)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.service.start()
        self.thread.start()
        self.wait_for(lambda: all(not camera["stale"] for camera in self.health().values()))
        return self

    def __exit__(self, *exc):
        for feed in self.feeds.values():
            feed.paused = False
        self.service.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def health(self):
        return self.request("GET", "/api/v1/health")[1]["cameras"]

    def reload(self, content):
        self.path.write_text(content)
        return self.request("POST", "/api/v1/config/reload")

    def begin(self, camera):
        episode_id = f"motion-{camera}"
        assert (
            self.request("POST", f"/api/v1/cameras/{camera}/episodes", {"episode_id": episode_id})[
                0
            ]
            == 201
        )
        assert (
            self.request(
                "PUT",
                f"/api/v1/cameras/{camera}/baseline",
                {"episode_id": episode_id, "source": {"type": "latest"}},
            )[0]
            == 200
        )

    def candidate(self, camera):
        self.wait_for(lambda: self.health()[camera]["episode"]["pending_candidate_id"] is not None)
        candidate_id = self.health()[camera]["episode"]["pending_candidate_id"]
        status, result = self.request("GET", f"/api/v1/candidates/{candidate_id}")
        assert status == 200
        return result


def test_reload_adds_camera_without_restarting_existing_session_or_loading_another_model(tmp_path):
    with RunningReload(tmp_path) as app:
        app.begin("room")
        original_encoder = app.service.encoder
        status, result = app.reload(
            configuration({"room": {"stream": "room_sub"}, "door": {"stream": "door_sub"}})
        )
        assert status == 200
        assert result["added"] == ["door"]
        assert result["updated"] == result["removed"] == result["restarted"] == []
        assert set(result["cameras"]) == {"room", "door"}
        app.wait_for(lambda: not app.health()["door"]["stale"])
        assert app.opened_urls.count("rtsp://localhost:8554/room_sub") == 1
        room = app.health()["room"]["episode"]
        assert room["episode_id"] == "motion-room" and room["baseline_ready"]
        assert app.service.encoder is original_encoder
        assert app.service.camera("door").encoder is original_encoder
        app.begin("door")
        app.feeds["door_sub"].value = 255
        assert app.candidate("door")["episode_id"] == "motion-door"


def test_unchanged_configuration_is_a_noop(tmp_path):
    with RunningReload(tmp_path) as app:
        app.begin("room")
        app.feeds["room_sub"].value = 255
        candidate = app.candidate("room")
        status, result = app.reload(configuration())
        assert status == 200
        assert (
            result["added"] == result["updated"] == result["removed"] == result["restarted"] == []
        )
        assert app.opened_urls == ["rtsp://localhost:8554/room_sub"]
        assert app.health()["room"]["episode"]["pending_candidate_id"] == candidate["candidate_id"]


@pytest.mark.parametrize(
    ("content", "error_location"),
    [
        ("[frigate", "TOML"),
        (configuration(detection={"inference_fps": 0}), "inference_fps"),
        (
            configuration(
                {
                    "room": {"stream": "room_sub", "inference_fps": 10},
                    "door": {"stream": "door_sub", "inference_fps": -1},
                }
            ),
            "door.inference_fps",
        ),
    ],
    ids=["invalid-toml", "invalid-global-rate", "invalid-camera-override"],
)
def test_invalid_reload_retains_all_previous_settings_and_session(
    tmp_path, content, error_location
):
    with RunningReload(tmp_path) as app:
        app.begin("room")
        before = app.health()["room"]["inference"]["frames_processed"]
        status, result = app.reload(content)
        assert status == 400
        assert error_location in result["error"]["message"]
        assert set(app.health()) == {"room"}
        room = app.health()["room"]
        assert room["inference"]["target_fps"] == 20
        assert room["episode"]["episode_id"] == "motion-room"
        assert room["episode"]["baseline_ready"]
        assert app.opened_urls == ["rtsp://localhost:8554/room_sub"]
        app.wait_for(lambda: app.health()["room"]["inference"]["frames_processed"] >= before + 2)


@pytest.mark.parametrize(
    "content",
    [
        configuration(root='data_dir="elsewhere"'),
        configuration(root="retention_hours=1"),
        configuration(extra='\n[model]\ndevice="cpu"\n'),
        configuration(extra="\n[server]\nport=18081\n"),
    ],
    ids=["data-directory", "retention", "model", "server"],
)
def test_changes_to_service_settings_require_restart_without_disrupting_cameras(tmp_path, content):
    with RunningReload(tmp_path) as app:
        app.begin("room")
        status, result = app.reload(content)
        assert status == 409
        assert "restart" in result["error"]["message"].lower()
        assert app.health()["room"]["episode"]["baseline_ready"]
        assert app.opened_urls == ["rtsp://localhost:8554/room_sub"]


def test_global_rate_change_reschedules_inheriting_cameras_and_keeps_explicit_override(tmp_path):
    cameras = {"room": {"stream": "room_sub"}, "door": {"stream": "door_sub", "inference_fps": 8}}
    with RunningReload(tmp_path, configuration(cameras, detection={"inference_fps": 4})) as app:
        app.begin("room")
        app.begin("door")
        app.wait_for(lambda: app.health()["room"]["inference"]["frames_processed"] >= 4)
        before = {name: row["inference"]["frames_processed"] for name, row in app.health().items()}
        status, result = app.reload(configuration(cameras, detection={"inference_fps": 8}))
        assert status == 200 and result["updated"] == ["room"]
        assert result["restarted"] == []
        app.wait_for(
            lambda: all(
                row["inference"]["frames_processed"] >= before[name] + 6
                for name, row in app.health().items()
            )
        )
        health = app.health()
        counts = [
            row["inference"]["frames_processed"] - before[name] for name, row in health.items()
        ]
        assert max(counts) - min(counts) <= 2
        assert all(row["inference"]["target_fps"] == 8 for row in health.values())
        assert all(row["episode"]["baseline_ready"] for row in health.values())
        assert len(app.opened_urls) == 2


def test_detection_parameter_reload_keeps_pending_images_deadline_and_baseline(tmp_path):
    with RunningReload(tmp_path) as app:
        app.begin("room")
        app.feeds["room_sub"].value = 255
        candidate = app.candidate("room")
        images = [app.request("GET", frame["url"]) for frame in candidate["frames"]]
        status, result = app.reload(
            configuration(
                detection={
                    "stable_seconds": 0.4,
                    "similarity_threshold": 0.8,
                    "ack_timeout_seconds": 1,
                }
            )
        )
        assert status == 200 and result["updated"] == ["room"]
        assert result["restarted"] == []
        assert app.request("GET", f"/api/v1/candidates/{candidate['candidate_id']}") == (
            200,
            candidate,
        )
        assert [app.request("GET", frame["url"]) for frame in candidate["frames"]] == images
        state = app.health()["room"]["episode"]
        assert state["episode_id"] == "motion-room" and state["baseline_ready"]
        assert state["pending_candidate_id"] == candidate["candidate_id"]
        assert app.opened_urls == ["rtsp://localhost:8554/room_sub"]
        assert (
            app.request(
                "POST",
                f"/api/v1/candidates/{candidate['candidate_id']}/ack",
                {"episode_id": "motion-room", "triggered_vlm": False},
            )[0]
            == 200
        )
        next_candidate = app.candidate("room")
        assert next_candidate["candidate_id"] != candidate["candidate_id"]
        assert next_candidate["ack_deadline_at"] - next_candidate["created_at"] == 1


def test_parameter_reload_restarts_an_unconfirmed_change_window(tmp_path):
    with RunningReload(tmp_path, configuration(detection={"stable_seconds": 1.0})) as app:
        app.begin("room")
        app.feeds["room_sub"].value = 255
        app.wait_for(lambda: app.encoder.values.count(255) >= 6)
        assert app.health()["room"]["episode"]["pending_candidate_id"] is None
        before_reload = time.time()
        assert (
            app.reload(
                configuration(detection={"stable_seconds": 1.0, "similarity_threshold": 0.8})
            )[0]
            == 200
        )
        candidate = app.candidate("room")
        # One freshly captured frame may predate the reload request by a capture interval.
        assert candidate["started_at"] >= before_reload - 0.05


@pytest.mark.parametrize("change", ["stream_changed", "disabled", "removed"])
def test_retiring_camera_ends_session_and_cancels_candidate_while_other_camera_continues(
    tmp_path, change
):
    cameras = {"room": {"stream": "room_sub"}, "door": {"stream": "door_sub"}}
    with RunningReload(tmp_path, configuration(cameras)) as app:
        app.begin("room")
        app.begin("door")
        app.feeds["room_sub"].value = 255
        candidate = app.candidate("room")
        retired_camera = app.service.camera("room")
        before = app.health()["door"]["inference"]["frames_processed"]
        connection, stream = app.subscribe()
        try:
            if change == "stream_changed":
                cameras["room"] = {"stream": "room_new"}
            elif change == "disabled":
                cameras["room"] = {"stream": "room_sub", "enabled": False}
            else:
                del cameras["room"]
            status, result = app.reload(configuration(cameras))
            assert status == 200
            event = next_event(stream)
            assert event["type"] == "episode_ended"
            assert event["data"] == {
                "camera": "room",
                "episode_id": "motion-room",
                "reason": change,
            }
            if change == "stream_changed":
                assert result["restarted"] == ["room"]
                assert app.health()["room"]["episode"]["active"] is False
                assert app.health()["room"]["episode"]["baseline_ready"] is False
            else:
                assert result["removed"] == ["room"]
                assert "room" not in app.health()
            status, old = app.request("GET", f"/api/v1/candidates/{candidate['candidate_id']}")
            assert status == 200 and old["status"] == "cancelled"
            assert app.request("GET", old["frames"][0]["url"])[0] == 200
            assert app.request(
                "POST",
                f"/api/v1/candidates/{candidate['candidate_id']}/ack",
                {"episode_id": "motion-room", "triggered_vlm": True},
            )[0] in {404, 409}
            with pytest.raises(ServiceError) as error:
                retired_camera.start_episode("new-motion")
            assert error.value.status == 409
            app.wait_for(
                lambda: app.health()["door"]["inference"]["frames_processed"] >= before + 3
            )
            assert app.opened_urls.count("rtsp://localhost:8554/door_sub") == 1
            assert app.health()["door"]["episode"]["episode_id"] == "motion-door"
        finally:
            stream.close()
            connection.close()


@pytest.mark.parametrize("change", ["endpoint", "credentials"])
def test_frigate_connection_changes_restart_all_affected_camera_sessions(tmp_path, change):
    cameras = {"room": {"stream": "room_sub"}, "door": {"stream": "door_sub"}}
    with RunningReload(tmp_path, configuration(cameras)) as app:
        app.begin("room")
        app.begin("door")
        if change == "endpoint":
            content = configuration(cameras).replace("localhost:8554", "frigate.lan:8554")
            expected_base = "rtsp://frigate.lan:8554"
        else:
            content = configuration(cameras, frigate='username="operator"\npassword="example"\n')
            expected_base = "rtsp://operator:example@localhost:8554"
        status, result = app.reload(content)
        assert status == 200 and set(result["restarted"]) == {"room", "door"}
        app.wait_for(
            lambda: all(f"{expected_base}/{name}_sub" in app.opened_urls for name in cameras)
        )
        assert all(not row["episode"]["active"] for row in app.health().values())


def test_new_camera_with_unavailable_stream_reports_its_failure_and_keeps_existing_camera_running(
    tmp_path,
):
    with RunningReload(tmp_path) as app:
        app.begin("room")
        app.feeds["down_sub"].connected = False
        before = app.health()["room"]["inference"]["frames_processed"]
        status, result = app.reload(
            configuration({"room": {"stream": "room_sub"}, "down": {"stream": "down_sub"}})
        )
        assert status == 200 and result["added"] == ["down"]
        assert result["cameras"]["down"]["stale"]
        assert not result["cameras"]["down"]["connected"]
        app.wait_for(lambda: "rtsp://localhost:8554/down_sub" in app.opened_urls)
        app.wait_for(lambda: app.health()["room"]["inference"]["frames_processed"] >= before + 2)
        assert app.request("GET", "/api/v1/health")[1]["status"] == "degraded"


def test_ack_encoding_during_stream_switch_cannot_modify_replacement_camera(tmp_path):
    with RunningReload(tmp_path) as app:
        app.begin("room")
        app.feeds["room_sub"].value = 255
        candidate = app.candidate("room")
        with ThreadPoolExecutor(max_workers=1) as pool:
            with block_next_encode(app.encoder) as entered:
                old_ack = pool.submit(
                    app.request,
                    "POST",
                    f"/api/v1/candidates/{candidate['candidate_id']}/ack",
                    {
                        "episode_id": "motion-room",
                        "triggered_vlm": True,
                        "baseline": jpeg_source(255),
                    },
                )
                assert entered.wait(1)
                assert app.reload(configuration({"room": {"stream": "replacement"}}))[0] == 200
                app.wait_for(lambda: not app.health()["room"]["stale"])
                app.begin("room")
            assert old_ack.result()[0] in {404, 409}
        state = app.health()["room"]["episode"]
        assert state["episode_id"] == "motion-room" and state["baseline_ready"]
        assert state["pending_candidate_id"] is None
        assert (
            app.request("GET", f"/api/v1/candidates/{candidate['candidate_id']}")[1]["status"]
            == "cancelled"
        )
        app.feeds["replacement"].value = 255
        assert app.candidate("room")["candidate_id"] != candidate["candidate_id"]


def test_service_can_reload_from_no_cameras_and_return_to_no_cameras(tmp_path):
    with RunningReload(tmp_path, configuration({})) as app:
        assert app.health() == {}
        assert app.reload(configuration())[0] == 200
        app.wait_for(lambda: not app.health()["room"]["stale"])
        app.begin("room")
        assert app.reload(configuration({}))[0] == 200
        assert app.health() == {}


def test_failed_candidate_cancellation_reports_reload_failure_and_can_be_retried(
    tmp_path, monkeypatch
):
    with RunningReload(tmp_path) as app:
        app.begin("room")
        app.feeds["room_sub"].value = 255
        candidate = app.candidate("room")
        original_replace = Path.replace

        def replace(source, target):
            if source.parent.name == candidate["candidate_id"] and source.name == "manifest.tmp":
                raise OSError("simulated disk write failure")
            return original_replace(source, target)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "replace", replace)
            status, result = app.reload(configuration({}))
            assert status == 503
            assert result["error"]["code"] == "reload_failed"
        health = app.request("GET", "/api/v1/health")[1]
        assert health["status"] == "degraded" and health["reload_error"]
        assert (
            health["cameras"]["room"]["episode"]["pending_candidate_id"]
            == candidate["candidate_id"]
        )
        assert (
            app.request("GET", f"/api/v1/candidates/{candidate['candidate_id']}")[1]["status"]
            == "pending"
        )
        assert app.request("POST", "/api/v1/config/reload")[0] == 200
        health = app.request("GET", "/api/v1/health")[1]
        assert health["cameras"] == {}
        assert health["reload_error"] is None
        assert (
            app.request("GET", f"/api/v1/candidates/{candidate['candidate_id']}")[1]["status"]
            == "cancelled"
        )


def test_baseline_request_reading_old_candidate_cannot_write_a_replacement_camera(
    tmp_path, monkeypatch
):
    with RunningReload(tmp_path) as app:
        app.begin("room")
        app.feeds["room_sub"].value = 255
        candidate = app.candidate("room")
        assert (
            app.request(
                "POST",
                f"/api/v1/candidates/{candidate['candidate_id']}/ack",
                {
                    "episode_id": "motion-room",
                    "triggered_vlm": True,
                    "baseline": jpeg_source(255),
                },
            )[0]
            == 200
        )
        entered = threading.Event()
        release = threading.Event()
        original_read = Path.read_bytes

        def read_bytes(path):
            if path.parent.name == candidate["candidate_id"] and path.name == "frame-0.jpg":
                entered.set()
                assert release.wait(3), "test did not release candidate image read"
            return original_read(path)

        with ThreadPoolExecutor(max_workers=1) as pool:
            with monkeypatch.context() as patch:
                patch.setattr(Path, "read_bytes", read_bytes)
                old_request = pool.submit(
                    app.request,
                    "PUT",
                    "/api/v1/cameras/room/baseline",
                    {
                        "episode_id": "motion-room",
                        "source": {
                            "type": "candidate",
                            "candidate_id": candidate["candidate_id"],
                            "frame_index": 0,
                        },
                    },
                )
                try:
                    assert entered.wait(1)
                    assert app.reload(configuration({"room": {"stream": "replacement"}}))[0] == 200
                    app.wait_for(lambda: not app.health()["room"]["stale"])
                    app.begin("room")
                finally:
                    release.set()
                assert old_request.result()[0] in {404, 409}
        state = app.health()["room"]["episode"]
        assert state["episode_id"] == "motion-room" and state["baseline_ready"]
        assert state["pending_candidate_id"] is None
        app.feeds["replacement"].value = 255
        assert app.candidate("room")["candidate_id"] != candidate["candidate_id"]


def test_reader_shutdown_timeout_reports_reload_failure_and_can_be_retried(tmp_path):
    settings = {"open_timeout_seconds": 0.01, "read_timeout_seconds": 0.01}
    with RunningReload(tmp_path, configuration(detection=settings)) as app:
        old_feed = app.feeds["room_sub"]
        old_feed.paused = True
        time.sleep(0.05)  # Let the external decoder enter its blocked read.
        content = configuration({"room": {"stream": "replacement"}}, detection=settings)
        try:
            status, result = app.reload(content)
            assert status == 503 and result["error"]["code"] == "reload_failed"
            health = app.request("GET", "/api/v1/health")[1]
            assert health["reload_error"] and health["status"] == "degraded"
        finally:
            old_feed.paused = False
        assert app.request("POST", "/api/v1/config/reload")[0] == 200
        assert app.request("GET", "/api/v1/health")[1]["reload_error"] is None
