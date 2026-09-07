import base64
import io
import time

from PIL import Image

from tests.helpers import RunningService, next_event


def test_agent_receives_candidate_fetches_five_frames_and_updates_baseline(tmp_path):
    with RunningService(tmp_path) as app:
        connection, stream = app.subscribe()
        try:
            app.start_episode()
            app.feed.value = 255
            event = next_event(stream)
            assert event["type"] == "candidate_change"
            candidate = event["data"]
            assert candidate["camera"] == "room"
            assert candidate["episode_id"] == "motion-1"
            assert len(candidate["frames"]) == 5
            for frame in candidate["frames"]:
                status, jpeg = app.request("GET", frame["url"])
                assert status == 200 and jpeg.startswith(b"\xff\xd8")
            encoded = len(app.encoder.values)
            time.sleep(0.15)
            assert (
                len(app.encoder.values) == encoded
            )  # Waiting for the agent doesn't run inference.
            status, result = app.request(
                "POST",
                f"/api/v1/candidates/{candidate['candidate_id']}/ack",
                {
                    "episode_id": "motion-1",
                    "triggered_vlm": True,
                    "baseline": {
                        "type": "candidate",
                        "candidate_id": candidate["candidate_id"],
                        "frame_index": 4,
                    },
                },
            )
            assert status == 200 and result["status"] == "acknowledged"
            app.wait_for(lambda: len(app.encoder.values) > encoded + 1)
            health = app.request("GET", "/api/v1/health")[1]
            assert health["cameras"]["room"]["episode"]["pending_candidate_id"] is None
        finally:
            stream.close()
            connection.close()


def test_waiting_for_baseline_and_disconnected_video_do_not_run_detection(tmp_path):
    with RunningService(tmp_path) as app:
        assert (
            app.request("POST", "/api/v1/cameras/room/episodes", {"episode_id": "motion-1"})[0]
            == 201
        )
        time.sleep(0.15)
        assert app.encoder.values == []
        app.start_episode()
        app.feed.connected = False
        app.wait_for(lambda: app.request("GET", "/api/v1/health")[1]["cameras"]["room"]["stale"])
        status, _ = app.request(
            "PUT",
            "/api/v1/cameras/room/baseline",
            {
                "episode_id": "motion-1",
                "source": {"type": "latest"},
            },
        )
        assert status == 503
        time.sleep(0.05)  # Allow an already-running encode to finish.
        count = len(app.encoder.values)
        time.sleep(0.15)
        assert len(app.encoder.values) == count
        connection, stream = app.subscribe()
        try:
            app.feed.value = 255
            app.feed.connected = True
            candidate = next_event(stream)["data"]
            assert candidate["camera"] == "room"
            assert len(candidate["frames"]) == 5
        finally:
            stream.close()
            connection.close()


def test_uploaded_baseline_and_stopping_a_session_through_http(tmp_path):
    jpeg = io.BytesIO()
    Image.new("RGB", (24, 24), (0, 0, 0)).save(jpeg, format="JPEG")
    with RunningService(tmp_path) as app:
        app.request("POST", "/api/v1/cameras/room/episodes", {"episode_id": "motion-1"})
        status, result = app.request(
            "PUT",
            "/api/v1/cameras/room/baseline",
            {
                "episode_id": "motion-1",
                "source": {
                    "type": "jpeg_base64",
                    "data": base64.b64encode(jpeg.getvalue()).decode(),
                },
            },
        )
        assert status == 200 and result["baseline_ready"] is True
        status, result = app.request("DELETE", "/api/v1/cameras/room/episodes/motion-1")
        assert status == 200 and result["state"] == "stopped"
        episode = app.request("GET", "/api/v1/health")[1]["cameras"]["room"]["episode"]
        assert episode["active"] is False and episode["baseline_ready"] is False


def test_connected_camera_without_new_frames_is_not_reencoded(tmp_path):
    with RunningService(tmp_path) as app:
        app.start_episode()
        app.feed.paused = True
        time.sleep(0.1)  # Finish the last frame already in flight.
        count = len(app.encoder.values)
        time.sleep(0.15)
        assert len(app.encoder.values) == count
        camera = app.request("GET", "/api/v1/health")[1]["cameras"]["room"]
        assert camera["connected"] and not camera["stale"]
