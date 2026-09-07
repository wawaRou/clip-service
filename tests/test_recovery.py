import base64
import io
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest
from PIL import Image

from tests.helpers import RunningService


def episode(app):
    return app.request("GET", "/api/v1/health")[1]["cameras"]["room"]["episode"]


def wait_candidate(app, previous_id=None):
    app.wait_for(lambda: episode(app)["pending_candidate_id"] not in {None, previous_id})
    candidate_id = episode(app)["pending_candidate_id"]
    status, candidate = app.request("GET", f"/api/v1/candidates/{candidate_id}")
    assert status == 200
    return candidate


def ack(app, candidate, **extra):
    return app.request(
        "POST",
        f"/api/v1/candidates/{candidate['candidate_id']}/ack",
        {"episode_id": "motion-1", "triggered_vlm": True} | extra,
    )


def jpeg_source(value):
    output = io.BytesIO()
    Image.new("RGB", (24, 24), (value, value, value)).save(output, format="JPEG")
    return {"type": "jpeg_base64", "data": base64.b64encode(output.getvalue()).decode()}


def test_missed_sse_notification_can_be_reconciled_over_http(tmp_path):
    with RunningService(tmp_path) as app:
        connection, stream = app.subscribe()
        stream.close()
        connection.close()
        app.start_episode()
        app.feed.value = 255
        candidate = wait_candidate(app)

        for endpoint in ("/api/v1/candidates", "/api/v1/candidates?status=pending"):
            status, result = app.request("GET", endpoint)
            assert status == 200
            assert result["candidates"] == [candidate]
        assert ack(app, candidate)[0] == 200
        assert app.request("GET", "/api/v1/candidates")[1] == {"candidates": []}


def test_timeout_keeps_frames_and_baseline_while_late_ack_cannot_clear_new_candidate(tmp_path):
    with RunningService(tmp_path, settings="ack_timeout_seconds=0.15\n") as app:
        app.start_episode()
        app.feed.value = 255
        first = wait_candidate(app)
        second = wait_candidate(app, first["candidate_id"])

        status, expired = app.request("GET", f"/api/v1/candidates/{first['candidate_id']}")
        assert status == 200 and expired["status"] == "expired"
        assert expired["expires_at"] - expired["created_at"] == 24 * 3600
        for frame in expired["frames"]:
            status, jpeg = app.request("GET", frame["url"])
            assert status == 200 and jpeg.startswith(b"\xff\xd8")
        assert ack(app, first, baseline=jpeg_source(255))[0] == 409
        assert episode(app)["pending_candidate_id"] == second["candidate_id"]


def test_duplicate_ack_returns_record_without_reapplying_baseline_even_after_stop(tmp_path):
    with RunningService(tmp_path) as app:
        app.start_episode()
        app.feed.value = 255
        candidate = wait_candidate(app)
        status, acknowledged = ack(app, candidate, baseline=jpeg_source(255))
        assert status == 200 and acknowledged["status"] == "acknowledged"

        # A retry must not resolve this now-invalid source or mutate the live baseline.
        assert ack(app, candidate, baseline={"type": "not-a-source"}) == (200, acknowledged)
        assert ack(app, candidate, triggered_vlm=False)[0] == 409
        assert ack(app, candidate, episode_id="another-episode")[0] == 409
        assert app.request("DELETE", "/api/v1/cameras/room/episodes/motion-1")[0] == 200
        assert ack(app, candidate, baseline={"type": "not-a-source"}) == (200, acknowledged)


def test_restart_cancels_old_pending_and_agent_can_establish_new_session(tmp_path):
    with RunningService(tmp_path) as app:
        app.start_episode()
        app.feed.value = 255
        previous = wait_candidate(app)

    with RunningService(tmp_path) as app:
        state = episode(app)
        assert state["episode_id"] is None
        assert state["active"] is False
        assert state["baseline_ready"] is False
        assert state["pending_candidate_id"] is None
        assert app.request("GET", "/api/v1/candidates")[1] == {"candidates": []}
        status, old = app.request("GET", f"/api/v1/candidates/{previous['candidate_id']}")
        assert status == 200 and old["status"] == "cancelled"
        assert app.request("GET", old["frames"][0]["url"])[0] == 200
        app.start_episode()
        app.feed.value = 255
        assert wait_candidate(app)["candidate_id"] != previous["candidate_id"]


@contextmanager
def block_next_encode(encoder):
    """Hold one request at the external model boundary while another request completes."""
    original = encoder.encode_jpeg
    entered = threading.Event()
    release = threading.Event()
    claim = threading.Lock()

    def encode(jpeg):
        with claim:
            first = not entered.is_set()
            entered.set()
        if first:
            assert release.wait(3), "test did not release the blocked encoder"
        return original(jpeg)

    encoder.encode_jpeg = encode
    try:
        yield entered
    finally:
        release.set()
        encoder.encode_jpeg = original


def set_baseline(app, value):
    return app.request(
        "PUT",
        "/api/v1/cameras/room/baseline",
        {"episode_id": "motion-1", "source": jpeg_source(value)},
    )


def test_encoding_ack_cannot_overwrite_a_newer_baseline(tmp_path):
    with RunningService(tmp_path) as app:
        app.start_episode()
        app.feed.value = 255
        candidate = wait_candidate(app)
        with ThreadPoolExecutor(max_workers=1) as pool:
            with block_next_encode(app.encoder) as entered:
                slow_ack = pool.submit(ack, app, candidate, baseline=jpeg_source(255))
                assert entered.wait(1)
                assert set_baseline(app, 0)[0] == 200
            assert slow_ack.result()[0] == 409
        assert episode(app)["pending_candidate_id"] == candidate["candidate_id"]
        assert ack(app, candidate)[0] == 200
        # White still differs from the winning black baseline, so detection resumes.
        assert wait_candidate(app, candidate["candidate_id"])["status"] == "pending"


def test_encoding_baseline_cannot_overwrite_a_completed_ack(tmp_path):
    with RunningService(tmp_path) as app:
        app.start_episode()
        app.feed.value = 255
        candidate = wait_candidate(app)
        with ThreadPoolExecutor(max_workers=1) as pool:
            with block_next_encode(app.encoder) as entered:
                slow_baseline = pool.submit(set_baseline, app, 255)
                assert entered.wait(1)
                assert ack(app, candidate, baseline=jpeg_source(0))[0] == 200
            assert slow_baseline.result()[0] == 409
        assert wait_candidate(app, candidate["candidate_id"])["status"] == "pending"


@pytest.mark.parametrize("operation", ["ack", "baseline"])
def test_encoding_request_cannot_restore_a_stopped_session(tmp_path, operation):
    with RunningService(tmp_path) as app:
        app.start_episode()
        app.feed.value = 255
        candidate = wait_candidate(app)
        with ThreadPoolExecutor(max_workers=1) as pool:
            with block_next_encode(app.encoder) as entered:
                slow_request = (
                    pool.submit(ack, app, candidate, baseline=jpeg_source(255))
                    if operation == "ack"
                    else pool.submit(set_baseline, app, 255)
                )
                assert entered.wait(1)
                assert app.request("DELETE", "/api/v1/cameras/room/episodes/motion-1")[0] == 200
                # Even reusing the episode ID does not make an old in-flight write valid.
                assert (
                    app.request(
                        "POST", "/api/v1/cameras/room/episodes", {"episode_id": "motion-1"}
                    )[0]
                    == 201
                )
            assert slow_request.result()[0] == 409
        state = episode(app)
        assert state["active"] is True
        assert state["baseline_ready"] is False
        assert state["pending_candidate_id"] is None
        assert (
            app.request("GET", f"/api/v1/candidates/{candidate['candidate_id']}")[1]["status"]
            == "cancelled"
        )


def test_concurrent_identical_acknowledgements_both_return_the_saved_result(tmp_path):
    with RunningService(tmp_path) as app:
        app.start_episode()
        app.feed.value = 255
        candidate = wait_candidate(app)
        with ThreadPoolExecutor(max_workers=1) as pool:
            with block_next_encode(app.encoder) as entered:
                slow_ack = pool.submit(ack, app, candidate, baseline=jpeg_source(255))
                assert entered.wait(1)
                result = ack(app, candidate, baseline=jpeg_source(255))
                assert result[0] == 200
            assert slow_ack.result() == result
        assert episode(app)["pending_candidate_id"] is None


def test_ack_that_finishes_encoding_after_timeout_cannot_install_its_baseline(tmp_path):
    with RunningService(tmp_path, settings="ack_timeout_seconds=0.15\n") as app:
        app.start_episode()
        app.feed.value = 255
        candidate = wait_candidate(app)
        with ThreadPoolExecutor(max_workers=1) as pool:
            with block_next_encode(app.encoder) as entered:
                slow_ack = pool.submit(ack, app, candidate, baseline=jpeg_source(255))
                assert entered.wait(1)
                app.wait_for(
                    lambda: (
                        app.request("GET", f"/api/v1/candidates/{candidate['candidate_id']}")[1][
                            "status"
                        ]
                        == "expired"
                    )
                )
            assert slow_ack.result()[0] == 409
        assert wait_candidate(app, candidate["candidate_id"])["status"] == "pending"


def test_manifest_failure_leaves_pending_and_baseline_unchanged_for_retry(tmp_path, monkeypatch):
    with RunningService(tmp_path) as app:
        app.start_episode()
        app.feed.value = 255
        candidate = wait_candidate(app)
        original_replace = Path.replace

        def replace(source, target):
            if source.parent.name == candidate["candidate_id"] and source.name == "manifest.tmp":
                raise OSError("simulated disk write failure")
            return original_replace(source, target)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "replace", replace)
            assert ack(app, candidate, baseline=jpeg_source(255))[0] == 500
        assert episode(app)["pending_candidate_id"] == candidate["candidate_id"]
        assert (
            app.request("GET", f"/api/v1/candidates/{candidate['candidate_id']}")[1]["status"]
            == "pending"
        )
        assert ack(app, candidate)[0] == 200
        # The failed acknowledgement must not have installed the white baseline.
        assert wait_candidate(app, candidate["candidate_id"])["status"] == "pending"


def test_removed_pending_files_do_not_block_timeout_recovery(tmp_path):
    with RunningService(tmp_path, settings="ack_timeout_seconds=0.2\n") as app:
        app.start_episode()
        app.feed.value = 255
        candidate = wait_candidate(app)
        shutil.rmtree(app.service.config.data_dir / "candidates" / candidate["candidate_id"])

        next_candidate = wait_candidate(app, candidate["candidate_id"])

        assert next_candidate["status"] == "pending"
        assert app.request("GET", f"/api/v1/candidates/{candidate['candidate_id']}")[0] == 404
        assert ack(app, candidate, baseline=jpeg_source(255))[0] == 404


def test_missing_pending_files_do_not_prevent_stopping_session(tmp_path):
    with RunningService(tmp_path) as app:
        app.start_episode()
        app.feed.value = 255
        candidate = wait_candidate(app)
        shutil.rmtree(app.service.config.data_dir / "candidates" / candidate["candidate_id"])

        assert app.request("DELETE", "/api/v1/cameras/room/episodes/motion-1")[0] == 200

        assert episode(app)["active"] is False
        assert episode(app)["pending_candidate_id"] is None
        assert app.request("GET", f"/api/v1/candidates/{candidate['candidate_id']}")[0] == 404


def test_failed_candidate_publish_is_visible_in_health_until_storage_recovers(
    tmp_path, monkeypatch
):
    with RunningService(tmp_path) as app:
        app.start_episode()
        original_replace = Path.replace

        def fail_candidate_publish(source, target):
            if source.name.startswith(".candidate-") and source.name.endswith(".tmp"):
                raise OSError("private filesystem failure details")
            return original_replace(source, target)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "replace", fail_candidate_publish)
            app.feed.value = 255
            app.wait_for(lambda: app.request("GET", "/api/v1/health")[1]["status"] == "degraded")
            status, health = app.request("GET", "/api/v1/health")
            assert status == 200
            camera = health["cameras"]["room"]
            assert camera["storage_error"] == "candidate storage write failed"
            assert camera["inference_error"] is None
            assert camera["episode"]["pending_candidate_id"] is None

        assert wait_candidate(app)["status"] == "pending"
        _, health = app.request("GET", "/api/v1/health")
        assert health["status"] == "ok"
        assert health["cameras"]["room"]["storage_error"] is None
