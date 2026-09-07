"""Opt-in native acceptance against one real Frigate stream and an offline CLIP model.

Three logical cameras duplicate the same source; this is not three-camera field testing.
Only the JSON report survives: temporary configuration, logs and candidate images are removed.
"""

import argparse
import base64
import http.client
import io
import json
import math
import os
import platform
import queue
import socket
import subprocess
import sys
import tempfile
import threading
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def request(port, method, path, body=None) -> Any:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    try:
        connection.request(
            method,
            path,
            None if body is None else json.dumps(body),
            {} if body is None else {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        content = response.read()
        content_type = response.getheader("Content-Type", "")
        assert response.status in {200, 201}, (method, path, response.status, content)
        if "json" in content_type:
            return json.loads(content)
        assert content_type == "image/jpeg" and content.startswith(b"\xff\xd8"), content_type
        return content
    finally:
        connection.close()


def configure(args, path, port, cameras, threshold):
    quote = json.dumps
    path.write_text(
        f'data_dir = "data"\n[server]\nhost = "127.0.0.1"\nport = {port}\n'
        f"[model]\npath = {quote(str(args.model.resolve()))}\ndevice = {quote(args.device)}\n"
        f"precision = {quote(args.precision)}\n"
        f"[frigate]\nrtsp_base_url = {quote(args.frigate_url)}\n"
        f"[detection]\ninference_fps = {args.fps}\nsimilarity_threshold = {threshold}\n"
        "stable_seconds = 0.3\ncandidate_frame_interval = 0.075\n"
        + ("[cameras]\n" if not cameras else "")
        + "".join(f"[cameras.{name}]\nstream = {quote(args.stream)}\n" for name in cameras),
        encoding="utf-8",
    )


def wait_ready(port, cameras, process):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        assert process.poll() is None, "service exited before cameras became ready"
        try:
            health = request(port, "GET", "/api/v1/health")
            if all(
                name in health["cameras"] and not health["cameras"][name]["stale"]
                for name in cameras
            ):
                return
        except (ConnectionError, OSError):
            pass
        time.sleep(0.2)
    raise TimeoutError("Frigate streams did not become fresh within 60 seconds")


def percentiles(values):
    return {
        "samples": len(values),
        **{
            key: float(np.percentile(values, percentile)) if values else None
            for key, percentile in (("p50", 50), ("p95", 95), ("max", 100))
        },
    }


def measure(port, cameras, seconds, warmup):
    time.sleep(warmup)
    start = time.monotonic()
    before = request(port, "GET", "/api/v1/health")["cameras"]
    observations = {name: {"processing": [], "latency": [], "stale_samples": 0} for name in cameras}
    counts = {name: before[name]["inference"]["frames_processed"] for name in cameras}
    health_samples = 0
    while time.monotonic() - start < seconds:
        time.sleep(0.2)
        after = request(port, "GET", "/api/v1/health")["cameras"]
        health_samples += 1
        for name in cameras:
            stats, entry = after[name]["inference"], observations[name]
            episode = after[name]["episode"]
            assert episode["active"] and episode["baseline_ready"], episode
            assert episode["pending_candidate_id"] is None, episode
            entry["stale_samples"] += int(after[name]["stale"])
            assert after[name]["inference_error"] is None, after[name]
            if stats["frames_processed"] != counts[name]:
                counts[name] = stats["frames_processed"]
                entry["processing"].append(stats["last_processing_seconds"])
                entry["latency"].append(stats["last_frame_latency_seconds"])
    elapsed = time.monotonic() - start
    result = {}
    for name, entry in observations.items():
        completed = counts[name] - before[name]["inference"]["frames_processed"]
        assert completed > 0, f"{name} completed no inference"
        result[name] = {
            "target_fps": after[name]["inference"]["target_fps"],
            "actual_fps": completed / elapsed,
            "frames_processed_delta": completed,
            "stale_health_samples": entry["stale_samples"],
            "processing_seconds": percentiles(entry["processing"]),
            "receive_to_encoding_seconds": percentiles(entry["latency"]),
        }
    return {"elapsed_seconds": elapsed, "health_samples": health_samples, "cameras": result}


def consume_events(response, messages):
    event = {}
    try:
        while line := response.readline():
            line = line.decode().strip()
            if not line and event:
                messages.put((event, time.time()))
                event = {}
            elif line.startswith("event: "):
                event["type"] = line[7:]
            elif line.startswith("data: "):
                event["data"] = json.loads(line[6:])
    except (OSError, ValueError) as error:
        messages.put(error)


def receive(messages, event_type, cameras):
    received = {}
    deadline = time.monotonic() + 45
    while set(received) != set(cameras):
        item = messages.get(timeout=max(0.01, deadline - time.monotonic()))
        if isinstance(item, Exception):
            raise item
        event, received_at = item
        if event.get("type") == event_type and event["data"]["camera"] in cameras:
            received.setdefault(event["data"]["camera"], (event["data"], received_at))
        if time.monotonic() >= deadline:
            raise TimeoutError(f"missing {event_type}: {set(cameras) - set(received)}")
    return received


def verify_candidates(port, received, cameras):
    pending = request(port, "GET", "/api/v1/candidates?status=pending")["candidates"]
    assert {item["candidate_id"] for item in pending} == {
        record["candidate_id"] for record, _ in received.values()
    }
    results = {}
    for name, (record, received_at) in received.items():
        assert record["episode_id"] == f"candidate-{len(cameras)}-{name}"
        assert len(record["frames"]) == 5
        for frame in record["frames"]:
            image = request(port, "GET", frame["url"])
            with Image.open(io.BytesIO(image)) as decoded:
                decoded.verify()
        candidate_id = record["candidate_id"]
        ack = {
            "episode_id": record["episode_id"],
            "triggered_vlm": False,
            "baseline": {"type": "candidate", "candidate_id": candidate_id, "frame_index": 4},
        }
        acknowledged = request(port, "POST", f"/api/v1/candidates/{candidate_id}/ack", ack)
        assert acknowledged["status"] == "acknowledged"
        assert request(port, "POST", f"/api/v1/candidates/{candidate_id}/ack", ack) == acknowledged
        assert request(port, "GET", f"/api/v1/candidates/{candidate_id}") == acknowledged
        results[name] = {
            "candidate_id": candidate_id,
            "started_at": record["started_at"],
            "sse_received_at": received_at,
            "receive_to_sse_seconds": received_at - record["started_at"],
            "five_jpegs_readable": True,
            "pending_reconciled": True,
            "ack_idempotent": True,
        }
    return results


def run(args, directory, port, report):
    path = directory / "service.toml"
    cameras = ["camera_1"]
    configure(args, path, port, cameras, -1)
    # Explicit CLI inputs take precedence; only optional Frigate auth is inherited.
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("CLIP_") or key in {"CLIP_FRIGATE_USERNAME", "CLIP_FRIGATE_PASSWORD"}
    }
    environment["HF_HUB_OFFLINE"] = "1"
    sse = None
    with (directory / "service.log").open("w+") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "clip_service", "--config", str(path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        try:
            wait_ready(port, cameras, process)
            sse = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
            sse.request("GET", "/api/v1/events")
            response = sse.getresponse()
            assert response.status == 200
            messages = queue.Queue()
            listener = threading.Thread(
                target=consume_events, args=(response, messages), daemon=True
            )
            listener.start()
            buffer = io.BytesIO()
            Image.new("RGB", (320, 240), "black").save(buffer, format="JPEG")
            black = {"type": "jpeg_base64", "data": base64.b64encode(buffer.getvalue()).decode()}
            for count in (1, 3):
                print(f"Measuring {count} logical camera(s) on {args.device}...", flush=True)
                cameras = [f"camera_{index}" for index in range(1, count + 1)]
                configure(args, path, port, cameras, -1)
                reload_result = request(port, "POST", "/api/v1/config/reload")
                if count == 3:
                    assert set(reload_result["added"]) == {"camera_2", "camera_3"}
                    assert reload_result["restarted"] == reload_result["removed"] == []
                wait_ready(port, cameras, process)
                for name in cameras:
                    episode = f"throughput-{count}-{name}"
                    request(
                        port, "POST", f"/api/v1/cameras/{name}/episodes", {"episode_id": episode}
                    )
                    request(
                        port,
                        "PUT",
                        f"/api/v1/cameras/{name}/baseline",
                        {"episode_id": episode, "source": {"type": "latest"}},
                    )
                model_status = request(port, "GET", "/api/v1/health")["model"]
                assert model_status["precision"] == args.precision
                assert model_status["weight_dtype"] == (
                    "float16" if args.precision == "fp16" else "float32"
                )
                phase = {
                    "logical_cameras": count,
                    "added_camera_ids": reload_result["added"],
                    "throughput": measure(port, cameras, args.seconds, args.warmup),
                }
                for name in cameras:
                    request(
                        port, "DELETE", f"/api/v1/cameras/{name}/episodes/throughput-{count}-{name}"
                    )
                configure(args, path, port, cameras, 0.9999)
                request(port, "POST", "/api/v1/config/reload")
                for name in cameras:
                    episode = f"candidate-{count}-{name}"
                    request(
                        port, "POST", f"/api/v1/cameras/{name}/episodes", {"episode_id": episode}
                    )
                    request(
                        port,
                        "PUT",
                        f"/api/v1/cameras/{name}/baseline",
                        {"episode_id": episode, "source": black},
                    )
                # Stop fresh candidates after these pending ones are acknowledged.
                received = receive(messages, "candidate_change", cameras)
                configure(args, path, port, cameras, -1)
                request(port, "POST", "/api/v1/config/reload")
                phase["candidate_notifications"] = verify_candidates(port, received, cameras)
                report["phases"].append(phase)
                assert request(port, "GET", "/api/v1/candidates")["candidates"] == []
                if count == 1:
                    request(
                        port, "DELETE", "/api/v1/cameras/camera_1/episodes/candidate-1-camera_1"
                    )
                    continue
                configure(args, path, port, [], -1)
                request(port, "POST", "/api/v1/config/reload")
                ended = receive(messages, "episode_ended", cameras)
                assert all(data["reason"] == "removed" for data, _ in ended.values())
                assert request(port, "GET", "/api/v1/health")["cameras"] == {}
                phase["remove_cameras_ended_sessions"] = True
        except Exception:
            log.flush()
            log.seek(0)
            print(log.read()[-20000:], file=sys.stderr)
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise RuntimeError("service did not stop within 30 seconds") from None
            finally:
                if sse is not None:
                    sse.close()
        if process.returncode != 0:
            log.seek(0)
            print(log.read()[-20000:], file=sys.stderr)
            raise RuntimeError(f"service exited with {process.returncode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cuda"), required=True)
    parser.add_argument("--precision", choices=("fp32", "tf32", "fp16"), default="fp32")
    parser.add_argument("--frigate-url", required=True)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=8)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--warmup", type=float, default=3)
    args = parser.parse_args()
    if not (
        args.fps > 0
        and args.seconds > 0
        and args.warmup >= 0
        and all(math.isfinite(value) for value in (args.fps, args.seconds, args.warmup))
    ):
        parser.error("fps and seconds must be positive; warmup must be nonnegative")
    import torch

    assert (
        torch.backends.mps.is_available() if args.device == "mps" else torch.cuda.is_available()
    ), f"{args.device} is unavailable"
    report = {
        "started_at": time.time(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "service_version": version("clip-service"),
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else platform.processor(),
        "transformers": version("transformers"),
        "opencv": version("opencv-python-headless"),
        "device": args.device,
        "precision": args.precision,
        "model": str(args.model.resolve()),
        "frigate_url": args.frigate_url,
        "stream": args.stream,
        "source_count": 1,
        "target_fps_per_camera": args.fps,
        "warmup_seconds": args.warmup,
        "measurement_seconds_per_phase": args.seconds,
        "stable_seconds": 0.3,
        "candidate_frame_interval": 0.075,
        "measurement": {
            "throughput": "completed encoding counter delta / host monotonic elapsed time",
            "latencies": "latest completion sampled every 0.2s, not all encoded frames; seconds",
            "processing": "preprocessing + CLIP + feature processing; excludes candidate storage",
            "notification": "successful candidate window's first low-similarity frame receive "
            "on CLIP host to local SSE receive; "
            "includes confirmation/storage/notification; "
            "excludes camera/Frigate transit",
            "trigger": "synthetic black baseline against real stream; not physical scene change",
            "three_camera_phase": "three logical readers of the same physical Frigate stream",
        },
        "phases": [],
    }
    with tempfile.TemporaryDirectory(prefix="clip-native-") as temporary:
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        run(args, Path(temporary), port, report)
    report["completed_at"] = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Native acceptance passed. Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
