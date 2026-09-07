import socket
import threading
import time

import pytest

from clip_service.config import DetectionConfig
from clip_service.video import open_capture


@pytest.mark.parametrize("timeout", [0.2, 0.0001])
def test_unresponsive_rtsp_peer_times_out_without_logging_credentials(capfd, timeout):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(2)
    stop = threading.Event()

    def stalled_peer():
        with listener:
            connection, _ = listener.accept()
            with connection:
                stop.wait(2)

    peer = threading.Thread(target=stalled_peer)
    peer.start()
    started = time.monotonic()
    try:
        capture = open_capture(
            f"rtsp://secret-user:secret-password@127.0.0.1:{listener.getsockname()[1]}/stream",
            DetectionConfig(open_timeout_seconds=timeout, read_timeout_seconds=timeout),
        )
        try:
            assert not capture.isOpened()
            assert time.monotonic() - started < 1.5
        finally:
            capture.release()
    finally:
        stop.set()
        peer.join(3)
    output = capfd.readouterr()
    assert "secret-user" not in output.out + output.err
    assert "secret-password" not in output.out + output.err
