"""HTTP interface shared by status, camera control and event streaming."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .service import ClipService


class ServiceHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], service: ClipService) -> None:
        self.service = service
        super().__init__(address, RequestHandler)


class RequestHandler(BaseHTTPRequestHandler):
    server: ServiceHTTPServer  # pyright: ignore[reportIncompatibleVariableOverride]
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if urlsplit(self.path).path == "/api/v1/health":
            self._json(200, self.server.service.health())
        else:
            self._json(404, {"error": {"code": "not_found", "message": "endpoint not found"}})

    def _json(self, status: int, value: Any) -> None:
        content = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: Any) -> None:
        pass
