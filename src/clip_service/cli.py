"""Start the service from a local TOML configuration."""

import argparse
import logging
import signal
import threading
from importlib.metadata import version
from pathlib import Path

from .config import ConfigError, load_config
from .http_api import ServiceHTTPServer
from .service import ClipService


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="clip-service",
        description="Frigate CLIP visual change detection service",
    )
    parser.add_argument("--version", action="version", version=version("clip-service"))
    parser.add_argument("--config", type=Path, default=Path("config.local.toml"))
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        config = load_config(args.config)
    except ConfigError as error:
        parser.error(str(error))
    service = ClipService(config, config_path=args.config.resolve())
    server = ServiceHTTPServer((config.server.host, config.server.port), service)

    def stop(signum: int, frame: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        service.start()
        logging.info("CLIP Service listening on http://%s:%s", *server.server_address[:2])
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
        service.close()
