"""Command-line entry point; service implementation will be migrated separately."""

import argparse
from importlib.metadata import version


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="clip-service",
        description="CLIP Service project skeleton. Video service is not implemented yet.",
    )
    parser.add_argument("--version", action="version", version=version("clip-service"))
    parser.parse_args()
    parser.print_help()
