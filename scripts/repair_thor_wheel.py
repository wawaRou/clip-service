"""Prepare the pinned Thor wheel before uv sync; uses only the standard library."""

import base64
import csv
import filecmp
import hashlib
import io
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

FILENAME = "nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_aarch64.whl"
URL = "https://pypi.nvidia.com/nvidia-cusparselt-cu13/" + FILENAME
SHA256 = "4dca476c50bf4780d46cd0bfbd82e2bc10a08e4fef7950917ce8d7578d22a23f"
PREFIX = "nvidia_cusparselt_cu13-0.8.1.dist-info/"


def repair(source: Path, target: Path, expected_sha256: str) -> None:
    with source.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("Upstream wheel SHA256 mismatch")
    with zipfile.ZipFile(source) as original:
        wheel = original.read(PREFIX + "WHEEL")
        old = b"Tag: py3-none-manylinux2014_sbsa"
        if wheel.count(old) != 1:
            raise ValueError("Unexpected upstream WHEEL tag")
        wheel = wheel.replace(old, b"Tag: py3-none-manylinux2014_aarch64")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
            output = Path(temporary) / target.name
            rows = []
            with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as patched:
                for entry in original.infolist():
                    if entry.filename == PREFIX + "RECORD":
                        continue
                    data = wheel if entry.filename == PREFIX + "WHEEL" else original.read(entry)
                    checksum = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                    rows.append(
                        (entry.filename, "sha256=" + checksum.rstrip(b"=").decode(), str(len(data)))
                    )
                    entry.compress_type = zipfile.ZIP_STORED
                    patched.writestr(entry, data)
                record = io.StringIO(newline="")
                csv.writer(record, lineterminator="\n").writerows(
                    [*rows, (PREFIX + "RECORD", "", "")]
                )
                info = zipfile.ZipInfo(PREFIX + "RECORD", date_time=(2026, 1, 1, 0, 0, 0))
                patched.writestr(info, record.getvalue())
            if not target.exists() or not filecmp.cmp(output, target, shallow=False):
                output.replace(target)


def main() -> None:
    target = Path(__file__).resolve().parents[1] / "vendor" / FILENAME
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / FILENAME
        print("Downloading pinned upstream wheel...", flush=True)
        with urllib.request.urlopen(URL, timeout=120) as response, source.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        repair(source, target, SHA256)
    print(f"Prepared {target}")


if __name__ == "__main__":
    main()
