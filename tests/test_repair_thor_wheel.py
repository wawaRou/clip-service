import base64
import csv
import hashlib
import io
import os
import runpy
import zipfile
from pathlib import Path

import pytest

repair = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/repair_thor_wheel.py"))[
    "repair"
]


def test_repair_preserves_payload_and_rebuilds_record(tmp_path):
    source = tmp_path / "original.whl"
    target = tmp_path / "repaired.whl"
    prefix = "nvidia_cusparselt_cu13-0.8.1.dist-info/"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(prefix + "WHEEL", "Tag: py3-none-manylinux2014_sbsa\n")
        archive.writestr(prefix + "METADATA", "Name: nvidia-cusparselt-cu13\nVersion: 0.8.1\n")
        archive.writestr("nvidia/lib.so", b"unchanged CUDA binary")
        archive.writestr(prefix + "RECORD", "old record")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    repair(source, target, digest)
    with zipfile.ZipFile(target) as archive:
        assert archive.read("nvidia/lib.so") == b"unchanged CUDA binary"
        assert archive.read(prefix + "METADATA") == (
            b"Name: nvidia-cusparselt-cu13\nVersion: 0.8.1\n"
        )
        assert archive.read(prefix + "WHEEL") == b"Tag: py3-none-manylinux2014_aarch64\n"
        rows = list(csv.reader(io.StringIO(archive.read(prefix + "RECORD").decode())))
        assert len(rows) == 4
        for name, checksum, size in rows:
            if name.endswith("/RECORD"):
                assert checksum == size == ""
            else:
                payload = archive.read(name)
                expected = base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
                assert checksum == "sha256=" + expected.rstrip(b"=").decode()
                assert int(size) == len(payload)
    first = target.read_bytes()
    os.utime(target, ns=(1_000_000_000, 1_000_000_000))
    first_mtime = target.stat().st_mtime_ns
    repair(source, target, digest)
    assert target.read_bytes() == first
    assert target.stat().st_mtime_ns == first_mtime


def test_repair_rejects_untrusted_source(tmp_path):
    source = tmp_path / "bad.whl"
    source.write_bytes(b"untrusted")
    target = tmp_path / "out.whl"
    with pytest.raises(ValueError, match="SHA256"):
        repair(source, target, "0" * 64)
    assert not target.exists()
