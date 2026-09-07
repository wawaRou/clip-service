import io
from pathlib import Path

import numpy as np
import pytest

from clip_service.model import ClipEncoder


def test_encoder_is_lazy_and_reports_missing_offline_model(tmp_path: Path) -> None:
    encoder = ClipEncoder(tmp_path / "missing", requested_device="cpu")

    assert not encoder.loaded
    assert encoder.device == "cpu"
    with pytest.raises(FileNotFoundError, match="offline model"):
        encoder.encode_jpeg(b"unused")
    assert not encoder.loaded


@pytest.fixture
def tiny_clip(tmp_path: Path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from PIL import Image

    torch.manual_seed(0)
    config = transformers.CLIPConfig(
        text_config={
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "max_position_embeddings": 8,
        },
        vision_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 8,
            "patch_size": 4,
        },
        projection_dim=4,
    )
    transformers.CLIPModel(config).save_pretrained(tmp_path)
    transformers.CLIPImageProcessor(
        size={"shortest_edge": 8}, crop_size={"height": 8, "width": 8}
    ).save_pretrained(tmp_path)
    image = Image.new("RGB", (8, 8), color=(220, 30, 60))
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return tmp_path, output.getvalue()


def test_offline_encoder_produces_repeatable_unit_image_embeddings(tiny_clip):
    path, jpeg = tiny_clip
    encoder = ClipEncoder(path, requested_device="cpu")
    embedding = encoder.encode_jpeg(jpeg)

    assert encoder.loaded
    assert encoder.device == "cpu"
    assert embedding.shape == (4,)
    assert embedding.dtype == np.float32
    assert np.isfinite(embedding).all()
    assert np.linalg.norm(embedding) == pytest.approx(1)
    np.testing.assert_allclose(encoder.encode_jpeg(jpeg), embedding)


def test_precision_rejects_invalid_or_incompatible_modes(tmp_path):
    with pytest.raises(ValueError, match="precision"):
        ClipEncoder(tmp_path, "cpu", precision="int8")
    for precision in ["tf32", "fp16"]:
        encoder = ClipEncoder(tmp_path, "cpu", precision=precision)
        with pytest.raises(ValueError, match="requires"):
            encoder.load()


def test_mps_fp16_keeps_normalized_float32_output(tiny_clip):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware unavailable")
    path, jpeg = tiny_clip
    fp32 = ClipEncoder(path, "mps", precision="fp32").encode_jpeg(jpeg)
    half = ClipEncoder(path, "mps", precision="fp16")
    embedding = half.encode_jpeg(jpeg)
    assert half.weight_dtype == "float16"
    assert embedding.dtype == np.float32
    assert np.isfinite(embedding).all()
    assert np.linalg.norm(embedding) == pytest.approx(1)
    assert float(embedding @ fp32) > 0.999


@pytest.mark.parametrize(
    "precision,dtype", [("fp32", "float32"), ("tf32", "float32"), ("fp16", "float16")]
)
def test_cuda_precision_applies_to_real_encoder(tiny_clip, precision, dtype):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware unavailable")
    path, jpeg = tiny_clip
    encoder = ClipEncoder(path, "cuda", precision=precision)
    embedding = encoder.encode_jpeg(jpeg)
    assert encoder.weight_dtype == dtype
    assert embedding.dtype == np.float32
    assert np.isfinite(embedding).all()
    assert np.linalg.norm(embedding) == pytest.approx(1)
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        assert torch.backends.cuda.matmul.fp32_precision == (
            "tf32" if precision == "tf32" else "ieee"
        )
        assert torch.backends.cudnn.conv.fp32_precision == (
            "tf32" if precision == "tf32" else "ieee"
        )
    else:
        assert torch.backends.cuda.matmul.allow_tf32 == (precision == "tf32")
        assert torch.backends.cudnn.allow_tf32 == (precision == "tf32")
