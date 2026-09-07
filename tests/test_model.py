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


def test_offline_encoder_produces_repeatable_unit_image_embeddings(tmp_path: Path) -> None:
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
    encoder = ClipEncoder(tmp_path, requested_device="cpu")

    embedding = encoder.encode_jpeg(output.getvalue())

    assert encoder.loaded
    assert encoder.device == "cpu"
    assert embedding.shape == (4,)
    assert embedding.dtype == np.float32
    assert np.isfinite(embedding).all()
    assert np.linalg.norm(embedding) == pytest.approx(1)
    np.testing.assert_allclose(encoder.encode_jpeg(output.getvalue()), embedding)
