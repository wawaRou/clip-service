from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class ImageEncoder(Protocol):
    @property
    def device(self) -> str: ...

    @property
    def loaded(self) -> bool: ...

    def encode_jpeg(self, jpeg: bytes) -> np.ndarray: ...


class ClipEncoder:
    """Load one offline CLIP model on the first image encoding request."""

    def __init__(self, model_path: Path, requested_device: str = "auto") -> None:
        if requested_device not in {"auto", "mps", "cuda", "cpu"}:
            raise ValueError("device must be auto, mps, cuda, or cpu")
        self.model_path = model_path
        self._device = requested_device
        self._model: Any = None
        self._processor: Any = None
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        return self._device

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._lock:
            if self.loaded:
                return
            if not self.model_path.is_dir():
                raise FileNotFoundError(f"offline model snapshot not found: {self.model_path}")

            import torch  # pyright: ignore[reportMissingImports]
            from transformers import CLIPImageProcessor, CLIPModel

            device = self._device
            if device == "auto":
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
            elif device == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS was requested but is not available")
            elif device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available")

            processor = CLIPImageProcessor.from_pretrained(
                str(self.model_path), local_files_only=True
            )
            model = CLIPModel.from_pretrained(str(self.model_path), local_files_only=True)
            model.eval()
            # Transformers' @wraps annotation loses the bound self in some releases.
            model.to(device)  # pyright: ignore[reportArgumentType]
            self._processor = processor
            self._device = device
            self._model = model

    def encode_jpeg(self, jpeg: bytes) -> np.ndarray:
        self.load()
        import torch  # pyright: ignore[reportMissingImports]
        from PIL import Image

        with Image.open(io.BytesIO(jpeg)) as image:
            inputs = self._processor(images=image.convert("RGB"), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)
        with self._lock, torch.inference_mode():
            features = self._model.get_image_features(pixel_values=pixel_values)
            # Transformers 4 returns a tensor; 5 returns the projected pooler output.
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            vector = features.detach().float().cpu().numpy()[0]
        norm = float(np.linalg.norm(vector))
        if norm == 0 or not np.isfinite(norm):
            raise RuntimeError("CLIP returned an invalid embedding")
        return (vector / norm).astype(np.float32)
