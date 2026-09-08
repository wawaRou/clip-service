from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class PrecisionError(ValueError):
    """A safe configuration diagnostic for the resolved inference backend."""


class ImageEncoder(Protocol):
    @property
    def device(self) -> str: ...

    @property
    def loaded(self) -> bool: ...

    def encode_jpeg(self, jpeg: bytes) -> np.ndarray: ...

    def encode_bgr(self, pixels: np.ndarray) -> np.ndarray: ...


class ClipEncoder:
    """Load one offline CLIP model on the first image encoding request."""

    def __init__(
        self,
        model_path: Path | None = None,
        requested_device: str = "auto",
        *,
        precision: str = "fp32",
        model_id: str = "openai/clip-vit-base-patch16",
    ) -> None:
        if requested_device not in {"auto", "mps", "cuda", "cpu"}:
            raise ValueError("device must be auto, mps, cuda, or cpu")
        self.model_path = model_path
        self.model_id = model_id
        if precision not in {"fp32", "tf32", "fp16"}:
            raise ValueError("precision must be fp32, tf32, or fp16")
        self.precision = precision
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

    @property
    def weight_dtype(self) -> str | None:
        return str(self._model.dtype).removeprefix("torch.") if self.loaded else None

    def _offline_path(self) -> Path:
        if self.model_path is not None:
            if not self.model_path.is_dir():
                raise FileNotFoundError(f"offline model snapshot not found: {self.model_path}")
            return self.model_path
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            return Path(snapshot_download(self.model_id, local_files_only=True))
        except LocalEntryNotFoundError:
            raise FileNotFoundError(
                f"offline model not found in Hugging Face cache: {self.model_id}"
            ) from None

    @property
    def available_offline(self) -> bool:
        try:
            self._offline_path()
        except (OSError, ValueError):
            return False
        return True

    def load(self) -> None:
        with self._lock:
            if self.loaded:
                return
            model_path = self._offline_path()

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

            if self.precision == "tf32" and device != "cuda":
                raise PrecisionError(f"precision=tf32 requires CUDA; resolved device is {device}")
            if self.precision == "fp16" and device == "cpu":
                raise PrecisionError("precision=fp16 requires CUDA or MPS")
            dtype = torch.float16 if self.precision == "fp16" else torch.float32
            if device == "cuda":
                # One shared model per service process. Do not mix the old and new APIs.
                if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
                    mode = "tf32" if self.precision == "tf32" else "ieee"
                    torch.backends.cuda.matmul.fp32_precision = mode
                    getattr(torch.backends.cudnn, "conv").fp32_precision = mode
                else:  # Orin's pinned PyTorch 2.8.
                    torch.backends.cuda.matmul.allow_tf32 = self.precision == "tf32"
                    torch.backends.cudnn.allow_tf32 = self.precision == "tf32"

            processor = CLIPImageProcessor.from_pretrained(str(model_path), local_files_only=True)
            model = CLIPModel.from_pretrained(str(model_path), local_files_only=True)
            model.eval()
            # Transformers' @wraps annotation loses the bound self in some releases.
            model.to(device, dtype=dtype)  # pyright: ignore[reportArgumentType]
            self._processor = processor
            self._device = device
            self._model = model

    def encode_jpeg(self, jpeg: bytes) -> np.ndarray:
        self.load()
        from PIL import Image

        with Image.open(io.BytesIO(jpeg)) as image:
            return self._encode_rgb(image.convert("RGB"))

    def encode_bgr(self, pixels: np.ndarray) -> np.ndarray:
        self.load()
        import cv2

        return self._encode_rgb(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))

    def _encode_rgb(self, image: Any) -> np.ndarray:
        import torch  # pyright: ignore[reportMissingImports]

        inputs = self._processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=self._device, dtype=self._model.dtype)
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
