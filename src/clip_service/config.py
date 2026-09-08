"""Load the service's TOML configuration and resolve per-camera settings."""

import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Literal
from urllib.parse import quote, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # pyright: ignore[reportMissingImports]  # Python 3.10 only


class ConfigError(ValueError):
    """An invalid configuration with a safe, user-facing location and explanation."""


class ConfigModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False, hide_input_in_errors=True
    )


class ServerConfig(ConfigModel):
    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=18080, ge=1, le=65535)


class ModelConfig(ConfigModel):
    id: str = Field(default="openai/clip-vit-base-patch16", min_length=1)
    path: Path | None = None
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    precision: Literal["fp32", "tf32", "fp16"] = "fp32"

    @model_validator(mode="after")
    def check_precision_backend(self):
        if self.precision == "tf32" and self.device not in {"auto", "cuda"}:
            raise ValueError("precision=tf32 requires device=cuda (or auto resolving to CUDA)")
        if self.precision == "fp16" and self.device == "cpu":
            raise ValueError("precision=fp16 requires CUDA or MPS")
        return self

    @field_validator("path", mode="before")
    @classmethod
    def parse_path(cls, value):
        return Path(value) if isinstance(value, str) else value


class FrigateConfig(ConfigModel):
    rtsp_base_url: str
    username: str | None = Field(default=None, repr=False)
    password: str | None = Field(default=None, repr=False)

    @field_validator("rtsp_base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        try:
            parts = urlsplit(value)
            port = parts.port
        except ValueError:
            raise ValueError("must be a valid RTSP base URL") from None
        if parts.scheme not in {"rtsp", "rtsps"} or not parts.hostname:
            raise ValueError("must be an RTSP base URL with a hostname")
        if port is not None and port < 1:
            raise ValueError("must use a port between 1 and 65535")
        if parts.username is not None or parts.password is not None:
            raise ValueError(
                "credentials belong in username/password fields or environment variables"
            )
        if parts.query or parts.fragment or any(character.isspace() for character in value):
            raise ValueError("must not contain whitespace, a query, or a fragment")
        return value.rstrip("/")

    def stream_url(self, stream: str) -> str:
        parts = urlsplit(self.rtsp_base_url)
        authority = parts.netloc
        if self.username is not None:
            auth = quote(self.username, safe="")
            if self.password is not None:
                auth += ":" + quote(self.password, safe="")
            authority = f"{auth}@{authority}"
        path = f"{parts.path.rstrip('/')}/{quote(stream, safe='')}"
        return urlunsplit((parts.scheme, authority, path, "", ""))


class DetectionConfig(ConfigModel):
    inference_fps: float = Field(default=8, gt=0)
    similarity_threshold: float = Field(default=0.9, ge=-1, le=1)
    stable_seconds: float = Field(default=1.0, gt=0)
    candidate_frame_count: Literal[5] = 5
    ring_seconds: float = Field(default=7, gt=0)
    ring_max_fps: float = Field(default=30, gt=0)
    frame_window_max_distance_seconds: float = Field(default=0.5, ge=0)
    jpeg_quality: int = Field(default=85, ge=1, le=100)
    open_timeout_seconds: float = Field(default=3, gt=0)
    read_timeout_seconds: float = Field(default=3, gt=0)
    reconnect_delay_seconds: float = Field(default=1, gt=0)
    frame_max_age_seconds: float = Field(default=1, gt=0)
    ack_timeout_seconds: float = Field(default=60, gt=0)

    @model_validator(mode="after")
    def check_buffer_window(self):
        if self.stable_seconds / (self.candidate_frame_count - 1) < 1 / self.ring_max_fps:
            raise ValueError("stable_seconds must cover four ring sampling intervals")
        if self.stable_seconds >= self.ring_seconds:
            raise ValueError("stable_seconds must be less than ring_seconds")
        return self


class CameraConfig(ConfigModel):
    stream: str = Field(min_length=1)
    enabled: bool = True
    inference_fps: float | None = None
    similarity_threshold: float | None = None
    stable_seconds: float | None = None
    candidate_frame_count: int | None = None
    ring_seconds: float | None = None
    ring_max_fps: float | None = None
    frame_window_max_distance_seconds: float | None = None
    jpeg_quality: int | None = None
    open_timeout_seconds: float | None = None
    read_timeout_seconds: float | None = None
    reconnect_delay_seconds: float | None = None
    frame_max_age_seconds: float | None = None
    ack_timeout_seconds: float | None = None


class ServiceConfig(ConfigModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    frigate: FrigateConfig
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    cameras: dict[str, CameraConfig] = Field(default_factory=dict)
    data_dir: Path = Path("data")
    retention_hours: float = Field(default=24, gt=0)

    @field_validator("data_dir", mode="before")
    @classmethod
    def parse_data_dir(cls, value):
        return Path(value) if isinstance(value, str) else value

    def camera_settings(self, name: str) -> DetectionConfig:
        overrides = self.cameras[name].model_dump(exclude={"stream", "enabled"}, exclude_none=True)
        return DetectionConfig.model_validate(self.detection.model_dump() | overrides)

    def camera_url(self, name: str) -> str:
        return self.frigate.stream_url(self.cameras[name].stream)


def load_config(path: Path, *, environ: Mapping[str, str] | None = None) -> ServiceConfig:
    environment = os.environ if environ is None else environ
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except OSError as error:
        raise ConfigError(f"{path}: {error.strerror}") from None
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        location = re.search(r"\(at (?:line \d+, column \d+|end of document)\)$", str(error))
        suffix = f" {location.group()}" if location else ""
        raise ConfigError(f"{path}: invalid TOML{suffix}") from None
    variables = {
        "CLIP_MODEL_PATH": ("model", "path"),
        "CLIP_DEVICE": ("model", "device"),
        "CLIP_FRIGATE_URL": ("frigate", "rtsp_base_url"),
        "CLIP_FRIGATE_USERNAME": ("frigate", "username"),
        "CLIP_FRIGATE_PASSWORD": ("frigate", "password"),
    }
    for variable, (section, key) in variables.items():
        if variable in environment:
            table = raw.setdefault(section, {})
            if not isinstance(table, dict):
                raise ConfigError(f"{path}: {section} must be a TOML table")
            table[key] = environment[variable]
    if "CLIP_DATA_DIR" in environment:
        raw["data_dir"] = environment["CLIP_DATA_DIR"]
    try:
        config = ServiceConfig.model_validate(raw)
    except ValidationError as error:
        raise ConfigError(_validation_message(path, error)) from None
    for name in config.cameras:
        try:
            config.camera_settings(name)
        except ValidationError as error:
            raise ConfigError(_validation_message(path, error, f"cameras.{name}.")) from None
    if config.model.path is not None:
        config.model.path = (path.parent / config.model.path.expanduser()).resolve()
    config.data_dir = (path.parent / config.data_dir.expanduser()).resolve()
    return config


def _validation_message(path: Path, error: ValidationError, prefix: str = "") -> str:
    details = []
    for item in error.errors(include_input=False, include_context=False, include_url=False):
        location = prefix + ".".join(str(part) for part in item["loc"])
        details.append(f"{location or 'configuration'}: {item['msg']}")
    return f"{path}: " + "; ".join(details)
