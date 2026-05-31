from __future__ import annotations

import importlib.util

from .base import Backend

__all__ = ["Backend", "load_backend"]


def _available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _autodetect() -> str:
    if _available("transformers") and _available("torch"):
        return "local"
    if _available("boto3"):
        return "aws"
    return "local"  # surfaces the friendly install hint on construction


def load_backend(spec, **kwargs) -> Backend:
    if isinstance(spec, Backend):
        return spec

    if spec in (None, "auto"):
        spec = _autodetect()

    spec = str(spec)
    if spec.startswith("local:"):
        kwargs.setdefault("model", spec.split(":", 1)[1])
        spec = "local"

    if spec == "local":
        from .local import LocalBackend

        return LocalBackend(model=kwargs.get("model"))
    if spec == "aws":
        from .aws import RekognitionBackend

        return RekognitionBackend(region=kwargs.get("region", "us-east-1"))

    raise ValueError(f"Unknown backend: {spec!r}. Use 'local', 'aws', or 'local:<model-id>'.")
