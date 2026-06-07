from __future__ import annotations

import importlib.util
import threading

from .base import Backend

__all__ = ["Backend", "load_backend", "clear_backend_cache"]


def _available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _autodetect() -> str:
    if _available("transformers") and _available("torch"):
        return "local"
    if _available("boto3"):
        return "aws"
    return "local"  # surfaces the friendly install hint on construction


# Constructing a backend loads model weights (~0.5 GB for the local ViT) and is the
# dominant setup cost. Cache by construction identity so the CLI's per-file loop and
# repeated Pipe()/scan() calls reuse one loaded model instead of reloading it each time.
# Backends are stateless after construction, so sharing one instance is safe; the lock
# keeps a concurrent first-miss from loading the same model twice.
_cache: dict[tuple, Backend] = {}
_cache_lock = threading.Lock()


def _construct(spec: str, kwargs: dict) -> Backend:
    if spec == "local":
        from .local import LocalBackend

        return LocalBackend(model=kwargs.get("model"))
    if spec == "aws":
        from .aws import RekognitionBackend

        return RekognitionBackend(region=kwargs.get("region", "us-east-1"))

    raise ValueError(f"Unknown backend: {spec!r}. Use 'local', 'aws', or 'local:<model-id>'.")


def load_backend(spec, *, cache: bool = True, **kwargs) -> Backend:
    # An already-built backend (e.g. a test fake or a user-constructed instance) is
    # passed straight through, never cached.
    if isinstance(spec, Backend):
        return spec

    if spec in (None, "auto"):
        spec = _autodetect()

    spec = str(spec)
    if spec.startswith("local:"):
        kwargs.setdefault("model", spec.split(":", 1)[1])
        spec = "local"

    if not cache:
        return _construct(spec, kwargs)

    # Key on exactly the inputs that change the constructed backend: load_backend only
    # varies model (local) and region (aws). Custom nsfw_labels/label_floor are reachable
    # only by constructing a backend directly, which bypasses this cache.
    key = (spec, kwargs.get("model"), kwargs.get("region", "us-east-1"))
    cached = _cache.get(key)
    if cached is not None:
        return cached
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None:
            return cached
        backend = _construct(spec, kwargs)  # construction failures are not cached
        _cache[key] = backend
        return backend


def clear_backend_cache() -> None:
    """Drop cached backends so their model weights can be garbage-collected. Useful in
    tests or to reclaim memory after a batch of scans."""
    with _cache_lock:
        _cache.clear()
