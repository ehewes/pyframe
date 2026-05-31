from __future__ import annotations

from ..errors import BackendUnavailableError
from ..results import Label
from .base import Backend

DEFAULT_MODEL = "AdamCodd/vit-base-nsfw-detector"
NSFW_LABELS = {"nsfw", "porn", "hentai", "sexy", "explicit", "drawings_nsfw"}
SAFE_LABELS = {"normal", "sfw", "safe", "neutral", "drawings"}


class LocalBackend(Backend):
    name = "local"
    cost_per_image = 0.0
    default_min_confidence = 0.5  # recall-safe: the model's own argmax point

    def __init__(self, model: str | None = None, nsfw_labels=None):
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise BackendUnavailableError(
                "local", "pip install 'pyframe-gif-moderation[local]'"
            ) from exc

        self.model = model or DEFAULT_MODEL
        self._classifier = pipeline("image-classification", model=self.model)
        self.nsfw_labels = {label.lower() for label in (nsfw_labels or NSFW_LABELS)}

    def _score(self, image):
        results = self._classifier(image)
        labels = [Label(name=r["label"], confidence=float(r["score"])) for r in results]

        nsfw = 0.0
        safe = None
        for r in results:
            name = r["label"].lower()
            conf = float(r["score"])
            if name in self.nsfw_labels:
                nsfw = max(nsfw, conf)
            elif name in SAFE_LABELS:
                safe = conf if safe is None else max(safe, conf)

        # Fall back to (1 - safe) for unknown label sets so a model that only
        # reports a "normal" class still yields a usable NSFW score.
        if nsfw == 0.0 and safe is not None:
            nsfw = 1.0 - safe
        return nsfw, labels, results
