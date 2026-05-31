from __future__ import annotations

import io

from ..errors import BackendUnavailableError
from ..results import Label
from .base import Backend


class RekognitionBackend(Backend):
    name = "aws"
    cost_per_image = 0.001
    default_min_confidence = 0.8

    def __init__(self, region: str = "us-east-1", label_floor: float = 50.0):
        try:
            import boto3
        except ImportError as exc:
            raise BackendUnavailableError(
                "aws", "pip install 'pyframe-gif-moderation[aws]'"
            ) from exc

        self.region = region
        self.label_floor = label_floor
        self._client = boto3.client("rekognition", region_name=region)

    def _score(self, image):
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG")

        response = self._client.detect_moderation_labels(
            Image={"Bytes": buffer.getvalue()},
            MinConfidence=self.label_floor,
        )
        labels = [
            Label(
                name=item["Name"],
                confidence=item["Confidence"] / 100.0,
                taxonomy=item.get("ParentName") or None,
            )
            for item in response.get("ModerationLabels", [])
        ]
        score = max((label.confidence for label in labels), default=0.0)
        return score, labels, response
