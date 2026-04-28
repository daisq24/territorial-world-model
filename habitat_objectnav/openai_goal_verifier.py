from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import request

import numpy as np


@dataclass
class GoalVerificationResult:
    should_stop: bool
    confidence: float
    rationale: str


class OpenAIGoalVerifier:
    """Use an OpenAI vision-capable model to verify goal presence near stop time.

    The verifier is intentionally conservative:
    - it is disabled by default
    - it only runs when the heuristic stack already believes a stop is plausible
    - it caches the most recent signature to avoid repeated identical calls
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        model: str = "gpt-4.1-mini",
        api_key: str | None = None,
    ) -> None:
        self.enabled = enabled
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.last_signature_key: tuple[int, ...] | None = None
        self.last_result: GoalVerificationResult | None = None

    @property
    def available(self) -> bool:
        return self.enabled and bool(self.api_key)

    def reset(self) -> None:
        self.last_signature_key = None
        self.last_result = None

    def verify(
        self,
        *,
        rgb: Any,
        goal_label: str | None,
        signature_key: tuple[int, ...] | None,
    ) -> GoalVerificationResult:
        if not self.available or rgb is None or not goal_label:
            return GoalVerificationResult(should_stop=False, confidence=0.0, rationale="verifier_unavailable")

        if signature_key is not None and signature_key == self.last_signature_key and self.last_result is not None:
            return self.last_result

        image_url = self._rgb_to_data_url(rgb)
        prompt = (
            "You are verifying a robot stop decision for embodied object navigation.\n"
            f"Target object category: {goal_label}.\n"
            "Decide whether the target object is clearly visible close enough that the agent should stop now.\n"
            "Return strict JSON with keys: should_stop (boolean), confidence (0 to 1 float), rationale (short string)."
        )
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_url, "detail": "low"},
                    ],
                }
            ],
        }

        req = request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        raw_text = body.get("output_text", "").strip()
        parsed = self._parse_json_output(raw_text)
        result = GoalVerificationResult(
            should_stop=bool(parsed.get("should_stop", False)),
            confidence=float(parsed.get("confidence", 0.0)),
            rationale=str(parsed.get("rationale", "no_rationale")),
        )
        self.last_signature_key = signature_key
        self.last_result = result
        return result

    def _rgb_to_data_url(self, rgb: Any) -> str:
        from PIL import Image
        from io import BytesIO

        rgb_arr = np.asarray(rgb, dtype=np.uint8)
        image = Image.fromarray(rgb_arr)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _parse_json_output(self, text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise
