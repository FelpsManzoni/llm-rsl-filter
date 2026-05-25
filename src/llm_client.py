from __future__ import annotations

import json
from dataclasses import dataclass

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass(frozen=True)
class LLMClientConfig:
    base_url: str
    api_key: str | None
    timeout_seconds: float
    max_retries: int


class LLMClient:
    def __init__(self, config: LLMClientConfig) -> None:
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key or "not-required",
            timeout=config.timeout_seconds,
        )

    def evaluate(self, model_name: str, system_prompt: str, user_prompt: str) -> dict:
        @retry(
            stop=stop_after_attempt(max(1, self.config.max_retries)),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )
        def _call() -> str:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            message = response.choices[0].message.content
            if not message:
                raise ValueError("Empty LLM response content.")
            return message

        raw = _call()
        return _safe_load_json(raw)


def _safe_load_json(text: str) -> dict:
    try:
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("LLM output JSON must be an object.")
        return payload
    except json.JSONDecodeError:
        extracted = _extract_first_json_object(text)
        payload = json.loads(extracted)
        if not isinstance(payload, dict):
            raise ValueError("LLM output JSON must be an object.")
        return payload


def _extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in LLM response.")

    depth = 0
    in_string = False
    escaped = False

    for idx in range(start, len(text)):
        char = text[idx]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    raise ValueError("Incomplete JSON object in LLM response.")
