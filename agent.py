"""Phase 4B — LLM agent (OpenAI). Chooses the next AuditAction from the observation JSON."""

from __future__ import annotations

import json
import os
from openai import OpenAI

from models import ActionType, AuditAction, AuditObservation, FraudType

__all__ = ["LLMAgent", "_as_observation"]

_VALID_ACTIONS = [e.value for e in ActionType]
_VALID_FRAUD = [e.value for e in FraudType]


def _as_observation(observation: AuditObservation | dict) -> AuditObservation:
    if isinstance(observation, AuditObservation):
        return observation
    return AuditObservation.model_validate(observation)


_SYSTEM = """You are a forensic auditor agent in a turn-based simulation.
You must propose exactly one next action as JSON matching the schema the user describes.
Be concise: use only document ids that appear in the observation."""


def _user_prompt(obs: AuditObservation) -> str:
    payload = obs.model_dump(mode="json")
    actions = ", ".join(_VALID_ACTIONS)
    frauds = ", ".join(_VALID_FRAUD)
    return f"""Current observation (JSON):
{json.dumps(payload, indent=2)}

Choose the single best next action.

action_type must be one of: {actions}

Params by action_type:
- request_category: {{"category": "<document category string>"}}
- cross_reference: {{"doc_a": "<id>", "doc_b": "<id>"}} — only if both ids are in documents_received
- flag_anomaly: {{"doc_id": "<id>", "description": "<short text>"}}
- request_external_confirmation: {{"category": "<category string>"}}
- submit_findings: {{"fraud_type": "<fraud enum string>", "evidence_chain": ["doc_id", ...], "obstruction_notes": [] (optional)}}

Valid fraud_type values: {frauds}

If observation.done is true, use submit_findings with your best assessment from what you already have.

Guidelines:
- If few categories are explored, prefer request_category on categories not listed in requested_categories_so_far.
- When you have multiple received docs with rich key_signals, consider cross_reference between two that might conflict.
- When observation shows CFO concealment or document_status_delta for known docs, mention these in obstruction_notes on submit.

Reply with ONLY a JSON object: {{"action_type": "...", "params": {{ ... }} }} — no markdown, no prose."""


def _parse_action(raw: str) -> AuditAction:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    data = json.loads(text)
    if not isinstance(data, dict) or "action_type" not in data:
        raise ValueError("LLM response must be a JSON object with action_type")
    params = data.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    return AuditAction(
        action_type=ActionType(data["action_type"]),
        params=params,
    )


class LLMAgent:
    """Calls OpenAI chat completions; returns a validated AuditAction."""

    def __init__(
        self,
        *,
        model: str | None = None,
        client: OpenAI | None = None,
    ):
        key = os.environ.get("OPENAI_API_KEY")
        if not key and client is None:
            raise RuntimeError(
                "OPENAI_API_KEY is not set; export it before using LLMAgent, "
                "or pass client=OpenAI(api_key=...)."
            )
        self._model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._client = client or OpenAI(api_key=key)

    def act(self, observation: AuditObservation | dict) -> AuditAction:
        obs = _as_observation(observation)
        if obs.done:
            return AuditAction(
                action_type=ActionType.submit_findings,
                params={
                    "fraud_type": FraudType.channel_stuffing.value,
                    "evidence_chain": list(obs.documents_received.keys())[:8],
                    "obstruction_notes": [],
                },
            )

        completion = self._client.chat.completions.create(
            model=self._model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _user_prompt(obs)},
            ],
        )
        content = completion.choices[0].message.content
        if not content:
            raise RuntimeError("Empty LLM response")
        try:
            return _parse_action(content)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            raise RuntimeError(f"Failed to parse LLM action: {content!r}") from e
