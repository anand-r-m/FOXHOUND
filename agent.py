"""Agents: BaselineAgent (heuristic) and LLMAgent (OpenAI)."""

from __future__ import annotations

import json
import os
import re
from typing import Any, cast

from openai import OpenAI

from models import ActionType, AuditAction, AuditObservation, DocumentCategory, FraudType

__all__ = ["BaselineAgent", "LLMAgent", "_as_observation"]

_VALID_ACTIONS = [e.value for e in ActionType]
_VALID_FRAUD = [e.value for e in FraudType]


def _as_observation(observation: AuditObservation | dict) -> AuditObservation:
    if isinstance(observation, AuditObservation):
        return observation
    return AuditObservation.model_validate(observation)


def _substantive_signals(key_signals: list[str]) -> list[str]:
    return [s for s in key_signals if not str(s).startswith("location:")]


def _signals_text(obs: AuditObservation) -> str:
    parts: list[str] = []
    for summary in obs.documents_received.values():
        parts.extend(_substantive_signals(list(summary.key_signals)))
    return " ".join(parts).lower()


def _guess_fraud_type(obs: AuditObservation) -> str:
    blob = _signals_text(obs)
    if re.search(r"wire|round\s*trip|cayman|incoming.*outgoing", blob):
        return FraudType.round_tripping.value
    if re.search(r"invoice|delivery|channel|q3.*q2|quarter\s*close", blob):
        return FraudType.channel_stuffing.value
    if re.search(
        r"cash receipt|phantom|dissolved|ar |accounts receivable|no wire|no corresponding",
        blob,
    ):
        return FraudType.phantom_revenue.value
    if "reserve" in blob or "cookie" in blob:
        return FraudType.cookie_jar_reservation.value
    if "hold" in blob and "bill" in blob:
        return FraudType.bill_and_hold.value
    return FraudType.channel_stuffing.value


class BaselineAgent:
    """Rule-based policy: sweep categories, one cross_reference, then submit."""

    _MIN_CATEGORIES: int = 6
    _MAX_CROSS_REFS: int = 2
    _SIGNAL_THRESHOLD: int = 2

    def __init__(self) -> None:
        self._category_order: list[str] = [
            DocumentCategory.financial_statements.value,
            DocumentCategory.bank_records.value,
            DocumentCategory.invoices.value,
            DocumentCategory.contracts.value,
            DocumentCategory.correspondence.value,
            DocumentCategory.audit_trails.value,
            DocumentCategory.tax_filings.value,
            DocumentCategory.hr_records.value,
        ]
        self._cross_referenced_count: int = 0
        self._flagged_doc_ids: set[str] = set()

    def act(self, observation: AuditObservation | dict) -> AuditAction:
        obs = _as_observation(observation)

        if obs.done:
            return AuditAction(
                action_type=ActionType.submit_findings,
                params={
                    "fraud_type": _guess_fraud_type(obs),
                    "evidence_chain": list(obs.documents_received.keys())[:12],
                    "obstruction_notes": [],
                },
            )

        requested = set(obs.requested_categories_so_far)

        if len(requested) < self._MIN_CATEGORIES:
            for cat in self._category_order:
                if cat not in requested:
                    return AuditAction(
                        action_type=ActionType.request_category,
                        params={"category": cat},
                    )

        for doc_id, summary in obs.documents_received.items():
            if doc_id not in self._flagged_doc_ids:
                signals = _substantive_signals(list(summary.key_signals))
                if len(signals) >= self._SIGNAL_THRESHOLD:
                    self._flagged_doc_ids.add(doc_id)
                    return AuditAction(
                        action_type=ActionType.flag_anomaly,
                        params={
                            "doc_id": doc_id,
                            "description": "; ".join(signals[:3]),
                        },
                    )

        if self._cross_referenced_count < self._MAX_CROSS_REFS and len(obs.documents_received) >= 2:
            pair = self._pick_cross_reference_pair(obs)
            if pair:
                self._cross_referenced_count += 1
                doc_a, doc_b = pair
                return AuditAction(
                    action_type=ActionType.cross_reference,
                    params={"doc_a": doc_a, "doc_b": doc_b},
                )

        obstruction: list[str] = []
        obstruction.extend(obs.cfo_visible_actions[:8])
        obstruction.extend(obs.document_status_delta[:8])
        return AuditAction(
            action_type=ActionType.submit_findings,
            params={
                "fraud_type": _guess_fraud_type(obs),
                "evidence_chain": list(obs.documents_received.keys())[:12],
                "obstruction_notes": obstruction,
            },
        )

    @staticmethod
    def _pick_cross_reference_pair(obs: AuditObservation) -> tuple[str, str] | None:
        scored: list[tuple[str, str, int]] = []
        for doc_id, summary in obs.documents_received.items():
            n = len(_substantive_signals(list(summary.key_signals)))
            scored.append((doc_id, summary.category.value, n))
        scored.sort(key=lambda x: -x[2])
        for i, (id_a, cat_a, n_a) in enumerate(scored):
            if n_a == 0:
                continue
            for id_b, cat_b, n_b in scored[i + 1 :]:
                if cat_a != cat_b and n_b > 0:
                    return (id_a, id_b)
        for id_a, cat_a, _ in scored:
            for id_b, cat_b, _ in scored:
                if id_a != id_b and cat_a != cat_b:
                    return (id_a, id_b)
        return None


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

If observation.done is true, you must use submit_findings only: best fraud_type, evidence_chain from documents you have seen (documents_received / anomalies), and obstruction_notes from cfo_visible_actions or document_status_delta if any.

Guidelines:
- If few categories are explored, prefer request_category on categories not listed in requested_categories_so_far.
- When you have multiple received docs with rich key_signals, consider cross_reference between two that might conflict.
- When observation shows CFO concealment or document_status_delta for known docs, mention these in obstruction_notes on submit.
- If remaining_steps is low (≤3) and you already have evidence, prefer submit_findings over new requests.
- request_external_confirmation only when critical docs are hidden (see document_status or env_feedback) and you still have confirmation budget implied by the scenario; it costs extra steps.
- Valid document category strings include: financial_statements, bank_records, invoices, contracts, correspondence, tax_filings, hr_records, audit_trails

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


def _sanitize_llm_action(action: AuditAction, obs: AuditObservation) -> AuditAction:
    """Keep the LLM inside valid ids and enums so the env does not reject the step."""
    received = obs.documents_received

    if action.action_type == ActionType.cross_reference:
        doc_a = action.params.get("doc_a")
        doc_b = action.params.get("doc_b")
        if doc_a not in received or doc_b not in received or doc_a == doc_b:
            pair = BaselineAgent._pick_cross_reference_pair(obs)
            if pair:
                return AuditAction(
                    action_type=ActionType.cross_reference,
                    params={"doc_a": pair[0], "doc_b": pair[1]},
                )
            if not received:
                return AuditAction(
                    action_type=ActionType.request_category,
                    params={"category": DocumentCategory.financial_statements.value},
                )
            return AuditAction(
                action_type=ActionType.flag_anomaly,
                params={
                    "doc_id": next(iter(received)),
                    "description": "cross_reference_ids_invalid_fallback",
                },
            )

    if action.action_type == ActionType.flag_anomaly:
        doc_id = action.params.get("doc_id")
        if doc_id not in received:
            if not received:
                return AuditAction(
                    action_type=ActionType.request_category,
                    params={"category": DocumentCategory.financial_statements.value},
                )
            doc_id = next(iter(received))
        desc = action.params.get("description") or "flagged"
        return AuditAction(
            action_type=ActionType.flag_anomaly,
            params={"doc_id": doc_id, "description": str(desc)[:500]},
        )

    if action.action_type == ActionType.submit_findings:
        chain_raw = action.params.get("evidence_chain") or []
        if isinstance(chain_raw, str):
            chain_raw = [chain_raw] if chain_raw.strip() else []
        if not isinstance(chain_raw, list):
            chain_raw = []
        filtered = [str(x).strip() for x in chain_raw if str(x).strip() in received]
        if not filtered:
            filtered = list(received.keys())[:12]
        else:
            filtered = filtered[:16]

        fraud_raw = action.params.get("fraud_type")
        fraud_s = _normalize_fraud_string(fraud_raw)
        if fraud_s not in _VALID_FRAUD:
            fraud_s = _guess_fraud_type(obs)

        notes_raw = action.params.get("obstruction_notes") or []
        if isinstance(notes_raw, str):
            notes_list = [notes_raw] if notes_raw.strip() else []
        elif isinstance(notes_raw, list):
            notes_list = [str(x).strip() for x in notes_raw if str(x).strip()]
        else:
            notes_list = []
        if not notes_list:
            notes_list = list(obs.cfo_visible_actions[:6]) + list(obs.document_status_delta[:6])

        return AuditAction(
            action_type=ActionType.submit_findings,
            params={
                "fraud_type": fraud_s,
                "evidence_chain": filtered,
                "obstruction_notes": notes_list[:20],
            },
        )

    if action.action_type == ActionType.request_category:
        cat = action.params.get("category")
        if isinstance(cat, DocumentCategory):
            cat = cat.value
        cat_s = str(cat).strip() if cat else ""
        valid_cats = {e.value for e in DocumentCategory}
        if cat_s not in valid_cats:
            for c in BaselineAgent()._category_order:
                if c not in obs.requested_categories_so_far:
                    return AuditAction(
                        action_type=ActionType.request_category,
                        params={"category": c},
                    )
            cat_s = DocumentCategory.financial_statements.value
        return AuditAction(
            action_type=ActionType.request_category,
            params={"category": cat_s},
        )

    if action.action_type == ActionType.request_external_confirmation:
        cat = action.params.get("category")
        if isinstance(cat, DocumentCategory):
            cat = cat.value
        cat_s = str(cat).strip() if cat else DocumentCategory.bank_records.value
        return AuditAction(
            action_type=ActionType.request_external_confirmation,
            params={"category": cat_s},
        )

    return action


def _normalize_fraud_string(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, FraudType):
        return raw.value
    return str(raw).strip()


class LLMAgent:
    """Calls OpenAI chat completions; returns a validated AuditAction (with local repair)."""

    def __init__(
        self,
        *,
        model: str | None = None,
        client: OpenAI | None = None,
        max_parse_retries: int = 2,
    ):
        # If a pre-built client is passed in (e.g. from inference.py), use it directly.
        # This is the preferred path — the caller controls exactly which proxy is used.
        if client is not None:
            self._client = client
        else:
            # Build client from env vars.
            # Scaler injects API_KEY + API_BASE_URL — these take absolute priority.
            key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("API_BASE_URL")

            if not key:
                raise RuntimeError(
                    "No API key found. Set API_KEY (Scaler proxy) or OPENAI_API_KEY."
                )

            if base_url:
                self._client = OpenAI(api_key=key, base_url=base_url)
            else:
                self._client = OpenAI(api_key=key)

        # MODEL_NAME (Scaler standard) > OPENAI_MODEL (legacy) > default
        self._model = model or os.environ.get("MODEL_NAME") or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._max_parse_retries = max(0, int(max_parse_retries))

    def act(self, observation: AuditObservation | dict) -> AuditAction:
        obs = _as_observation(observation)

        user_content = _user_prompt(obs)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_content},
        ]
        api_messages = cast(Any, messages)

        temperature = 0.1 if obs.done or obs.remaining_steps <= 3 else 0.2
        last_err: Exception | None = None

        for attempt in range(self._max_parse_retries + 1):
            completion = self._client.chat.completions.create(
                model=self._model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=api_messages,
            )
            content = completion.choices[0].message.content
            if not content:
                last_err = RuntimeError("Empty LLM response")
                messages.append(
                    {
                        "role": "user",
                        "content": "Your last reply was empty. Reply with ONLY one JSON object: "
                        '{"action_type": "...", "params": { ... }}',
                    }
                )
                continue
            try:
                parsed = _parse_action(content)
                if obs.done and parsed.action_type != ActionType.submit_findings:
                    parsed = AuditAction(
                        action_type=ActionType.submit_findings,
                        params=dict(parsed.params),
                    )
                return _sanitize_llm_action(parsed, obs)
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                last_err = e
                messages.append(
                    {
                        "role": "user",
                        "content": f"Invalid JSON or schema: {e}. Reply with ONLY valid JSON, "
                        f'keys "action_type" and "params", no markdown.',
                    }
                )

        assert last_err is not None
        raise RuntimeError(f"LLM did not return a valid action after retries: {last_err!r}") from last_err
