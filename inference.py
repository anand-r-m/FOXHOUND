"""
inference.py — FOXHOUND Inference Script
=========================================
Runs the LLM agent against all three tasks (easy, medium, hard) and emits
structured logs in the exact format required by Scaler's evaluation pipeline.

Environment Variables (injected by Scaler):
    API_KEY        - Competition proxy API key (REQUIRED)
    API_BASE_URL   - Competition proxy base URL (REQUIRED)
    MODEL_NAME     - Model identifier (default: gpt-4o-mini)
    ENV_URL        - FOXHOUND server URL (default: http://127.0.0.1:7860)
    LOCAL_IMAGE_NAME - Optional Docker image name
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import httpx
from openai import OpenAI

# Add project root to path for imports
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent import BaselineAgent, LLMAgent
from models import AuditObservation, TASK_SCORE_MIN, clamp_task_score

# ============================================================================
# OpenAI client — exactly as Scaler requires
# Uses os.environ["API_KEY"] and os.environ["API_BASE_URL"] (Scaler injects these)
# ============================================================================

API_KEY = os.environ.get("API_KEY")
API_BASE_URL = os.environ.get("API_BASE_URL")
# No hardcoded default: LiteLLM proxies often use different IDs; Scaler sets MODEL_NAME.
MODEL_NAME = os.environ.get("MODEL_NAME") or os.environ.get("OPENAI_MODEL")
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME")

# Build the client exactly as Scaler's sample shows
if API_KEY and API_BASE_URL:
    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
    )
elif API_KEY:
    client = OpenAI(api_key=API_KEY)
else:
    client = None

# ============================================================================
# Configuration
# ============================================================================

BASE_URL = os.environ.get("ENV_URL", "http://127.0.0.1:7860")
TASK_IDS = ["easy", "medium", "hard"]
MAX_STEPS_PER_EPISODE = 500
TIMEOUT = 120.0

# ============================================================================
# Structured Logging (Scaler format — do not change)
# ============================================================================

def log_start(task_id: str, difficulty: str) -> None:
    print(f"[START] {task_id} {difficulty}", flush=True)


def log_step(step: int, action_type: str, reward: float, done: bool) -> None:
    r = clamp_task_score(float(reward))
    print(f"[STEP] {step} {action_type} {r:.4f} {done}", flush=True)


def log_end(task_id: str, final_score: float) -> None:
    s = clamp_task_score(float(final_score))
    print(f"[END] {task_id} {s:.4f}", flush=True)


# ============================================================================
# Agent factory — pass the already-constructed client so LLMAgent
# always uses the proxy client, never re-reads env vars itself
# ============================================================================

def _make_agent():
    if client is not None:
        try:
            return LLMAgent(model=MODEL_NAME, client=client)
        except Exception as e:
            print(f"⚠ LLMAgent init failed ({e}), falling back to BaselineAgent", flush=True)
    return BaselineAgent()


def _is_llm_proxy_model_error(exc: BaseException) -> bool:
    """True when LiteLLM/proxy rejects the model name or returns 400."""
    msg = str(exc).lower()
    return (
        "does not exist" in msg
        or "notfounderror" in msg
        or "model_not_found" in msg
        or ("litellm" in msg and "model" in msg)
        or "invalid model" in msg
        or "unknown model" in msg
    )


def _act_or_fallback(agent, obs: AuditObservation):
    """Run agent.act; on proxy model errors, switch LLMAgent -> BaselineAgent once."""
    try:
        return agent, agent.act(obs)
    except Exception as e:
        if isinstance(agent, LLMAgent) and _is_llm_proxy_model_error(e):
            print(f"  ⚠ LLM proxy/model error, using BaselineAgent: {e}", flush=True)
            baseline = BaselineAgent()
            return baseline, baseline.act(obs)
        raise


# ============================================================================
# Episode runner
# ============================================================================

def run_task(http_client: httpx.Client, task_id: str) -> float:
    log_start(task_id, task_id)

    response = http_client.post(f"{BASE_URL}/reset", params={"task_id": task_id})
    response.raise_for_status()
    obs_dict = response.json()

    agent = _make_agent()
    cumulative_reward = 0.0
    step = 0
    done = False
    api_final_score: float | None = None  # info["score"] from the terminating step

    try:
        while not done and step < MAX_STEPS_PER_EPISODE:
            obs = AuditObservation.model_validate(obs_dict)
            agent, action = _act_or_fallback(agent, obs)

            step_resp = http_client.post(f"{BASE_URL}/step", json=action.model_dump())
            step_resp.raise_for_status()
            step_data = step_resp.json()

            obs_dict = step_data["observation"]
            reward = float(step_data["reward"])
            done = bool(step_data["terminated"])

            # Capture the graded final score the API returns on the terminating step.
            # This is grade_submission() output — the authoritative task score.
            if done:
                raw_score = step_data.get("info", {}).get("score")
                if raw_score is not None and math.isfinite(float(raw_score)):
                    api_final_score = clamp_task_score(float(raw_score))

            step += 1
            cumulative_reward += reward
            log_step(step, action.action_type.value, reward, done)

            if obs_dict.get("done", False):
                done = True
    except Exception as e:
        print(f"  ⚠ step error at step {step}: {e}", file=sys.stderr, flush=True)

    # Prefer the graded score from the API (grade_submission output).
    # Fall back to clamped cumulative step rewards only if the API didn't return one.
    if api_final_score is not None:
        final_score = api_final_score
    else:
        if not math.isfinite(cumulative_reward):
            cumulative_reward = 0.0
        final_score = clamp_task_score(
            cumulative_reward if cumulative_reward > 0 else TASK_SCORE_MIN
        )
    log_end(task_id, final_score)
    return final_score


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 60, flush=True)
    print("FOXHOUND Inference", flush=True)
    print("=" * 60, flush=True)
    print(f"API_KEY:      {'SET' if API_KEY else 'NOT SET'}", flush=True)
    print(f"API_BASE_URL: {API_BASE_URL or 'NOT SET'}", flush=True)
    print(f"MODEL_NAME:   {MODEL_NAME}", flush=True)
    print(f"ENV_URL:      {BASE_URL}", flush=True)
    print("=" * 60, flush=True)

    with httpx.Client(timeout=TIMEOUT) as http_client:
        # Retry health check — handles cold starts
        for attempt in range(1, 11):
            try:
                r = http_client.get(f"{BASE_URL}/health")
                r.raise_for_status()
                print(f"✓ Health check passed: {r.json()}", flush=True)
                break
            except Exception as e:
                if attempt == 10:
                    print(f"✗ Health check failed after {attempt} attempts: {e}",
                          file=sys.stderr, flush=True)
                    sys.exit(1)
                print(f"  Attempt {attempt}/10 failed ({e}), retrying in 3s...", flush=True)
                time.sleep(3)

        print("=" * 60, flush=True)

        results: dict[str, float] = {}
        for task_id in TASK_IDS:
            try:
                score = run_task(http_client, task_id)
                results[task_id] = score
                print(f"✓ {task_id}: {score:.4f}", flush=True)
            except Exception as e:
                print(f"✗ {task_id} FAILED: {e}", file=sys.stderr, flush=True)
                # Always emit [END] even on outer failure — validator requires it for every task
                fallback_score = TASK_SCORE_MIN
                log_end(task_id, fallback_score)
                results[task_id] = fallback_score

        print("=" * 60, flush=True)
        print("FINAL RESULTS:", flush=True)
        for task_id, score in results.items():
            print(f"  {task_id}: {score:.4f}", flush=True)
        print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
