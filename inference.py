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

from agent import LLMAgent, BaselineAgent
from models import AuditObservation

# ============================================================================
# OpenAI client — exactly as Scaler requires
# Uses os.environ["API_KEY"] and os.environ["API_BASE_URL"] (Scaler injects these)
# ============================================================================

API_KEY = os.environ.get("API_KEY")
API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
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
    print(f"[STEP] {step} {action_type} {reward:.4f} {done}", flush=True)


def log_end(task_id: str, final_score: float) -> None:
    print(f"[END] {task_id} {final_score:.4f}", flush=True)


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

    while not done and step < MAX_STEPS_PER_EPISODE:
        obs = AuditObservation.model_validate(obs_dict)
        action = agent.act(obs)

        step_resp = http_client.post(f"{BASE_URL}/step", json=action.model_dump())
        step_resp.raise_for_status()
        step_data = step_resp.json()

        obs_dict = step_data["observation"]
        reward = float(step_data["reward"])
        done = bool(step_data["terminated"])

        step += 1
        cumulative_reward += reward
        log_step(step, action.action_type.value, reward, done)

        if obs_dict.get("done", False):
            done = True

    # Clamp strictly within (0, 1) — validator requires exclusive bounds
    final_score = max(0.01, min(0.99, cumulative_reward))
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
                results[task_id] = 0.01  # strictly > 0 as required by validator

        print("=" * 60, flush=True)
        print("FINAL RESULTS:", flush=True)
        for task_id, score in results.items():
            print(f"  {task_id}: {score:.4f}", flush=True)
        print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
