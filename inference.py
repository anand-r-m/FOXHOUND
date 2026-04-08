"""
inference.py — FOXHOUND Baseline Inference Script
==================================================
Runs the baseline agent against all three tasks (easy, medium, hard) and emits
structured logs in the exact format required by Scaler's evaluation pipeline.

Environment Variables (REQUIRED):
    API_BASE_URL   - The API endpoint for the LLM (default provided for fallback)
    MODEL_NAME     - The model identifier to use for inference (default provided)
    HF_TOKEN       - Your Hugging Face / API key (NO DEFAULT - must be set)

Usage:
    export HF_TOKEN=hf_...
    python inference.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
from openai import OpenAI

# Add project root to path for imports
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent import BaselineAgent
from models import AuditObservation

# ============================================================================
# Environment Variables (as specified in Scaler requirements)
# ============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")  # NO DEFAULT - must be provided

# Optional: for Docker-based environments
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ============================================================================
# Scaler Required: OpenAI Client Configuration
# ============================================================================

# Initialize OpenAI client (required by Scaler even though baseline doesn't use LLM)
# This demonstrates compliance; LLMAgent in agent.py uses this pattern for actual LLM calls
if HF_TOKEN:
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
else:
    # Baseline agent doesn't require LLM, but we initialize for compliance
    client = None

# ============================================================================
# Configuration
# ============================================================================

BASE_URL = os.getenv("ENV_URL", "http://127.0.0.1:7860")  # Can override with ENV_URL
TASK_IDS = ["easy", "medium", "hard"]
MAX_STEPS_PER_EPISODE = 500
TIMEOUT = 120.0


# ============================================================================
# Structured Logging Functions (Scaler Format)
# ============================================================================

def log_start(task_id: str, difficulty: str) -> None:
    """Emit [START] log in exact Scaler format."""
    print(f"[START] {task_id} {difficulty}", flush=True)


def log_step(step: int, action_type: str, reward: float, done: bool) -> None:
    """Emit [STEP] log in exact Scaler format."""
    print(f"[STEP] {step} {action_type} {reward:.4f} {done}", flush=True)


def log_end(task_id: str, final_score: float) -> None:
    """Emit [END] log in exact Scaler format."""
    print(f"[END] {task_id} {final_score:.4f}", flush=True)


# ============================================================================
# Inference Logic
# ============================================================================

def run_task(client_http: httpx.Client, task_id: str) -> float:
    """
    Run the baseline agent on a single task and return the final score.
    
    Args:
        client_http: HTTP client for API calls
        task_id: Task identifier (easy/medium/hard)
        
    Returns:
        Final cumulative reward as float in [0.0, 1.0]
    """
    # Emit START log
    log_start(task_id, task_id)  # difficulty = task_id for this env
    
    # Reset episode
    response = client_http.post(f"{BASE_URL}/reset", params={"task_id": task_id})
    response.raise_for_status()
    obs_dict = response.json()
    
    # Initialize agent and tracking
    agent = BaselineAgent()
    cumulative_reward = 0.0
    step = 0
    done = False
    
    # Episode loop
    while not done and step < MAX_STEPS_PER_EPISODE:
        # Get observation object
        obs = AuditObservation.model_validate(obs_dict)
        
        # Agent decides action
        action = agent.act(obs)
        
        # Send action to environment
        step_response = client_http.post(
            f"{BASE_URL}/step",
            json=action.model_dump(),
        )
        step_response.raise_for_status()
        step_data = step_response.json()
        
        # Extract standardized response fields
        obs_dict = step_data["observation"]
        reward = float(step_data["reward"])  # scalar float per openM standard
        done = bool(step_data["terminated"])
        
        # Update tracking
        step += 1
        cumulative_reward += reward
        
        # Emit STEP log
        log_step(step, action.action_type.value, reward, done)
        
        # Safety: check if episode should terminate
        if obs_dict.get("done", False):
            done = True
    
    # Clamp final score to [0.0, 1.0]
    final_score = max(0.0, min(1.0, cumulative_reward))
    
    # Emit END log
    log_end(task_id, final_score)
    
    return final_score


def main() -> None:
    """Run inference on all tasks and report results."""
    print("=" * 60, flush=True)
    print("FOXHOUND Baseline Inference", flush=True)
    print("=" * 60, flush=True)
    print(f"API_BASE_URL: {API_BASE_URL}", flush=True)
    print(f"MODEL_NAME: {MODEL_NAME}", flush=True)
    print(f"HF_TOKEN: {'SET' if HF_TOKEN else 'NOT SET'}", flush=True)
    print(f"Target URL: {BASE_URL}", flush=True)
    print("=" * 60, flush=True)
    
    # Health check
    with httpx.Client(timeout=TIMEOUT) as http_client:
        try:
            health_resp = http_client.get(f"{BASE_URL}/health")
            health_resp.raise_for_status()
            print(f"✓ Health check passed: {health_resp.json()}", flush=True)
        except Exception as e:
            print(f"✗ Health check failed: {e}", file=sys.stderr, flush=True)
            print("Ensure the server is running: uvicorn server.app:app --host 0.0.0.0 --port 7860", file=sys.stderr, flush=True)
            sys.exit(1)
        
        print("=" * 60, flush=True)
        
        # Run all tasks
        results = {}
        for task_id in TASK_IDS:
            try:
                score = run_task(http_client, task_id)
                results[task_id] = score
                print(f"✓ {task_id}: {score:.4f}", flush=True)
            except Exception as e:
                print(f"✗ {task_id} FAILED: {e}", file=sys.stderr, flush=True)
                results[task_id] = 0.0
        
        print("=" * 60, flush=True)
        print("FINAL RESULTS:", flush=True)
        for task_id, score in results.items():
            print(f"  {task_id}: {score:.4f}", flush=True)
        print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
