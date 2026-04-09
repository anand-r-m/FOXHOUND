"""End-to-end demo: LLMAgent (default when API key is set) or BaselineAgent on the HTTP API."""

from __future__ import annotations
import math
import argparse
import os
import sys
from pathlib import Path

import httpx

# Allow `python demo.py` from repo app root (same layout as tests/conftest.py)
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent import BaselineAgent, LLMAgent
from grader import grade_submission
from models import AuditState

DEFAULT_URL = "http://127.0.0.1:7860"
TASK_IDS = ("easy", "medium", "hard")
MAX_STEPS_PER_EPISODE = 500


def _has_openai_credentials() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def _resolve_agent_mode(agent_arg: str) -> tuple[bool, str]:
    """
    Returns (use_llm, label for logs).

    auto: LLM when OPENAI_API_KEY is set, else baseline (safe for stage demos).
    llm: require key; exit with instructions if missing.
    baseline: always heuristic.
    """
    a = agent_arg.strip().lower()
    if a == "baseline":
        return False, "baseline (heuristic)"

    if a == "llm":
        if not _has_openai_credentials():
            print(
                "OPENAI_API_KEY is not set. Add it to your environment or Hugging Face Space secrets, "
                "or run with --agent baseline / --agent auto.",
                file=sys.stderr,
            )
            sys.exit(2)
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return True, f"llm (model={model})"

    # auto
    if _has_openai_credentials():
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return True, f"llm (model={model}, auto-selected)"
    return False, "baseline (heuristic, auto — no OPENAI_API_KEY)"


def _check_health(client: httpx.Client, base_url: str) -> None:
    r = client.get(f"{base_url.rstrip('/')}/health", timeout=30.0)
    r.raise_for_status()
    if r.json().get("status") != "ok":
        raise RuntimeError(f"Unexpected /health payload: {r.json()}")


def run_demo(base_url: str, *, use_llm: bool, agent_label: str) -> None:
    base = base_url.rstrip("/")
    timeout = 180.0 if use_llm else 120.0
    print(f"[demo] url={base} agent={agent_label}")
    with httpx.Client(timeout=timeout) as client:
        _check_health(client, base)

        for task_id in TASK_IDS:
            print(f"[START] task={task_id}")
            r = client.post(f"{base}/reset", params={"task_id": task_id})
            r.raise_for_status()
            obs: dict = r.json()
            agent = LLMAgent() if use_llm else BaselineAgent()
            total_step_reward = 0.0
            n_steps = 0

            while not obs.get("done") and n_steps < MAX_STEPS_PER_EPISODE:
                action = agent.act(obs)
                body = action.model_dump(mode="json")
                r = client.post(f"{base}/step", json=body)
                r.raise_for_status()

                print("RAW STEP RESPONSE:", r.text)  # <-- ADD THIS

                data = r.json()
                obs = data["observation"]
                # reward is now scalar float (openM standard), not dict
                step_total = float(data["reward"])
                total_step_reward += step_total
                print(
                    f"  [STEP] {action.action_type.value} → "
                    f"step_reward={step_total:.4f} remaining={obs.get('remaining_steps')}"
                )
                n_steps += 1

            if n_steps >= MAX_STEPS_PER_EPISODE and not obs.get("done"):
                print(f"  [STEP] aborted: max steps ({MAX_STEPS_PER_EPISODE})")

            rs = client.get(f"{base}/state")
            rs.raise_for_status()
            state = AuditState.model_validate(rs.json())
            grade = grade_submission(state)
            print("DEBUG FINAL GRADE:", grade.total, "isfinite:", math.isfinite(grade.total))

            if not (math.isfinite(grade.total) and 0.0 < grade.total < 1.0):
                print("🚨 INVALID FINAL SCORE DETECTED:", grade.total)

            print(
                f"  [END] task={task_id} steps={n_steps} "
                f"sum_step_reward={total_step_reward:.4f} final_grade={grade.total:.4f}"
            )


def main() -> None:
    p = argparse.ArgumentParser(description="FOXHOUND agent HTTP demo")
    p.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"API base URL (default: {DEFAULT_URL})",
    )
    p.add_argument(
        "--agent",
        choices=("auto", "baseline", "llm"),
        default="auto",
        help="auto=LLM if OPENAI_API_KEY is set else baseline (best for presentations); "
        "llm=OpenAI only; baseline=heuristic only",
    )
    args = p.parse_args()
    use_llm, agent_label = _resolve_agent_mode(args.agent)
    try:
        run_demo(args.url, use_llm=use_llm, agent_label=agent_label)
    except httpx.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
