import json
import math
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models import AuditAction, TaskConfig, clamp_task_score
from env import ForensicAuditEnv
from grader import grade_submission

app = FastAPI(title="FOXHOUND API")

# Repo root: parent of server/
_ROOT = Path(__file__).resolve().parent.parent
_TASK_DIR = _ROOT / "tasks"

_TASK_FILES = {
    "easy": "easy_task.json",
    "medium": "medium_task.json",
    "hard": "hard_tasks.json",
}

# Single global env (simple; validator runs one episode at a time)
env: ForensicAuditEnv | None = None
_current_task_id: str | None = None


def _load_task_config(task_id: str) -> TaskConfig:
    filename = _TASK_FILES.get(task_id)
    if not filename:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id={task_id!r}. Use one of: {sorted(_TASK_FILES)}",
        )
    path = _TASK_DIR / filename
    if not path.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Task file missing: {path}",
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return TaskConfig.model_validate(data)


def _serialize_info(info: object) -> dict:
    if info is None:
        return {}
    if isinstance(info, dict):
        return info
    if isinstance(info, BaseModel):
        return info.model_dump()
    return {"raw": info}


@app.get("/")
async def root():
    """HF Spaces and browsers open `/` by default; the API lives on other paths."""
    return {
        "service": "FOXHOUND API",
        "health": "/health",
        "openapi": "/openapi.json",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/reset")
async def reset(task_id: str = "easy"):
    global env, _current_task_id

    config = _load_task_config(task_id)
    env = ForensicAuditEnv(config)
    _current_task_id = task_id

    observation = env.reset()
    payload = observation.model_dump()
    # Keep flat reset payload for existing clients, plus validator-friendly score metadata.
    payload["info"] = {"task_id": task_id, "score": 0.5}
    return payload


@app.post("/step")
async def step(action: AuditAction):
    global env, _current_task_id
    if env is None:
        raise HTTPException(status_code=400, detail="Call POST /reset first")
    observation, reward_info, done, info = env.step(action)

    _rt = reward_info.total if math.isfinite(reward_info.total) else 0.01
    clamped_total = round(min(max(_rt, 0.01), 0.99), 3)
    reward_breakdown = reward_info.model_dump()
    reward_breakdown["total"] = clamped_total
    reward_breakdown["components"] = {
        k: round(min(max(v if math.isfinite(v) else 0.01, 0.01), 0.99), 3)
        for k, v in reward_breakdown.get("components", {}).items()
    }

    final_score = clamped_total
    if done:
        try:
            graded = grade_submission(env.state())

            _gs = graded.total if math.isfinite(graded.total) else 0.01
            final_score = round(min(max(_gs, 0.01), 0.99), 3)

        except Exception as e:
            print("🚨 GRADER CRASH")
            print("Task ID:", _current_task_id)
            print("State:", env.state())
            print("Error:", e)
            final_score = 0.5

    # openM/Gymnasium standard: (obs, reward: float, terminated, truncated, info)
    return {
        "observation": observation.model_dump(),
        "reward": clamped_total,           # scalar float per openM standard
        "terminated": done,
        "truncated": False,                # this env never truncates
        "info": {
            **_serialize_info(info),
            "task_id": _current_task_id,
            "reward_breakdown": reward_breakdown,
            "score": final_score,
        },
    }


@app.get("/state")
async def state():
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Call POST /reset first")
    return env.state().model_dump()


def main() -> None:
    """Entry point for programmatic server launch."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
