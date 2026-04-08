import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models import AuditAction, TaskConfig
from env import ForensicAuditEnv

app = FastAPI(title="FOXHOUND API")

# Repo root: parent of server/
_ROOT = Path(__file__).resolve().parent.parent
_TASK_DIR = _ROOT / "tasks"

_TASK_FILES = {
    "easy": "easy_task.json",
    "medium": "medium_task.json",
    "hard": "hard_tasks.json",
}

env: ForensicAuditEnv | None = None


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
    global env
    config = _load_task_config(task_id)
    env = ForensicAuditEnv(config)
    observation = env.reset()
    return observation.model_dump()


@app.post("/step")
async def step(action: AuditAction):
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Call POST /reset first")
    observation, reward_info, done, info = env.step(action)
    
    # openM/Gymnasium standard: (obs, reward: float, terminated, truncated, info)
    return {
        "observation": observation.model_dump(),
        "reward": reward_info.total,  # scalar float, not dict
        "terminated": done,
        "truncated": False,  # this env doesn't truncate episodes
        "info": {
            **_serialize_info(info),
            "reward_breakdown": reward_info.model_dump(),  # preserve detail in info
        },
    }


@app.get("/state")
async def state():
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Call POST /reset first")
    return env.state().model_dump()
