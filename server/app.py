from fastapi import FastAPI
from models import AuditAction
from env import ForensicAuditEnv

app = FastAPI(title="FOXHOUND API")
env: ForensicAuditEnv | None = None

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/reset")
async def reset(task_id: str = "easy"):
    global env
    env = ForensicAuditEnv()
    observation = env.reset()
    return observation.model_dump()

@app.post("/step")
async def step(action: AuditAction):
    global env
    if env is None:
        return {"error": "call /reset first"}
    observation, reward, done, info = env.step(action)

    return {
        "observation": observation.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info.model_dump()
    }

@app.get("/state")
async def state():
    global env
    if env is None:
        return {"error": "call /reset first"}
    return env.state().model_dump()



