from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from app.env import FundSageEnv
from graders.task1 import grade_task_1
from graders.task2 import grade_task_2
from graders.task3 import grade_task_3

app = FastAPI(title="FundSage OpenEnv", version="1.0.0")
env = FundSageEnv()

class Action(BaseModel):
    selected_funds: list[str]
    allocation:     list[float]

class GradeRequest(BaseModel):
    task_id: str
    action:  dict
    state:   dict

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/reset")
def reset():
    return JSONResponse(content={"state": env.reset()})

@app.get("/state")
def state():
    return JSONResponse(content={"state": env.state()})

@app.post("/step")
def step(action: Action):
    return JSONResponse(content=env.step(action.model_dump()))

@app.get("/tasks")
def tasks():
    return JSONResponse(content={"tasks": [
        {"id": "easy_risk_match",       "difficulty": "easy",   "description": "Recommend low-risk capital preservation funds."},
        {"id": "balanced_portfolio",    "difficulty": "medium", "description": "Build a balanced diversified portfolio."},
        {"id": "high_return_optimized", "difficulty": "hard",   "description": "Maximise returns under expense, volatility and tax constraints."},
    ]})

@app.post("/grade")
def grade(req: GradeRequest):
    graders = {
        "easy_risk_match":       grade_task_1,
        "balanced_portfolio":    grade_task_2,
        "high_return_optimized": grade_task_3,
    }
    grader = graders.get(req.task_id)
    if not grader:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {req.task_id}")
    return JSONResponse(content={"task_id": req.task_id, "score": grader(req.action, req.state)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
