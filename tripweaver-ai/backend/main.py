from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agent.planner import PlannerError, TripPlanner
from models.request_model import TripPlan, TripRequest


app = FastAPI(title="TripWeaver AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

planner = TripPlanner()


@app.post("/plan-trip", response_model=TripPlan)
async def plan_trip(request: TripRequest) -> TripPlan:
    try:
        return await planner.plan_trip(request)
    except PlannerError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
