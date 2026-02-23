from typing import List

from pydantic import BaseModel, Field, conint, confloat


class TripRequest(BaseModel):
    destination: str = Field(..., min_length=1)
    days: conint(gt=0) = Field(..., description="Number of days for the trip")
    budget: confloat(gt=0) = Field(..., description="Total budget for the trip in INR")
    interests: List[str] = Field(default_factory=list)


class DayPlan(BaseModel):
    day: conint(gt=0)
    activities: List[str]
    estimated_cost: confloat(ge=0)


class TripPlan(BaseModel):
    destination: str
    total_estimated_budget: confloat(ge=0)
    day_wise_plan: List[DayPlan]
    travel_tips: List[str]
