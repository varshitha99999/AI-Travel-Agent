from models.request_model import TripRequest


def build_system_prompt() -> str:
    return (
        "You are an expert travel planner. "
        "Given a destination, number of days, budget, and interests, "
        "you must create an optimized, realistic travel plan for a traveler in India. "
        "Respond strictly as a compact JSON object with this exact structure and keys:\n"
        "{\n"
        '  "destination": "string",\n'
        '  "total_estimated_budget": number,\n'
        '  "day_wise_plan": [\n'
        "    {\n"
        '      "day": number,\n'
        '      "activities": ["string"],\n'
        '      "estimated_cost": number\n'
        "    }\n"
        "  ],\n"
        '  "travel_tips": ["string"]\n'
        "}\n"
        "Do not include any text outside of this JSON. "
        "Costs should be plausible and in INR."
    )


def build_user_prompt(request: TripRequest) -> str:
    interests = ", ".join(request.interests) if request.interests else "general sightseeing"
    return (
        f"Plan a {request.days}-day trip to {request.destination} with a total budget of "
        f"{request.budget} INR. Interests: {interests}."
    )
