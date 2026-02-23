import json
import os
from typing import Any, Dict

import groq
from groq import AsyncGroq

from agent.prompts import build_system_prompt, build_user_prompt
from models.request_model import TripPlan, TripRequest


class PlannerError(Exception):
    pass


class TripPlanner:
    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise PlannerError("GROQ_API_KEY is not set")
        self.client = AsyncGroq(api_key=api_key)
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    async def plan_trip(self, request: TripRequest) -> TripPlan:
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(request)
        raw_content = await self._call_groq(system_prompt, user_prompt)
        data = self._parse_response(raw_content)
        return TripPlan(**data)

    async def _call_groq(self, system_prompt: str, user_prompt: str) -> str:
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
            )
            content = completion.choices[0].message.content
            if not content:
                raise PlannerError("Empty response from Groq")
            return content
        except groq.APIError as exc:
            raise PlannerError(f"Error calling Groq API: {exc}") from exc
        except Exception as exc:
            raise PlannerError(f"Unexpected error calling Groq API: {exc}") from exc

    def _parse_response(self, content: str) -> Dict[str, Any]:
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise PlannerError("Model response was not valid JSON") from exc

        required_keys = {"destination", "total_estimated_budget", "day_wise_plan", "travel_tips"}
        if not required_keys.issubset(data.keys()):
            raise PlannerError("Model response missing required fields")

        if not isinstance(data["day_wise_plan"], list):
            raise PlannerError("day_wise_plan must be a list")

        return data
