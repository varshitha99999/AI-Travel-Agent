import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from agent.budget import calculate_budget
from services.weather import get_weather
from services.hotels import search_hotels

load_dotenv()


class TripPlanner:
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
            temperature=0.7,
        )
        self.conversation_history = []

    def chat(self, user_input: str) -> str:
        try:
            # Add user message to history
            self.conversation_history.append(f"User: {user_input}")
            
            # Check if user wants specific tools
            response = self._process_with_tools(user_input)
            
            # Add AI response to history
            self.conversation_history.append(f"AI: {response}")
            
            return response
        except Exception as e:
            return f"Error: {str(e)}"

    def _process_with_tools(self, user_input: str) -> str:
        user_lower = user_input.lower()
        
        # Check for budget calculation
        if "budget" in user_lower and "," in user_input:
            # Extract budget parameters
            parts = [p.strip() for p in user_input.split() if "," in p]
            if parts:
                budget_result = calculate_budget(parts[0])
                return f"Budget calculation: {budget_result}"
        
        # Check for weather request
        if "weather" in user_lower:
            # Extract destination
            words = user_input.split()
            for i, word in enumerate(words):
                if word.lower() in ["in", "for", "at"] and i + 1 < len(words):
                    destination = words[i + 1].replace("?", "").replace(".", "")
                    weather_result = get_weather(destination)
                    return f"Weather info: {weather_result}"
        
        # Check for hotel request
        if any(word in user_lower for word in ["hotel", "stay", "accommodation"]):
            # Extract destination
            words = user_input.split()
            for i, word in enumerate(words):
                if word.lower() in ["in", "for", "at"] and i + 1 < len(words):
                    destination = words[i + 1].replace("?", "").replace(".", "")
                    hotel_result = search_hotels(destination)
                    return f"Hotel suggestions: {hotel_result}"
        
        # For general travel planning, use LLM
        return self._get_llm_response(user_input)

    def _get_llm_response(self, user_input: str) -> str:
        system_prompt = """You are an AI Travel Concierge for Indian travelers.

You help with:
- Trip planning and itineraries
- Travel advice and tips
- Destination recommendations
- Budget planning guidance

Be helpful, practical, and assume all costs are in INR unless specified otherwise.
Keep responses concise and actionable."""

        # Include recent conversation history for context
        context = ""
        if self.conversation_history:
            recent_history = self.conversation_history[-4:]  # Last 2 exchanges
            context = "\n".join(recent_history) + "\n"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{context}User: {user_input}")
        ]

        response = self.llm.invoke(messages)
        return response.content