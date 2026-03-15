from langchain_core.tools import tool
from agent.budget import calculate_budget
from services.weather import get_weather
from services.hotels import get_hotels


@tool
def budget_tool(input_text: str) -> str:
    """Calculate a daily travel budget breakdown from a total budget and trip duration.

    WHEN TO USE:
    - User explicitly states a NEW total amount AND number of days in the current message.
    - Examples: "my budget is ₹15000 for 3 days", "I have 50000 INR for 7 days"

    DO NOT USE:
    - When the user asks "what will the budget be?" or "how much will it cost?" — those are
      follow-up questions about an itinerary already given; summarise from that instead.
    - When only one of amount or days is mentioned.
    - When no specific numbers are provided.

    INPUT FORMAT: 'AMOUNT,DAYS' — digits only, no spaces, no currency symbols.
    Examples: '15000,3'  |  '50000,7'  |  '8000,2'
    """
    return calculate_budget(input_text)


@tool
def weather_tool(city: str) -> str:
    """Get real-time weather conditions and a 7-day forecast for a travel destination.

    WHEN TO USE:
    - User asks about weather, climate, temperature, rain, or forecast at any destination.
    - Phrases like "what's the weather in X", "will it rain in X", "is it cold in X",
      "what is the weather there" (use destination from context).
    - ALWAYS call this tool for weather questions — never answer from memory.

    INPUT: City name only — no country, no extra words.
    Examples: 'Goa'  |  'Manali'  |  'Jaipur'  |  'Mumbai'  |  'Darjeeling'
    """
    result = get_weather(city)
    if not result:
        return f"⚠️ Could not fetch weather data for {city}. Please try again later."
    return result


@tool
def hotel_tool(city: str) -> str:
    """Find hotels, hostels, and guest houses in a travel destination.

    WHEN TO USE:
    - User asks about hotels, accommodation, places to stay, or lodging.
    - Phrases like "suggest hotels in X", "where to stay in X", "what about hotels?"
      (use destination from context).
    - ALWAYS call this tool for hotel questions — never make up hotel names.

    INPUT: City name only — no country, no extra words.
    Examples: 'Goa'  |  'Manali'  |  'Jaipur'  |  'Mumbai'  |  'Rishikesh'
    """
    result = get_hotels(city)
    if not result:
        return f"⚠️ Could not fetch hotel data for {city}. Please try again later."
    return result


# All tools registered for the agent
ALL_TOOLS = [budget_tool, weather_tool, hotel_tool]
