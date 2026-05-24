"""
Agent Tools
-----------
All LangChain tools available to the TripWeaver agent.

Tools:
  1. budget_tool          — daily budget breakdown calculator
  2. weather_tool         — real-time weather + 7-day forecast (Open-Meteo)
  3. hotel_tool           — hotels / hostels / guest houses (OpenStreetMap / Amadeus)
  4. web_search_tool      — DuckDuckGo web search (no API key required)
  5. places_tool          — tourist attractions & POIs (OpenTripMap / static fallback)
  6. flight_tool          — flight search between Indian cities (Amadeus / static)
  7. save_itinerary_tool  — save a generated itinerary to SQLite
  8. search_history_tool  — retrieve past searches and saved itineraries
"""

from langchain_core.tools import tool

from agent.budget import calculate_budget
from agent.error_handler import safe_tool_call, validate_city, validate_budget_input, ToolError
from services.weather import get_weather
from services.hotels import get_hotels
from services.web_search import get_web_search
from services.places import get_places
from services.flights import get_flights
from database.db import (
    save_search, save_itinerary, get_recent_searches,
    get_itineraries, get_popular_destinations,
)


# ── 1. Budget Tool ────────────────────────────────────────────────────────────

@tool
def budget_tool(input_text: str) -> str:
    """Calculate a daily travel budget breakdown from a total budget and trip duration.

    WHEN TO USE:
    - User explicitly states a NEW total amount AND number of days in the current message.
    - Examples: "my budget is ₹15000 for 3 days", "I have 50000 INR for 7 days"

    DO NOT USE:
    - When the user asks "what will the budget be?" without giving a new amount.
    - When only one of amount or days is mentioned.

    INPUT FORMAT: 'AMOUNT,DAYS' — digits only, no spaces, no currency symbols.
    Examples: '15000,3'  |  '50000,7'  |  '8000,2'
    """
    try:
        validate_budget_input(input_text)   # raises ToolError on bad input
    except ToolError as exc:
        return f"❌ {exc}"
    return safe_tool_call(calculate_budget, input_text, tool_name="BudgetTool")


# ── 2. Weather Tool ───────────────────────────────────────────────────────────

@tool
def weather_tool(city: str) -> str:
    """Get real-time weather conditions and a 7-day forecast for a travel destination.

    WHEN TO USE:
    - User asks about weather, climate, temperature, rain, or forecast at any destination.
    - Phrases like "what's the weather in X", "will it rain in X", "is it cold in X".
    - ALWAYS call this tool for weather questions — never answer from memory.

    INPUT: City name only — no country, no extra words.
    Examples: 'Goa'  |  'Manali'  |  'Jaipur'  |  'Mumbai'  |  'Darjeeling'
    """
    try:
        city = validate_city(city)
    except ToolError as exc:
        return f"❌ {exc}"
    return safe_tool_call(get_weather, city, tool_name="WeatherTool")


# ── 3. Hotel Tool ─────────────────────────────────────────────────────────────

@tool
def hotel_tool(city: str) -> str:
    """Find hotels, hostels, and guest houses in a travel destination.

    WHEN TO USE:
    - User asks about hotels, accommodation, places to stay, or lodging.
    - Phrases like "suggest hotels in X", "where to stay in X", "what about hotels?".
    - ALWAYS call this tool for hotel questions — never make up hotel names.

    INPUT: City name only — no country, no extra words.
    Examples: 'Goa'  |  'Manali'  |  'Jaipur'  |  'Mumbai'  |  'Rishikesh'
    """
    try:
        city = validate_city(city)
    except ToolError as exc:
        return f"❌ {exc}"
    return safe_tool_call(get_hotels, city, tool_name="HotelTool")


# ── 4. Web Search Tool ────────────────────────────────────────────────────────

@tool
def web_search_tool(query: str) -> str:
    """Search the web for current travel information, news, events, or tips.

    WHEN TO USE:
    - User asks about current events, festivals, travel advisories, or visa requirements.
    - User asks about something not covered by weather/hotel/budget tools.
    - Phrases like "is it safe to travel to X", "what festivals are in X in [month]",
      "latest travel news for X", "visa requirements for X".
    - Use for any question that needs up-to-date information from the internet.

    DO NOT USE:
    - For weather (use WeatherTool), hotels (use HotelTool), or budget (use BudgetTool).
    - For general itinerary planning — use your own knowledge for that.

    INPUT: A clear, specific search query.
    Examples:
      'best time to visit Rajasthan travel tips'
      'Goa carnival 2025 dates'
      'India tourist visa requirements for foreigners'
      'is Manali safe to visit in January'
    """
    if not query.strip():
        return "❌ Search query cannot be empty."
    return safe_tool_call(get_web_search, query, tool_name="WebSearchTool")


# ── 5. Places Tool ────────────────────────────────────────────────────────────

@tool
def places_tool(city: str) -> str:
    """Get top tourist attractions and points of interest for a travel destination.

    WHEN TO USE:
    - User asks "what are the top places to visit in X?", "what to see in X?",
      "tourist attractions in X", "things to do in X", "must-visit spots in X".
    - Use this to enrich itinerary suggestions with real attraction names.
    - ALWAYS call this tool for attraction questions — never make up place names.

    INPUT: City name only.
    Examples: 'Goa'  |  'Jaipur'  |  'Manali'  |  'Delhi'  |  'Mumbai'  |  'Kerala'
    """
    try:
        city = validate_city(city)
    except ToolError as exc:
        return f"❌ {exc}"
    return safe_tool_call(get_places, city, tool_name="PlacesTool")


# ── 6. Flight Tool ────────────────────────────────────────────────────────────

@tool
def flight_tool(input_text: str) -> str:
    """Search for flights between two Indian cities.

    WHEN TO USE:
    - User asks about flights, airfare, how to fly between cities.
    - Phrases like "flights from Delhi to Goa", "how to fly to Manali",
      "cheapest flight from Mumbai to Jaipur", "flight options to Kerala".

    INPUT FORMAT: 'ORIGIN,DESTINATION' or 'ORIGIN,DESTINATION,DATE'
    - City names only (no airport codes needed).
    - Date is optional — defaults to tomorrow if not given.
    - Date format: YYYY-MM-DD or DD/MM/YYYY

    Examples:
      'Delhi,Goa'
      'Mumbai,Jaipur'
      'Delhi,Leh,2025-12-15'
      'Bangalore,Goa,25/12/2025'
    """
    parts = [p.strip() for p in input_text.split(",")]
    if len(parts) < 2:
        return "❌ Please provide origin and destination. Format: 'ORIGIN,DESTINATION' e.g. 'Delhi,Goa'"

    origin = parts[0]
    destination = parts[1]
    travel_date = parts[2] if len(parts) >= 3 else None

    try:
        origin = validate_city(origin)
        destination = validate_city(destination)
    except ToolError as exc:
        return f"❌ {exc}"

    return safe_tool_call(get_flights, origin, destination, travel_date, tool_name="FlightTool")


# ── 7. Save Itinerary Tool ────────────────────────────────────────────────────

@tool
def save_itinerary_tool(input_text: str) -> str:
    """Save a generated trip itinerary to the database for future reference.

    WHEN TO USE:
    - User says "save this itinerary", "save my trip plan", "remember this plan",
      "save this for later", "bookmark this trip".
    - After generating an itinerary, if the user wants to keep it.

    INPUT FORMAT: 'SESSION_ID|DESTINATION|CONTENT'
    - SESSION_ID: the current session identifier
    - DESTINATION: city name
    - CONTENT: the full itinerary text to save

    Example: 'session_abc|Goa|Day 1: Baga Beach...'

    Note: The agent should use the current session_id from context.
    If session_id is unknown, use 'default'.
    """
    parts = input_text.split("|", 2)
    if len(parts) < 3:
        return "❌ Format: 'SESSION_ID|DESTINATION|CONTENT'"

    session_id, destination, content = parts[0].strip(), parts[1].strip(), parts[2].strip()
    if not destination or not content:
        return "❌ Destination and content cannot be empty."

    try:
        destination = validate_city(destination)
    except ToolError as exc:
        return f"❌ {exc}"

    try:
        row_id = save_itinerary(
            session_id=session_id or "default",
            destination=destination,
            content=content,
        )
        return (
            f"✅ **Itinerary saved!**\n\n"
            f"📍 Destination: {destination}\n"
            f"🔖 ID: #{row_id}\n\n"
            f"You can retrieve it anytime by asking 'show my saved trips'."
        )
    except Exception as exc:
        return f"❌ Could not save itinerary: {exc}"


# ── 8. Search History Tool ────────────────────────────────────────────────────

@tool
def search_history_tool(session_id: str) -> str:
    """Retrieve the user's recent searches and saved itineraries from the database.

    WHEN TO USE:
    - User asks "what have I searched for?", "show my history",
      "what trips have I saved?", "show my saved itineraries",
      "what did I plan before?", "my recent searches".

    INPUT: The current session_id string.
    If unknown, use 'default'.
    """
    session_id = session_id.strip() or "default"

    try:
        searches = get_recent_searches(session_id, limit=8)
        itineraries = get_itineraries(session_id, limit=5)
        popular = get_popular_destinations(limit=5)

        lines = ["📋 **Your Travel History**\n"]

        if searches:
            lines.append("**🔍 Recent Searches:**")
            for s in searches:
                dest = f" → {s['destination']}" if s.get("destination") else ""
                lines.append(f"  • {s['query'][:60]}{dest}  _{s['created_at'][:10]}_")
        else:
            lines.append("**🔍 Recent Searches:** None yet")

        lines.append("")

        if itineraries:
            lines.append("**💾 Saved Itineraries:**")
            for it in itineraries:
                days = f" · {it['days']} days" if it.get("days") else ""
                budget = f" · ₹{it['budget']:,.0f}" if it.get("budget") else ""
                lines.append(
                    f"  • **{it['destination']}**{days}{budget}  "
                    f"_(#{it['id']}, saved {it['created_at'][:10]})_"
                )
        else:
            lines.append("**💾 Saved Itineraries:** None yet — ask me to save a trip plan!")

        if popular:
            lines.append("\n**🌟 Popular Destinations (all users):**")
            for p in popular:
                lines.append(f"  • {p['destination']} ({p['count']} searches)")

        return "\n".join(lines)

    except Exception as exc:
        return f"❌ Could not retrieve history: {exc}"


# ── Tool registry ─────────────────────────────────────────────────────────────

ALL_TOOLS = [
    budget_tool,
    weather_tool,
    hotel_tool,
    web_search_tool,
    places_tool,
    flight_tool,
    save_itinerary_tool,
    search_history_tool,
]

# Metadata for UI display
TOOL_METADATA = {
    "budget_tool":         {"icon": "💰", "label": "Budget Calculator",   "api": "Built-in"},
    "weather_tool":        {"icon": "🌤", "label": "Live Weather",         "api": "Open-Meteo (free)"},
    "hotel_tool":          {"icon": "🏨", "label": "Hotel Finder",         "api": "OpenStreetMap / Amadeus"},
    "web_search_tool":     {"icon": "🔍", "label": "Web Search",           "api": "DuckDuckGo (free)"},
    "places_tool":         {"icon": "🗺️", "label": "Places & Attractions", "api": "OpenTripMap / static"},
    "flight_tool":         {"icon": "✈️", "label": "Flight Search",        "api": "Amadeus / static"},
    "save_itinerary_tool": {"icon": "💾", "label": "Save Itinerary",       "api": "SQLite (local)"},
    "search_history_tool": {"icon": "📋", "label": "Search History",       "api": "SQLite (local)"},
}
