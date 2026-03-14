import chainlit as cl
from agent.planner import TripPlanner


QUICK_ACTIONS = [
    cl.Action(name="weather", label="🌤 Check Weather", payload={"query": "What's the weather in Goa?"}),
    cl.Action(name="hotels", label="🏨 Find Hotels", payload={"query": "Suggest hotels in Jaipur"}),
    cl.Action(name="budget", label="💰 Budget Help", payload={"query": "My budget is 15000 INR 3 days. What's my daily budget?"}),
    cl.Action(name="plan", label="📅 Plan a Trip", payload={"query": "Plan a 3-day budget trip to Manali"}),
]


def _detect_type(user_input: str) -> str:
    u = user_input.lower()
    if any(w in u for w in ["weather", "climate", "rain", "forecast", "temperature"]):
        return "weather"
    if any(w in u for w in ["hotel", "stay", "accommodation", "hostel", "resort"]):
        return "hotel"
    if any(w in u for w in ["budget", "cost", "calculate", "expense", "daily"]):
        return "budget"
    if any(w in u for w in ["plan", "visit", "travel to"]):
        return "itinerary"
    return "general"


def _add_header(response: str, response_type: str) -> str:
    headers = {
        "weather":   "## 🌤 Weather Report\n\n",
        "hotel":     "## 🏨 Accommodation Options\n\n",
        "budget":    "## 💰 Budget Breakdown\n\n",
        "itinerary": "## 🗺️ Your Travel Plan\n\n",
        "general":   "",
    }
    footer = "\n\n---\n💡 *Ask me anything about your trip!*"
    return headers.get(response_type, "") + response + footer


def _context_bar(ctx) -> str:
    """Build a compact context status line"""
    parts = []
    if ctx.destination:
        parts.append(f"📍 {ctx.destination}")
    if ctx.days:
        parts.append(f"📅 {ctx.days}d")
    if ctx.budget:
        parts.append(f"💰 ₹{ctx.budget}")
    if ctx.travel_style:
        parts.append(f"🎯 {ctx.travel_style.title()}")
    if ctx.accommodation:
        parts.append(f"🏨 {ctx.accommodation.title()}")
    return "  ·  ".join(parts) if parts else "No context yet"


@cl.on_chat_start
async def start():
    planner = TripPlanner()
    cl.user_session.set("planner", planner)

    await cl.Message(
        content=(
            "# 🌍 AI Travel Concierge ✈️\n\n"
            "Welcome! I'm your personal travel assistant for India.\n\n"
            "**What I can do:**\n"
            "| | Feature | Example |\n"
            "|---|---|---|\n"
            "| 📅 | Trip Planning | *Plan a 3-day trip to Goa* |\n"
            "| 💰 | Budget Calculator | *Budget is 15000 INR for 3 days* |\n"
            "| 🌤 | Live Weather | *What's the weather in Manali?* |\n"
            "| 🏨 | Hotel Finder | *Suggest hotels in Jaipur* |\n"
            "| 🎯 | Travel Tips | *Best time to visit Kerala?* |\n\n"
            "**Or pick a quick action below 👇**"
        ),
        actions=QUICK_ACTIONS,
    ).send()


@cl.action_callback("weather")
@cl.action_callback("hotels")
@cl.action_callback("budget")
@cl.action_callback("plan")
async def on_quick_action(action: cl.Action):
    """Handle quick action button clicks"""
    query = action.payload.get("query", "")
    await cl.Message(content=f"_{query}_").send()
    await _handle_query(query)


@cl.on_message
async def main(message: cl.Message):
    await _handle_query(message.content)


async def _handle_query(user_input: str):
    """Core handler — runs agent and streams formatted response"""
    planner = cl.user_session.get("planner")

    # Show context bar while thinking
    async with cl.Step(name="🤖 Thinking...", type="tool") as step:
        step.input = user_input
        response = planner.chat(user_input)
        step.output = _context_bar(planner.memory.context)

    # Format and send response
    response_type = _detect_type(user_input)
    formatted = _add_header(response, response_type)
    await cl.Message(content=formatted).send()


@cl.on_chat_end
async def end():
    planner = cl.user_session.get("planner")
    if planner:
        planner.clear_memory()


if __name__ == "__main__":
    cl.run()
