import chainlit as cl
from agent.planner import TripPlanner


@cl.on_chat_start
async def start():
    """Initialize the trip planner when chat starts"""
    planner = TripPlanner()
    cl.user_session.set("planner", planner)
    
    # Welcome message
    await cl.Message(
        content="🌍 **Welcome to AI Travel Concierge!** ✈️\n\nI'm here to help you plan amazing trips across India! I can:\n\n• 📅 Plan detailed itineraries\n• 💰 Calculate budgets\n• 🌤️ Check weather info\n• 🏨 Suggest hotels\n• 🎯 Give travel tips\n\n**Try asking:**\n- \"Plan a 3-day trip to Goa with 15000 INR\"\n- \"What's the weather in Manali?\"\n- \"Calculate budget 20000,4\"\n- \"Suggest hotels in Jaipur\"\n\nWhat would you like to explore today?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    planner = cl.user_session.get("planner")
    
    # Show typing indicator
    async with cl.Step(name="thinking", type="tool") as step:
        step.output = "Processing your travel request..."
        
        # Get response from planner
        response = planner.chat(message.content)
    
    # Send response
    await cl.Message(content=response).send()


if __name__ == "__main__":
    cl.run()