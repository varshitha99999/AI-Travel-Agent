import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

from agent.memory import TravelMemory
from agent.tools import ALL_TOOLS

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=str(_ENV_PATH))

SYSTEM_PROMPT = """You are TripWeaver, an AI Travel Concierge for Indian travelers.

━━━ TOOL USAGE RULES (follow strictly) ━━━

1. WeatherTool — call this when the user asks about weather, rain, temperature, climate, or forecast.
   - "what's the weather in Goa" → call WeatherTool("Goa")
   - "what is the weather there" / "what's the weather" → use destination from context, call WeatherTool("Goa")
   - After getting the tool result, FIRST copy the COMPLETE weather tool output word for word, THEN suggest places.
   - NEVER respond to a weather question with an itinerary. NEVER skip calling WeatherTool.
   - The weather tool output starts with "🌤 **Current Weather in..." — copy everything from that line until "_Source:..." including temperature, humidity, condition, wind, travel advice, and 7-day forecast.
   - Response format for weather:
     [PASTE FULL WeatherTool output here — every single line]

     **🗺️ Best Places to Visit in [Destination] Given This Weather:**
     - **[Place 1]** — [one line: why it suits current weather/temperature]
     - **[Place 2]** — [one line: why it suits current weather/temperature]
     - **[Place 3]** — [one line: why it suits current weather/temperature]

2. HotelTool — call this when the user asks about hotels, accommodation, where to stay, or hostels.
   - "suggest hotels in Jaipur" → call HotelTool("Jaipur")
   - "what about hotels?" → use destination from context, call HotelTool
   - NEVER make up hotel names. ALWAYS call HotelTool.

3. BudgetTool — call this when the user provides BOTH a total amount AND number of days.
   - "my budget is ₹15000 for 3 days" → call BudgetTool("15000,3")
   - "budget is 15000 INR for 3 days" → call BudgetTool("15000,3")
   - "what will the budget be?" → DO NOT call BudgetTool. Summarise costs from the itinerary already given.
   - INPUT must be exactly "AMOUNT,DAYS" — extract only the digits, e.g. "15000,3"
   - Call BudgetTool ONCE only. Do not retry if it returns a result.

4. Itineraries, packing lists, travel tips → use your own knowledge, NO tools needed.

━━━ FOLLOW-UP QUESTION RULES ━━━
- "what will the budget be?" / "how much will it cost?" → summarise the cost breakdown from the itinerary you already gave. Do NOT call BudgetTool unless the user gives a new total amount.
- "what is the weather there?" / "how's the weather?" → call WeatherTool with the destination already in context. Then present the FULL tool output to the user — do not paraphrase or shorten it.
- "suggest hotels" / "where to stay?" → call HotelTool with the destination already in context. Then present the FULL list of hotels from the tool output — do not paraphrase or say "I found some hotels".
- NEVER ask for the destination again if it is already in the conversation context.

━━━ TOOL OUTPUT RULES ━━━
- After calling WeatherTool: copy the tool output directly into your response. Do not say "I hope this helps" instead of the data.
- After calling HotelTool: copy the full hotel list from the tool output. Do not say "I found some hotels" without listing them.
- After calling BudgetTool: copy the full breakdown from the tool output.
- NEVER replace tool output with a generic message. If a tool returns data, show it.

━━━ EDGE CASES ━━━
- Unknown destination: say you don't have info, suggest nearby popular alternatives.
- Very low budget (under ₹500/day): acknowledge the constraint honestly, suggest hostels/street food/local buses.
- Vague query ("plan a trip"): ask ONE clarifying question — destination, duration, or budget.
- Off-topic queries: politely redirect to travel planning.

━━━ RESPONSE FORMAT ━━━

For trip itineraries:
**Day 1: [Theme]**
- 🌅 Morning: [activity] — ₹X
- ☀️ Afternoon: [activity] — ₹X
- 🌙 Evening: [activity] — ₹X

(repeat for each day)

**💰 Budget Summary**
- 🏨 Accommodation: ₹X/night
- 🍽 Food: ₹X/day
- 🚌 Transport: ₹X total
- 🎯 Activities: ₹X total
- **Total Estimate: ₹X**

**✈️ Travel Tips**
- [practical tip]
- [best time / what to pack / local advice]

For all responses:
- Use **bold** for section headings
- Use bullet points and emojis for readability
- Always use ₹ for costs
- Keep responses concise — avoid walls of text
- Tailor suggestions to travel style (budget/luxury/adventure/family/honeymoon/solo)"""


class TripPlanner:
    def __init__(self):
        # Initialize LangChain ChatGroq with tool-calling support
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=0.4,
            max_tokens=2048,
            timeout=30,
        )

        # Initialize memory
        self.memory = TravelMemory()

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)

        # Prompt template for tool-calling agent
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Use llm_with_tools so tool-calling is reliably triggered
        agent = create_tool_calling_agent(self.llm_with_tools, ALL_TOOLS, self.prompt)

        # Agent executor with error handling
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=ALL_TOOLS,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=5,
            max_execution_time=60,
            return_intermediate_steps=True,
        )

    def _resolve_input(self, user_input: str) -> str:
        """
        If the user asks a follow-up weather/hotel question without naming a destination,
        inject the known destination from memory so the agent always gets an explicit city.
        """
        u = user_input.lower().strip()
        dest = self.memory.context.destination

        if not dest:
            return user_input

        # Weather follow-ups without an explicit city
        weather_followups = [
            "what is the weather there", "what's the weather there",
            "how's the weather", "how is the weather", "weather there",
            "weather in that place", "what about the weather",
            "will it rain", "is it hot", "is it cold", "what is the climate",
            "what's the weather", "what is the weather", "hows the weather",
            "weather forecast", "check weather", "tell me the weather",
        ]
        if any(phrase in u for phrase in weather_followups) and dest.lower() not in u:
            return f"What is the weather in {dest}?"

        # Hotel follow-ups without an explicit city
        hotel_followups = [
            "suggest hotels", "find hotels", "what about hotels",
            "where to stay", "hotels there", "accommodation there",
            "suggest accommodation", "find hostels", "hotels within my budget",
            "suggest hotels within my budget",
        ]
        if any(phrase in u for phrase in hotel_followups) and dest.lower() not in u:
            return f"Suggest hotels in {dest}"

        return user_input

    def chat(self, user_input: str) -> str:
        # Rewrite vague follow-ups to include explicit destination
        resolved_input = self._resolve_input(user_input)
        try:
            chat_history = self.memory.get_chat_history()

            result = self.agent_executor.invoke({
                "input": resolved_input,
                "chat_history": chat_history,
            })

            response = result.get("output", "")

            # If agent output is empty or stopped, use raw tool output from intermediate steps
            if not response or "agent stopped" in response.lower():
                steps = result.get("intermediate_steps", [])
                if steps:
                    # Grab the last tool output
                    tool_output = steps[-1][1] if steps else ""
                    if tool_output:
                        response = tool_output
                    else:
                        raise ValueError("Empty response")
                else:
                    raise ValueError("Empty response")

        except Exception:
            # Fallback: use plain LLM chain without tools
            try:
                fallback_prompt = ChatPromptTemplate.from_messages([
                    ("system", SYSTEM_PROMPT),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                ])
                fallback_chain = fallback_prompt | self.llm | StrOutputParser()
                response = fallback_chain.invoke({
                    "input": resolved_input,
                    "chat_history": self.memory.get_chat_history(),
                })
            except Exception as e:
                return f"I apologize, I encountered an error: {str(e)}. Please try again."

        # Only user messages update travel context; AI messages stored directly
        self.memory.add_user_message(user_input)
        self.memory.messages.append(AIMessage(content=response))
        return response

    def clear_memory(self):
        self.memory.clear_memory()

    def get_conversation_summary(self) -> str:
        chat_history = self.memory.get_chat_history()
        if not chat_history:
            return "No conversation history available."
        summary = "Recent conversation:\n"
        for msg in chat_history[-4:]:
            if isinstance(msg, HumanMessage):
                summary += f"User: {msg.content[:100]}...\n"
            elif isinstance(msg, AIMessage):
                summary += f"AI: {msg.content[:100]}...\n"
        return summary

        