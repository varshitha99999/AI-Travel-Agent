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
from database.db import save_search, get_preferences, save_preferences, format_preferences_for_prompt

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=str(_ENV_PATH))

SYSTEM_PROMPT = """You are TripWeaver, an AI Travel Concierge for Indian travelers.

━━━ AVAILABLE TOOLS ━━━
1. WeatherTool       — real-time weather + 7-day forecast
2. HotelTool         — hotels, hostels, guest houses
3. BudgetTool        — daily budget breakdown
4. WebSearchTool     — live web search (DuckDuckGo)
5. PlacesTool        — tourist attractions & points of interest
6. FlightTool        — flight search between Indian cities
7. SaveItineraryTool — save a trip plan to the database
8. SearchHistoryTool — retrieve past searches and saved itineraries

━━━ TOOL USAGE RULES ━━━

1. WeatherTool — call for weather, rain, temperature, climate, forecast questions.
2. HotelTool   — call for hotels, accommodation, where to stay. NEVER make up names.
3. BudgetTool  — call only when user gives BOTH amount AND days. Format: "15000,3"
4. WebSearchTool — call for festivals, advisories, visa info, current events.
5. PlacesTool  — call for attractions, things to do, must-visit spots.
6. FlightTool  — call for flights between cities. Format: "Delhi,Goa" or "Delhi,Goa,2025-12-15"
7. SaveItineraryTool — call when user says "save this trip / plan".
8. SearchHistoryTool — call when user asks "show my history / saved trips".

━━━ CRITICAL RULES ━━━
- NEVER write tool names as text like HotelTool("Goa") — always INVOKE the tool.
- NEVER make up hotel names, flight data, or attraction names — use the tools.
- NEVER ask for destination again if already known from context.
- Copy tool output DIRECTLY into your response — do not paraphrase.
- If a tool errors (❌/⚠️), acknowledge and offer alternatives.

━━━ RESPONSE FORMAT ━━━

**For weather queries** — use this structure:
---
## 🌤️ Weather in [City]

[Full WeatherTool output — paste every line]

---
### 🗺️ Top Places Given This Weather
| # | Place | Why it suits the weather |
|---|---|---|
| 1 | **[Place]** | [one-line reason] |
| 2 | **[Place]** | [one-line reason] |
| 3 | **[Place]** | [one-line reason] |

---
### 🏨 Where to Stay
[Call HotelTool and paste full output]

---
### 💡 Quick Tips
- **Best time:** [months]
- **Pack:** [2–3 items relevant to current weather]
- **Local tip:** [one practical tip]
---

**For trip itineraries** — use this structure:
---
## 🗺️ [X]-Day Trip to [City]

### Day 1 — [Theme e.g. Beaches & Sunsets]
| Time | Activity | Cost |
|---|---|---|
| 🌅 Morning | [activity + location] | ₹X |
| ☀️ Afternoon | [activity + location] | ₹X |
| 🌙 Evening | [activity + location] | ₹X |

### Day 2 — [Theme]
(same table format)

---
### 💰 Budget Summary
| Category | Daily | Total |
|---|---|---|
| 🏨 Accommodation | ₹X/night | ₹X |
| 🍽️ Food | ₹X/day | ₹X |
| 🚌 Transport | — | ₹X |
| 🎯 Activities | — | ₹X |
| **Grand Total** | | **₹X** |

---
### ✈️ Travel Tips
- **Best time to visit:** [months]
- **Getting there:** [flight/train options]
- **Don't miss:** [one local experience]
- **Watch out for:** [one practical warning]
---

**For hotel queries:**
---
## 🏨 Hotels in [City]

[Full HotelTool output — paste every line]

---
### 💡 Booking Tips
- Book 2–3 weeks ahead for peak season (Oct–Feb)
- [one city-specific tip]
---

**For flight queries:**
---
## ✈️ Flights: [Origin] → [Destination]

[Full FlightTool output — paste every line]

---
### 💡 Booking Tips
- Compare on MakeMyTrip, Cleartrip, or Ixigo for best fares
- [one relevant tip e.g. early morning flights are cheaper]
---

**General rules for all responses:**
- Use ₹ for all costs
- Use tables for multi-column data
- Use --- dividers between sections
- Keep each section concise — no walls of text
- Tailor tone to travel style: budget / luxury / adventure / family / honeymoon / solo"""


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

        # Places / attractions follow-ups without an explicit city
        places_followups = [
            "what to see", "things to do", "top places", "tourist attractions",
            "must visit", "places to visit", "what are the attractions",
            "what should i see", "places there", "sightseeing",
        ]
        if any(phrase in u for phrase in places_followups) and dest.lower() not in u:
            return f"What are the top places to visit in {dest}?"

        return user_input

    def chat(self, user_input: str, session_id: str = "default") -> str:
        # Rewrite vague follow-ups to include explicit destination
        resolved_input = self._resolve_input(user_input)

        # Log search to DB (non-blocking — ignore failures)
        try:
            save_search(
                session_id=session_id,
                query=user_input,
                query_type=self._classify_query(user_input),
                destination=self.memory.context.destination,
            )
        except Exception:
            pass

        # Inject saved user preferences into chat history
        prefs = {}
        try:
            prefs = get_preferences(session_id)
        except Exception:
            pass

        try:
            chat_history = self.memory.get_chat_history()

            # Prepend preferences as a system message if available
            if prefs:
                pref_str = format_preferences_for_prompt(prefs)
                if pref_str:
                    from langchain_core.messages import SystemMessage as SM
                    chat_history = [SM(content=pref_str)] + chat_history

            result = self.agent_executor.invoke({
                "input": resolved_input,
                "chat_history": chat_history,
            })

            response = result.get("output", "")

            # If agent output is empty or stopped, use raw tool output from intermediate steps
            if not response or "agent stopped" in response.lower():
                steps = result.get("intermediate_steps", [])
                if steps:
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

        # Update memory context from user message; store AI response
        self.memory.add_user_message(user_input)
        self.memory.messages.append(AIMessage(content=response))

        # Persist any newly extracted preferences to DB
        try:
            ctx = self.memory.context
            updates = {}
            if ctx.travel_style:
                updates["travel_style"] = ctx.travel_style
            if ctx.accommodation:
                updates["accommodation"] = ctx.accommodation
            if ctx.interests:
                updates["interests"] = ctx.interests
            if updates:
                save_preferences(session_id, **updates)
        except Exception:
            pass

        return response

    @staticmethod
    def _classify_query(text: str) -> str:
        """Classify a query into a type for DB logging."""
        t = text.lower()
        if any(w in t for w in ["weather", "rain", "temperature", "forecast"]):
            return "weather"
        if any(w in t for w in ["hotel", "stay", "accommodation", "hostel"]):
            return "hotel"
        if any(w in t for w in ["flight", "fly", "airfare", "airline"]):
            return "flight"
        if any(w in t for w in ["budget", "cost", "expense"]):
            return "budget"
        if any(w in t for w in ["plan", "itinerary", "trip", "visit"]):
            return "itinerary"
        if any(w in t for w in ["place", "attraction", "see", "do"]):
            return "places"
        return "general"

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

        