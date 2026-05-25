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

━━━ STRICT TOOL ROUTING — READ CAREFULLY ━━━

Match the user's query to EXACTLY ONE primary tool and call it FIRST:

| User asks about | Call this tool FIRST |
|---|---|
| weather / rain / temperature / forecast | weather_tool(city) |
| hotels / stay / accommodation / hostel | hotel_tool(city) |
| budget / cost — with amount AND days given | budget_tool("AMOUNT,DAYS") |
| flights / fly / airfare | flight_tool("ORIGIN,DESTINATION") |
| attractions / places / things to do | places_tool(city) |
| festivals / visa / travel news / safety | web_search_tool(query) |
| save this trip / remember this plan | save_itinerary_tool(...) |
| my history / saved trips | search_history_tool(session_id) |

━━━ ABSOLUTE RULES ━━━
1. Call the tool for the CURRENT query — ignore previous conversation topics.
2. NEVER call hotel_tool when asked about weather. NEVER call places_tool when asked about budget.
3. NEVER write tool syntax as text — always invoke the tool.
4. NEVER make up hotel names, flight numbers, or attraction names.
5. Paste tool output VERBATIM — do not paraphrase or shorten it.
6. For follow-ups ("what about hotels?" / "how's the weather?") — use destination from context.

━━━ BUDGET RULE ━━━
- Call budget_tool ONLY when user gives BOTH a rupee amount AND number of days.
- "My budget is ₹15000 for 3 days" → budget_tool("15000,3")
- "What will it cost?" → estimate from your knowledge, do NOT call budget_tool.

━━━ RESPONSE FORMAT ━━━

**Weather query response:**
## 🌤️ Weather in [City]
[Paste full weather_tool output here — every line]

---
### 🗺️ Best Places Given This Weather
| # | Place | Why it suits the weather |
|---|---|---|
| 1 | **[Place]** | [reason] |
| 2 | **[Place]** | [reason] |
| 3 | **[Place]** | [reason] |

---
### 💡 Quick Tips
- **Pack:** [items for current weather]
- **Tip:** [one local advice]

---

**Hotel query response:**
## 🏨 Hotels in [City]
[Paste full hotel_tool output here — every line]

---
### 💡 Booking Tips
- Book 2–3 weeks ahead for peak season (Oct–Feb)
- [one city-specific tip]

---

**Flight query response:**
## ✈️ Flights: [Origin] → [Destination]
[Paste full flight_tool output here — every line]

---
### 💡 Tips
- Compare on MakeMyTrip, Cleartrip, or Ixigo for best fares
- Early morning flights are usually cheapest

---

**Budget query response:**
## 💰 Budget Breakdown
[Paste full budget_tool output here — every line]

---

**Trip itinerary response:**
## 🗺️ [X]-Day Trip to [City]

### Day 1 — [Theme]
| Time | Activity | Cost |
|---|---|---|
| 🌅 Morning | [activity] | ₹X |
| ☀️ Afternoon | [activity] | ₹X |
| 🌙 Evening | [activity] | ₹X |

(repeat for each day)

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
- **Best time:** [months]
- **Getting there:** [options]
- **Don't miss:** [experience]
- **Watch out for:** [warning]

---

**Places query response:**
## 🗺️ Top Places in [City]
[Paste full places_tool output here — every line]

---

**General rules:**
- Use ₹ for all costs
- Use markdown tables for structured data
- Use --- dividers between sections
- Keep responses concise — no walls of text
- Tailor to travel style: budget / luxury / adventure / family / honeymoon / solo"""

class TripPlanner:
    def __init__(self):
        # Initialize LangChain ChatGroq with tool-calling support
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=0.4,
            max_tokens=1024,   # reduced from 2048 — faster responses
            timeout=20,        # reduced from 30
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
            max_iterations=3,        # reduced from 5 — fewer LLM round-trips
            max_execution_time=30,   # reduced from 60 — fail fast
            return_intermediate_steps=True,
        )

    def _resolve_input(self, user_input: str) -> str:
        """
        Only inject context destination for vague follow-ups that contain
        NO explicit city name. If the user names a city, pass through unchanged.
        """
        u = user_input.lower().strip()
        dest = self.memory.context.destination

        if not dest:
            return user_input

        # If the input already contains any known city name, don't override
        known_cities = list(self.memory.context.__class__.__init__.__defaults__ or [])
        from agent.memory import TravelContext
        # Check against all known destinations in the static list
        known_destinations = [
            "goa", "jaipur", "manali", "delhi", "mumbai", "kerala",
            "udaipur", "shimla", "bangalore", "chennai", "kolkata",
            "agra", "varanasi", "rishikesh", "darjeeling", "ooty",
            "ladakh", "kashmir", "rajasthan", "coorg", "munnar",
            "leh", "kochi", "cochin", "pune", "hyderabad", "amritsar",
            "jodhpur", "mysore", "mysuru", "hampi", "kodaikanal",
        ]
        # If user explicitly named a city, pass through unchanged
        if any(city in u for city in known_destinations):
            return user_input

        # Only inject for truly vague follow-ups with no city mentioned
        weather_followups = [
            "what is the weather there", "what's the weather there",
            "how's the weather", "how is the weather", "weather there",
            "will it rain", "is it hot", "is it cold",
            "what's the weather", "what is the weather",
            "weather forecast", "check weather",
        ]
        if any(phrase in u for phrase in weather_followups):
            return f"What is the weather in {dest}?"

        hotel_followups = [
            "suggest hotels", "find hotels", "what about hotels",
            "where to stay", "hotels there", "accommodation there",
            "suggest accommodation", "find hostels",
        ]
        if any(phrase in u for phrase in hotel_followups):
            return f"Suggest hotels in {dest}"

        places_followups = [
            "what to see there", "things to do there", "places there",
            "what should i see", "sightseeing there", "attractions there",
        ]
        if any(phrase in u for phrase in places_followups):
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

        