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
   - "what is the weather there" → use destination from context, call WeatherTool
   - NEVER respond to a weather question with an itinerary. ALWAYS call WeatherTool.

2. HotelTool — call this when the user asks about hotels, accommodation, where to stay, or hostels.
   - "suggest hotels in Jaipur" → call HotelTool("Jaipur")
   - "what about hotels?" → use destination from context, call HotelTool
   - NEVER make up hotel names. ALWAYS call HotelTool.

3. BudgetTool — call this ONLY when the user provides BOTH a total amount AND number of days.
   - "my budget is ₹15000 for 3 days" → call BudgetTool("15000,3")
   - "what will the budget be?" → DO NOT call BudgetTool. Instead summarise costs from the itinerary already given.
   - NEVER invent a new budget. If no new budget is stated, refer to the existing itinerary costs.

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
            model_name="llama-3.3-70b-versatile",
            temperature=0.7,
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
            max_iterations=3,
        )

    def chat(self, user_input: str) -> str:
        try:
            chat_history = self.memory.get_chat_history()

            result = self.agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history,
            })

            response = result.get("output", "")
            if not response:
                raise ValueError("Empty response from agent")

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
                    "input": user_input,
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