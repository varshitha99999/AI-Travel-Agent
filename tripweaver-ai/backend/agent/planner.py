import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from agent.budget import calculate_budget
from agent.memory import TravelMemory
from services.weather import get_weather
from services.hotels import search_hotels

load_dotenv()


class TripPlanner:
    def __init__(self):
        # Initialize LangChain ChatGroq
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=1000,
            timeout=30,
        )
        
        # Initialize memory
        self.memory = TravelMemory()
        
        # LangChain output parser
        self.output_parser = StrOutputParser()
        
        # Create LangChain prompt template with memory
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an AI Travel Concierge for Indian travelers.

You help with:
- Trip planning and detailed itineraries
- Travel advice and practical tips
- Destination recommendations
- Budget planning guidance
- Transportation suggestions
- Local culture and food recommendations

Guidelines:
- Be helpful, practical, and enthusiastic about travel
- Assume all costs are in INR unless specified otherwise
- Provide specific, actionable advice
- Include practical tips for Indian travelers
- Suggest realistic budgets and timeframes
- Consider seasonal factors and local events
- Use conversation history to provide contextual responses
- If asked about previous plans, refer to the conversation history

IMPORTANT: When providing trip plans, always include:
- Day-wise itinerary
- Estimated costs
- Travel tips
- Accommodation suggestions"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Create LangChain chain
        self.chain = self.prompt_template | self.llm | self.output_parser

    def chat(self, user_input: str) -> str:
        try:
            # Check if user wants specific tools first
            tool_response = self._check_tools(user_input)
            if tool_response:
                # Add to memory
                self.memory.add_user_message(user_input)
                self.memory.add_ai_message(tool_response)
                return tool_response
            
            # Get chat history from memory
            chat_history = self.memory.get_chat_history()
            
            # Use LangChain chain for general travel planning with memory
            response = self.chain.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            # Add conversation to memory
            self.memory.add_user_message(user_input)
            self.memory.add_ai_message(response)
            
            return response
            
        except Exception as e:
            error_msg = f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."
            return error_msg

    def _check_tools(self, user_input: str) -> str:
        """Check if user input requires specific tools"""
        user_lower = user_input.lower()
        
        # Budget calculation tool
        if "budget" in user_lower and "," in user_input:
            parts = [p.strip() for p in user_input.split() if "," in p]
            if parts:
                budget_result = calculate_budget(parts[0])
                return budget_result
        
        # Check for budget questions related to previous conversations
        if any(word in user_lower for word in ["budget", "cost", "expense", "price"]) and not "," in user_input:
            # Let the LLM handle budget questions with conversation context
            return None
        
        # Weather tool
        if any(word in user_lower for word in ["weather", "climate", "temperature"]):
            destination = self._extract_destination(user_input)
            if destination:
                weather_result = get_weather(destination)
                return weather_result
        
        # Hotel tool
        if any(word in user_lower for word in ["hotel", "stay", "accommodation", "lodge", "resort"]):
            destination = self._extract_destination(user_input)
            if destination:
                hotel_result = search_hotels(destination)
                return hotel_result
        
        return None

    def _extract_destination(self, user_input: str) -> str:
        """Extract destination from user input"""
        words = user_input.split()
        
        # Look for common prepositions followed by destination
        prepositions = ["in", "for", "at", "to", "near", "around"]
        
        for i, word in enumerate(words):
            if word.lower() in prepositions and i + 1 < len(words):
                # Get the next word and clean it
                destination = words[i + 1].replace("?", "").replace(".", "").replace(",", "")
                return destination
        
        # If no preposition found, look for known destinations
        known_destinations = [
            "goa", "jaipur", "manali", "delhi", "mumbai", "kerala", 
            "udaipur", "shimla", "bangalore", "chennai", "kolkata",
            "agra", "varanasi", "rishikesh", "darjeeling", "ooty"
        ]
        
        for word in words:
            clean_word = word.lower().replace("?", "").replace(".", "").replace(",", "")
            if clean_word in known_destinations:
                return clean_word
        
        return None

    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear_memory()

    def get_conversation_summary(self):
        """Get a summary of the current conversation"""
        chat_history = self.memory.get_chat_history()
        if not chat_history:
            return "No conversation history available."
        
        summary = "Recent conversation:\n"
        for i, message in enumerate(chat_history[-4:]):  # Last 2 exchanges
            if isinstance(message, HumanMessage):
                summary += f"User: {message.content[:100]}...\n"
            elif isinstance(message, AIMessage):
                summary += f"AI: {message.content[:100]}...\n"
        
        return summary