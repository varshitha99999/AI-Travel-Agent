import re
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import List, Optional


class TravelContext:
    """Stores extracted key travel details and user preferences that persist across the conversation"""

    def __init__(self):
        self.destination: Optional[str] = None
        self.days: Optional[int] = None
        self.budget: Optional[str] = None
        self.interests: List[str] = []
        # User preferences
        self.travel_style: Optional[str] = None       # luxury, budget, adventure, family, honeymoon, solo
        self.accommodation: Optional[str] = None      # hotel, hostel, resort, homestay, camping

    def update_from_text(self, text: str):
        """Extract and update travel context from a message"""
        text_lower = text.lower()

        # Extract destination
        dest_patterns = [
            r"trip to ([a-zA-Z]+)",
            r"travel to ([a-zA-Z]+)",
            r"visit ([a-zA-Z]+)",
            r"going to ([a-zA-Z]+)",
            r"plan.*?([a-zA-Z]+)\s+trip",
        ]
        known_destinations = [
            "goa", "jaipur", "manali", "delhi", "mumbai", "kerala",
            "udaipur", "shimla", "bangalore", "chennai", "kolkata",
            "agra", "varanasi", "rishikesh", "darjeeling", "ooty",
            "ladakh", "kashmir", "rajasthan", "coorg", "munnar",
        ]
        for pattern in dest_patterns:
            match = re.search(pattern, text_lower)
            if match:
                candidate = match.group(1).strip()
                if candidate in known_destinations:
                    self.destination = candidate.title()
                    break
        # Fallback: scan for known destination words
        if not self.destination:
            for word in text_lower.split():
                clean = re.sub(r"[^a-z]", "", word)
                if clean in known_destinations:
                    self.destination = clean.title()
                    break

        # Extract number of days
        days_match = re.search(r"(\d+)\s*(?:-\s*)?day", text_lower)
        if days_match:
            self.days = int(days_match.group(1))

        # Extract budget (INR amounts)
        budget_match = re.search(r"(?:budget|inr|rs\.?|₹)\s*[:\-]?\s*([\d,]+)", text_lower)
        if budget_match:
            self.budget = budget_match.group(1).replace(",", "")
        else:
            # Pattern like "15000 INR" or "₹20000"
            budget_match2 = re.search(r"([\d,]+)\s*(?:inr|rs\.?|₹)", text_lower)
            if budget_match2:
                self.budget = budget_match2.group(1).replace(",", "")

        # Extract interests
        interest_keywords = [
            "beach", "adventure", "trekking", "hiking", "culture", "history",
            "food", "nightlife", "shopping", "wildlife", "nature", "temples",
            "backpacking", "luxury", "budget", "family", "honeymoon", "solo",
        ]
        for keyword in interest_keywords:
            if keyword in text_lower and keyword not in self.interests:
                self.interests.append(keyword)

        # Extract travel style
        style_map = {
            "luxury": ["luxury", "premium", "5 star", "five star", "high end"],
            "budget": ["budget", "cheap", "affordable", "backpacking", "low cost"],
            "adventure": ["adventure", "thrilling", "extreme", "adrenaline"],
            "family": ["family", "kids", "children", "family trip"],
            "honeymoon": ["honeymoon", "romantic", "couple", "anniversary"],
            "solo": ["solo", "alone", "by myself", "single traveler"],
        }
        for style, keywords in style_map.items():
            if any(kw in text_lower for kw in keywords):
                self.travel_style = style
                break

        # Extract accommodation preference
        accommodation_map = {
            "hostel": ["hostel", "dorm", "backpacker"],
            "hotel": ["hotel", "3 star", "4 star", "5 star"],
            "resort": ["resort", "spa", "beachfront"],
            "homestay": ["homestay", "home stay", "airbnb", "local stay"],
            "camping": ["camping", "tent", "camp"],
        }
        for acc_type, keywords in accommodation_map.items():
            if any(kw in text_lower for kw in keywords):
                self.accommodation = acc_type
                break

    def to_context_string(self) -> Optional[str]:
        """Format travel context and preferences as a string for injection into the prompt"""
        if not any([self.destination, self.days, self.budget, self.interests, self.travel_style, self.accommodation]):
            return None

        parts = ["📌 Current Travel Context:"]
        if self.destination:
            parts.append(f"  • Destination: {self.destination}")
        if self.days:
            parts.append(f"  • Duration: {self.days} days")
        if self.budget:
            parts.append(f"  • Budget: ₹{self.budget} INR")
        if self.travel_style:
            parts.append(f"  • Travel Style: {self.travel_style.title()}")
        if self.accommodation:
            parts.append(f"  • Accommodation: {self.accommodation.title()}")
        if self.interests:
            parts.append(f"  • Interests: {', '.join(self.interests)}")
        return "\n".join(parts)

    def is_empty(self) -> bool:
        return not any([self.destination, self.days, self.budget, self.interests, self.travel_style, self.accommodation])

    def clear(self):
        self.destination = None
        self.days = None
        self.budget = None
        self.interests = []
        self.travel_style = None
        self.accommodation = None


class TravelMemory:
    """
    Improved memory for travel conversations.

    - Sliding window of k=5 exchanges for recent messages
    - Persistent TravelContext that survives window trimming
    - Context is injected as a SystemMessage at the start of chat history
    """

    def __init__(self, k: int = 5):
        self.k = k
        self.messages: List = []
        self.context = TravelContext()

    def add_user_message(self, message: str):
        """Add user message and extract travel context from it"""
        self.context.update_from_text(message)
        self.messages.append(HumanMessage(content=message))
        self._trim_memory()

    def add_ai_message(self, message: str):
        """Add AI response and extract any travel context from it"""
        self.context.update_from_text(message)
        self.messages.append(AIMessage(content=message))
        self._trim_memory()

    def get_chat_history(self) -> List:
        """
        Return chat history with persistent travel context prepended
        as a SystemMessage so the LLM always has key details.
        """
        context_str = self.context.to_context_string()
        if context_str:
            return [SystemMessage(content=context_str)] + self.messages
        return self.messages

    def _trim_memory(self):
        """Keep only the last k exchanges (k*2 messages)"""
        max_messages = self.k * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def clear_memory(self):
        """Clear all conversation memory and travel context"""
        self.messages = []
        self.context.clear()

    def has_context(self) -> bool:
        return len(self.messages) > 0

    def get_memory_summary(self) -> str:
        """Human-readable summary of current memory state"""
        parts = []
        ctx = self.context.to_context_string()
        if ctx:
            parts.append(ctx)
        parts.append(f"\nRecent messages: {len(self.messages)} stored (window: {self.k * 2})")
        return "\n".join(parts) if parts else "No conversation history."
