from langchain_core.messages import HumanMessage, AIMessage
from typing import List


class TravelMemory:
    """Simple memory management for travel conversations using LangChain message format"""
    
    def __init__(self, k: int = 5):
        self.k = k  # Keep last k exchanges (k*2 messages)
        self.messages: List = []
    
    def add_user_message(self, message: str):
        """Add user message to memory"""
        self.messages.append(HumanMessage(content=message))
        self._trim_memory()
    
    def add_ai_message(self, message: str):
        """Add AI response to memory"""
        self.messages.append(AIMessage(content=message))
        self._trim_memory()
    
    def get_chat_history(self):
        """Get chat history as LangChain messages"""
        return self.messages
    
    def _trim_memory(self):
        """Keep only the last k exchanges (k*2 messages)"""
        max_messages = self.k * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.messages = []
    
    def get_memory_summary(self):
        """Get a text summary of the conversation"""
        if not self.messages:
            return "No conversation history."
        
        summary = []
        for msg in self.messages[-6:]:  # Last 3 exchanges
            if isinstance(msg, HumanMessage):
                summary.append(f"User: {msg.content[:100]}...")
            elif isinstance(msg, AIMessage):
                summary.append(f"AI: {msg.content[:100]}...")
        
        return "\n".join(summary)
    
    def has_context(self) -> bool:
        """Check if there's conversation context"""
        return len(self.messages) > 0