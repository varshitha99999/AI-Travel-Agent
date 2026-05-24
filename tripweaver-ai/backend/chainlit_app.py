import asyncio
import os
import warnings

# Suppress TensorFlow/oneDNN noise before any heavy imports
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

from pathlib import Path

import chainlit as cl

from agent.planner import TripPlanner
from rag.document_store import TravelDocumentStore
from rag.rag_chain import answer_from_docs


QUICK_ACTIONS = [
    cl.Action(name="weather", label="🌤 Check Weather", payload={"query": "What's the weather in Goa?"}),
    cl.Action(name="hotels",  label="🏨 Find Hotels",   payload={"query": "Suggest hotels in Jaipur"}),
    cl.Action(name="budget",  label="💰 Budget Help",   payload={"query": "My budget is 15000 INR 3 days. What's my daily budget?"}),
    cl.Action(name="plan",    label="📅 Plan a Trip",   payload={"query": "Plan a 3-day budget trip to Manali"}),
]


# ── helpers ──────────────────────────────────────────────────────────────────

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


def _context_bar(ctx) -> str:
    parts = []
    if ctx.destination:  parts.append(f"📍 {ctx.destination}")
    if ctx.days:         parts.append(f"📅 {ctx.days}d")
    if ctx.budget:       parts.append(f"💰 ₹{ctx.budget}")
    if ctx.travel_style: parts.append(f"🎯 {ctx.travel_style.title()}")
    if ctx.accommodation:parts.append(f"🏨 {ctx.accommodation.title()}")
    return "  ·  ".join(parts) if parts else "No context yet"


RESPONSE_HEADERS = {
    "weather":   "## 🌤 Weather Report\n\n",
    "hotel":     "## 🏨 Accommodation Options\n\n",
    "budget":    "## 💰 Budget Breakdown\n\n",
    "itinerary": "## 🗺️ Your Travel Plan\n\n",
    "general":   "",
}
FOOTER = "\n\n---\n💡 *Ask me anything about your trip!*"


async def _stream_response(text: str) -> None:
    """Stream a string word-by-word into a Chainlit message."""
    msg = cl.Message(content="")
    await msg.send()
    words = text.split(" ")
    for i, word in enumerate(words):
        await msg.stream_token(word + (" " if i < len(words) - 1 else ""))
    await msg.update()


# ── lifecycle ─────────────────────────────────────────────────────────────────

@cl.on_chat_start
async def start():
    import uuid
    session_id = str(uuid.uuid4())[:8]
    cl.user_session.set("planner", TripPlanner())
    cl.user_session.set("doc_store", TravelDocumentStore())
    cl.user_session.set("session_id", session_id)

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
            "| 🔍 | Web Search | *What festivals are in Rajasthan in October?* |\n"
            "| 🗺️ | Top Attractions | *What are the top places to visit in Delhi?* |\n"
            "| 📄 | Doc Q&A | *Upload a PDF travel guide and ask questions* |\n\n"
            "**📎 Upload a travel document** (PDF or TXT) using the attachment icon "
            "to ask questions about it.\n\n"
            "**Or pick a quick action below 👇**"
        ),
        actions=QUICK_ACTIONS,
    ).send()


@cl.on_chat_end
async def end():
    planner = cl.user_session.get("planner")
    if planner:
        planner.clear_memory()


# ── file upload ───────────────────────────────────────────────────────────────

@cl.on_message
async def main(message: cl.Message):
    # Handle uploaded files first
    if message.elements:
        await _handle_uploads(message.elements)
        # If the message also has text, answer it using RAG
        if message.content.strip():
            await _handle_query(message.content)
        return

    await _handle_query(message.content)


async def _handle_uploads(elements) -> None:
    """Process uploaded PDF / TXT files into the RAG document store."""
    doc_store: TravelDocumentStore = cl.user_session.get("doc_store")
    processed = []

    for element in elements:
        # Only handle file elements
        if not hasattr(element, "path") or not element.path:
            continue

        file_name = getattr(element, "name", Path(element.path).name)
        ext = Path(file_name).suffix.lower()

        if ext not in (".pdf", ".txt", ".md"):
            await cl.Message(
                content=f"⚠️ **{file_name}** — unsupported format. Please upload PDF or TXT files."
            ).send()
            continue

        async with cl.Step(name=f"📄 Processing {file_name}…", type="tool") as step:
            step.input = file_name
            chunks = await asyncio.get_event_loop().run_in_executor(
                None, doc_store.add_file, element.path, file_name
            )
            step.output = f"{chunks} chunks indexed"

        processed.append((file_name, chunks))

    if processed:
        summary = "\n".join(
            f"  ✅ **{name}** — {chunks} chunks indexed" for name, chunks in processed
        )
        await cl.Message(
            content=(
                f"📚 **Documents uploaded successfully!**\n\n{summary}\n\n"
                "You can now ask me questions about these documents."
            )
        ).send()


# ── quick actions ─────────────────────────────────────────────────────────────

@cl.action_callback("weather")
@cl.action_callback("hotels")
@cl.action_callback("budget")
@cl.action_callback("plan")
async def on_quick_action(action: cl.Action):
    query = action.payload.get("query", "")
    await cl.Message(content=f"_{query}_").send()
    await _handle_query(query)


# ── core query handler ────────────────────────────────────────────────────────

async def _handle_query(user_input: str) -> None:
    """
    Route the query:
    1. If documents are loaded and the question looks doc-related → RAG answer
    2. Otherwise → agent (tools + LLM)
    """
    planner: TripPlanner = cl.user_session.get("planner")
    doc_store: TravelDocumentStore = cl.user_session.get("doc_store")

    # ── RAG path ──────────────────────────────────────────────────────────────
    if doc_store.has_documents() and _is_doc_question(user_input, doc_store):
        async with cl.Step(name="📄 Searching documents…", type="tool") as step:
            step.input = user_input
            rag_response = await asyncio.get_event_loop().run_in_executor(
                None, answer_from_docs, user_input, doc_store, None
            )
            step.output = f"Retrieved from: {', '.join(doc_store.document_names)}"

        if rag_response:
            full = "## 📄 From Your Documents\n\n" + rag_response + FOOTER
            await _stream_response(full)
            return
        # Fall through to agent if RAG returned nothing useful

    # ── Agent path ────────────────────────────────────────────────────────────
    async with cl.Step(name="🤖 Thinking…", type="tool") as step:
        step.input = user_input
        session_id = cl.user_session.get("session_id", "default")
        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: planner.chat(user_input, session_id=session_id)
        )
        step.output = _context_bar(planner.memory.context)

    response_type = _detect_type(user_input)
    full = RESPONSE_HEADERS.get(response_type, "") + response + FOOTER
    await _stream_response(full)


def _is_doc_question(user_input: str, doc_store: TravelDocumentStore) -> bool:
    """
    Heuristic: route to RAG if the question seems to reference uploaded content.
    Falls back to agent for standard travel queries (weather, hotels, budget, plan).
    """
    if not doc_store.has_documents():
        return False

    u = user_input.lower()

    # Explicit doc references
    doc_keywords = [
        "document", "pdf", "file", "uploaded", "guide", "brochure",
        "according to", "based on", "from the", "in the document",
        "what does it say", "what does the", "tell me from",
        "visa", "itinerary", "checklist", "packing list",
    ]
    if any(kw in u for kw in doc_keywords):
        return True

    # Standard tool-based queries → keep with agent
    agent_keywords = [
        "weather", "hotel", "budget", "plan a trip", "calculate",
        "forecast", "temperature", "accommodation",
    ]
    if any(kw in u for kw in agent_keywords):
        return False

    # Default: if docs are loaded, try RAG first for general questions
    return True


if __name__ == "__main__":
    cl.run()
