# TripWeaver AI 🌍✈️

> An intelligent travel concierge for Indian travelers — powered by Groq LLM, LangChain, and real-world APIs.

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green)](https://langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📸 Features

| | Feature | Details |
|---|---|---|
| 🌤️ | **Live Weather** | Real-time conditions + 7-day forecast (Weatherstack / Open-Meteo fallback) |
| 🏨 | **Hotel Finder** | Hotels, hostels, guest houses (Amadeus / OpenStreetMap fallback) |
| ✈️ | **Flight Search** | Live fares between Indian cities (Amadeus / static fallback) |
| 💰 | **Budget Planner** | Daily breakdown by accommodation, food, transport, activities |
| 🗺️ | **Trip Itinerary** | Day-wise plans with cost estimates tailored to travel style |
| 🔍 | **Web Search** | Live travel news, festivals, visa info (DuckDuckGo) |
| 📍 | **Attractions** | Tourist spots & POIs (OpenTripMap / curated static data) |
| 📄 | **Doc Q&A** | Upload PDF/TXT travel guides and ask questions (TF-IDF RAG) |
| 💾 | **Save & Export** | Save itineraries to SQLite DB, export chat as TXT or JSON |

---

## 🏗️ Architecture

```
tripweaver-ai/backend/
├── agent/
│   ├── planner.py          # LangChain tool-calling agent + conversation memory
│   ├── memory.py           # Sliding-window memory + travel context extraction
│   ├── tools.py            # 8 LangChain tools
│   ├── budget.py           # Budget breakdown calculator
│   └── error_handler.py    # Retry decorator, safe_tool_call, input validators
├── services/
│   ├── weather.py          # Weatherstack + Open-Meteo fallback
│   ├── hotels.py           # Amadeus + OpenStreetMap fallback
│   ├── flights.py          # Amadeus Flights + static schedule fallback
│   ├── places.py           # OpenTripMap + curated static data
│   └── web_search.py       # DuckDuckGo search
├── database/
│   └── db.py               # SQLite — searches, itineraries, user_preferences
├── rag/
│   ├── document_store.py   # TF-IDF document retrieval (no ML deps)
│   └── rag_chain.py        # RAG Q&A chain
├── tests/
│   ├── test_agent.py       # Memory, budget, weather, hotel tests
│   ├── test_tools.py       # All 8 tools + error handler (46 tests)
│   └── test_week56.py      # DB, flights, save/history tools (48 tests)
├── streamlit_app.py        # Production Streamlit UI
├── chainlit_app.py         # Chainlit chat UI (local dev)
├── main.py                 # FastAPI server
├── requirements.txt
├── runtime.txt
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone <your-repo-url>
cd tripweaver-ai/backend
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional — all have free fallbacks if not set
WEATHER_API_KEY=your_weatherstack_key_here
AMADEUS_API_KEY=your_amadeus_key_here
AMADEUS_API_SECRET=your_amadeus_secret_here
OPENTRIPMAP_API_KEY=your_opentripmap_key_here
```

### 3. Run

**Streamlit (recommended):**
```bash
streamlit run streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501)

**Chainlit (local dev):**
```bash
chainlit run chainlit_app.py
```
Open [http://localhost:8000](http://localhost:8000)

---

## 🔑 API Keys

| Key | Where to get | Free tier | Required? |
|---|---|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | Yes (generous) | ✅ Yes |
| `WEATHER_API_KEY` | [weatherstack.com](https://weatherstack.com) | 250 calls/month | ❌ No — falls back to Open-Meteo |
| `AMADEUS_API_KEY` + `SECRET` | [developers.amadeus.com](https://developers.amadeus.com) | 2,000 calls/month | ❌ No — falls back to OpenStreetMap / static |
| `OPENTRIPMAP_API_KEY` | [opentripmap.com](https://opentripmap.com/product) | No hard cap | ❌ No — falls back to curated static data |

> **All APIs have free fallbacks.** The app works fully without any optional keys.

---

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# Individual suites
pytest tests/test_agent.py -v      # memory, budget, weather, hotels
pytest tests/test_tools.py -v      # all 8 tools + error handler
pytest tests/test_week56.py -v     # database, flights, save/history
```

**Test coverage: 94 tests, all offline (mocked APIs)**

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Set:
   - **Repository**: your repo
   - **Branch**: `main`
   - **Main file path**: `backend/streamlit_app.py`
4. Under **Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
5. Click **Deploy**

---

## 💬 Example Prompts

```
Plan a 3-day trip to Goa with ₹15,000 budget
What's the weather in Manali right now?
Suggest hotels in Jaipur under ₹2000/night
Flights from Delhi to Goa this weekend
Top places to visit in Varanasi
My budget is 25000 INR for 5 days — break it down
What festivals are happening in Rajasthan in October?
Is it safe to travel to Ladakh in January?
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq (`llama-3.1-8b-instant`) |
| Agent framework | LangChain tool-calling agent |
| Chat UI | Streamlit / Chainlit |
| Database | SQLite (built-in Python) |
| RAG | Pure-Python TF-IDF (no ML deps) |
| Weather | Weatherstack → Open-Meteo |
| Hotels & Flights | Amadeus → OpenStreetMap |
| Attractions | OpenTripMap → static data |
| Web search | DuckDuckGo (free) |

---

## 📁 Project Structure Notes

- **No TensorFlow or PyTorch** — the RAG uses pure-Python TF-IDF, keeping startup fast and dependencies minimal
- **Graceful degradation** — every paid API has a free fallback; the app never crashes due to quota limits
- **Thread-safe SQLite** — uses WAL mode + module-level lock for concurrent access
- **Secure by default** — all API keys via environment variables, `.env` in `.gitignore`

---

## 👥 Team

Built as part of an 8-week AI Agent Development project.

---

## 📄 License

MIT
