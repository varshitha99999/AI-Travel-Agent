# AI Travel Concierge 🌍✈️

An intelligent travel planning assistant for Indian travelers, powered by Groq LLM and built with Chainlit.

## Features

- 📅 **Trip Planning**: Get detailed day-wise itineraries
- 💰 **Budget Calculator**: Calculate per-day travel budgets
- 🌤️ **Weather Info**: Check weather conditions for destinations
- 🏨 **Hotel Suggestions**: Get budget hotel recommendations
- 🎯 **Travel Tips**: Receive practical travel advice

## Tech Stack

- **LLM**: Groq (llama-3.3-70b-versatile)
- **Framework**: LangChain
- **Frontend**: Chainlit
- **Language**: Python 3.13

## Project Structure

```
backend/
├── agent/
│   ├── planner.py      # Main trip planner with LLM integration
│   ├── prompts.py      # System prompts
│   └── budget.py       # Budget calculation tool
├── services/
│   ├── weather.py      # Weather service
│   └── hotels.py       # Hotel search service
├── models/
│   └── request_model.py # Data models (legacy)
├── chainlit_app.py     # Chainlit chat interface
├── streamlit_app.py    # Streamlit interface (alternative)
├── main.py             # FastAPI server (legacy)
└── requirements.txt    # Python dependencies
```

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd tripweaver-ai/backend
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the root directory:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

Get your Groq API key from [https://console.groq.com/](https://console.groq.com/)

### 4. Run the application

```bash
chainlit run chainlit_app.py
```

The app will be available at `http://localhost:8000`

## Usage Examples

Try these prompts:

- "Plan a 3-day trip to Goa with 15000 INR"
- "Calculate budget 20000,4"
- "What's the weather in Manali?"
- "Suggest hotels in Jaipur"
- "Best time to visit Kerala?"

## Tools Available

### Budget Tool
Calculates per-day budget based on total budget and number of days.

**Input format**: `total_budget,days`

### Weather Tool
Provides weather information for destinations (currently mock data).

### Hotel Tool
Suggests budget hotels for destinations (currently mock data).

## Development

### Running with Streamlit (Alternative UI)

```bash
streamlit run streamlit_app.py
```

### Running FastAPI Server (Legacy)

```bash
uvicorn main:app --reload
```

## Future Enhancements

- [ ] Real weather API integration
- [ ] Real hotel booking API integration
- [ ] Multi-language support
- [ ] User authentication
- [ ] Trip history and favorites
- [ ] Export itinerary as PDF
- [ ] Cost optimization suggestions

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
