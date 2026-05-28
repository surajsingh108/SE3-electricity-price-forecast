# SE3 Energy Analyst Agent

A live LangGraph agent that answers natural language questions about 
the SE3 Swedish electricity market (Stockholm/Mälardalen price zone).

**Live demo:** https://tinyurl.com/se3-forecast

## What it does

Ask it anything about SE3 electricity prices in plain English. It fetches 
live data from three sources, reasons over them, and returns a structured 
answer with confidence score and cited sources.

Example questions:
- "Why is SE3 electricity expensive right now?"
- "What will prices look like tomorrow?"
- "Should I run my appliances now or wait?"
- "Why is the price high?"

## Architecture

```
User question (Streamlit UI)
        │
        ▼
  LangGraph Agent
        │
  ┌─────┼──────┐
  ▼     ▼      ▼
Price  Forecast Weather
Tool   Tool     Tool
  │     │       │
  └─────┴───────┘
        │
  Synthesis Node (Gemini 3.5 Flash)
        │
        ▼
  Structured answer + sources + confidence
```

## Tech stack

| Component | Choice |
|---|---|
| Agent orchestration | LangGraph |
| LLM | Gemini 3.5 Flash (Google AI Studio) |
| Price data | ENTSO-E Transparency API |
| Forecast | SE3 LightGBM model (quantile regression) |
| Weather | Open-Meteo API |
| Data layer | DuckDB (GCS-mounted) |
| API | FastAPI |
| Frontend | Streamlit |
| Deployment | GCP Cloud Run |

## Repo structure

```
se3-agent/
├── agent/
│   ├── graph.py          # LangGraph StateGraph
│   ├── state.py          # AgentState dataclass
│   ├── nodes.py          # router, synthesiser nodes
│   └── tools/
│       ├── price.py      # ENTSO-E price fetcher (DuckDB)
│       ├── forecast.py   # SE3 model forecast (DuckDB)
│       └── weather.py    # Open-Meteo weather (DuckDB)
├── api.py                # FastAPI — includes /ask endpoint
├── dashboard.py          # Streamlit frontend
├── pipeline.py           # Data sync pipeline
├── ml.py                 # LightGBM forecasting model
├── supervisord.conf      # Runs api + dashboard in one container
├── Dockerfile
└── requirements.txt
```

## Running locally

```bash
# Set env vars
cp .env.example .env
# Add GEMINI_API_KEY and ENTSO_E_TOKEN

# Install dependencies
pip install -r requirements.txt

# Start both services
supervisord -c supervisord.conf
```

## Deployment

Deployed on GCP Cloud Run (europe-north1). Single container runs 
FastAPI on port 8000 (internal) and Streamlit on port 8080 (public).

Built by [Suraj Singh](https://github.com/surajsingh108)
