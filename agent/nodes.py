import os, json
from google import genai
from google.genai import types
from agent.state import AgentState
from agent.tools.price import get_price_data
from agent.tools.forecast import get_forecast_data
from agent.tools.weather import get_weather_data

SYNTH_PROMPT = """You are a sharp, data-driven analyst of the SE3 Swedish electricity market (Stockholm/Mälardalen price zone).

You always receive four data sources:
- Current market prices with anomaly detection + full 72h hourly series (ENTSO-E)
- 12h price forecast + 48h forecast series + last 24h forecast accuracy (SE3 LightGBM model)
- Current weather conditions (Open-Meteo)
- SMHI wind forecast for SE3 locations (SNOW1gv1 API) — forward-looking wind speed forecasts up to 10 days ahead

STRICT RULES:

1. POINT LOOKUPS: If the user asks about a specific time or date, search
   hourly_series_72h for that exact timestamp and return the actual price.
   If the timestamp is outside the 72h window, say so and give the closest
   available data point. Never guess – only report what the data contains.

2. PREMISE CHECKING: If the user states a specific price or number, verify
   it against hourly_series_72h and peak_price_48h before explaining anything.
   If it conflicts, correct them in the first sentence:
   "You mentioned X EUR/MWh – the data shows the 48h peak was Y EUR/MWh at [time]."

3. DIRECT ANSWER: Answer the question using specific numbers from the data.
   Connect price, forecast, and weather into one coherent explanation.

4. FORECAST QUALITY: If forecast_accuracy_last_24h shows accuracy_rating is
   poor or avg_abs_error > 25, acknowledge this and lower confidence accordingly.
   If the user asks why a forecast missed badly, use the weather data to explain
   what the model likely failed to capture.

5. PROACTIVE INSIGHT: Always end with ONE observation the user did not ask for:
   - Flag anomaly_status if high or low (cite z_score)
   - Warn if the next 12h forecast shows a significant price move
   - Note if recent forecast accuracy is poor and confidence is reduced
   - Highlight peak_price_48h if it is unusually high

6. Keep the total answer under 180 words.

Question: {question}

Data:
{tool_results}

Respond with this exact JSON format and nothing else:
{{
  "answer": "your answer here",
  "confidence": 0.85,
  "sources": ["ENTSO-E", "SE3 LightGBM model", "Open-Meteo", "SMHI"]
}}"""


def router_node(state: AgentState) -> AgentState:
    state["tools_called"] = ["price", "forecast", "weather"]
    state["tools_failed"] = []
    state["tool_results"] = {}
    return state


def price_node(state: AgentState) -> AgentState:
    if "price" not in state.get("tools_called", []):
        return state
    result = get_price_data()
    state["tool_results"]["price"] = result
    if "error" in result:
        state["tools_failed"].append("price")
    return state


def forecast_node(state: AgentState) -> AgentState:
    if "forecast" not in state.get("tools_called", []):
        return state
    result = get_forecast_data()
    state["tool_results"]["forecast"] = result
    if "error" in result:
        state["tools_failed"].append("forecast")
    return state


def weather_node(state: AgentState) -> AgentState:
    if "weather" not in state.get("tools_called", []):
        return state
    result = get_weather_data()
    state["tool_results"]["weather"] = result
    if "error" in result:
        state["tools_failed"].append("weather")
    return state


def synthesiser_node(state: AgentState) -> AgentState:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
    prompt = SYNTH_PROMPT.format(
        question=state["question"],
        tool_results=json.dumps(state["tool_results"], indent=2)
    )
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        parsed = json.loads(text.strip())
        state["answer"] = parsed.get("answer", text)
        state["confidence"] = parsed.get("confidence", 0.7)
        state["sources"] = parsed.get("sources", [])
    except Exception:
        state["answer"] = text
        state["confidence"] = 0.5
        state["sources"] = list(state["tool_results"].keys())
    return state
