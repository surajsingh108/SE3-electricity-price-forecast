import os, json
from google import genai
from google.genai import types
from agent.state import AgentState
from agent.tools.price import get_price_data
from agent.tools.forecast import get_forecast_data
from agent.tools.weather import get_weather_data

SYNTH_PROMPT = """You are an expert analyst of the SE3 Swedish electricity market (Stockholm/Mälardalen price zone).

You always receive three data sources: current market prices (ENTSO-E),
a 24h price forecast (SE3 LightGBM model), and current weather conditions
(Open-Meteo). Use all of them to give a complete answer.

Rules:
- Answer the question directly in the first sentence
- Connect price, forecast and weather data into a coherent explanation
- Be specific with numbers
- Keep the answer under 150 words
- If a data source failed, acknowledge it briefly and continue with what you have

Question: {question}

Data:
{tool_results}

Respond with this exact JSON format and nothing else:
{{
  "answer": "your answer here",
  "confidence": 0.85,
  "sources": ["ENTSO-E", "SE3 LightGBM model", "Open-Meteo"]
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
