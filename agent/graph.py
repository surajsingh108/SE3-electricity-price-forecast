from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import (
    router_node, price_node, forecast_node,
    weather_node, synthesiser_node
)

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("router", router_node)
    g.add_node("price", price_node)
    g.add_node("forecast", forecast_node)
    g.add_node("weather", weather_node)
    g.add_node("synthesiser", synthesiser_node)

    g.set_entry_point("router")

    g.add_edge("router", "price")
    g.add_edge("price", "forecast")
    g.add_edge("forecast", "weather")
    g.add_edge("weather", "synthesiser")
    g.add_edge("synthesiser", END)

    return g.compile()

agent_graph = build_graph()
