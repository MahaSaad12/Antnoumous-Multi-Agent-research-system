"""
graph.py
Builds and compiles the LangGraph multi-agent research pipeline.

Flow:
  START → orchestrator → researcher → analyst → writer → END
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from config.state import ResearchState
from agents.orchestrator import orchestrator_node
from agents.researcher import researcher_node
from agents.analyst import analyst_node
from agents.writer import writer_node


def build_graph() -> StateGraph:
    g = StateGraph(ResearchState)

    # Register nodes
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("researcher", researcher_node)
    g.add_node("analyst", analyst_node)
    g.add_node("writer", writer_node)

    # Linear pipeline edges
    g.add_edge(START, "orchestrator")
    g.add_edge("orchestrator", "researcher")
    g.add_edge("researcher", "analyst")
    g.add_edge("analyst", "writer")
    g.add_edge("writer", END)

    return g.compile()


# Singleton — import this in main.py
research_graph = build_graph()
