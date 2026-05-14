"""
main.py
Entry point for the Multi-Agent Research System.

Usage:
    python main.py                            # interactive prompt
    python main.py "Your research query"      # direct query
    python main.py --seed                     # seed demo data then query
    python main.py --clear                    # clear RAG memory
    python main.py --stats                    # show memory stats
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule

load_dotenv()
console = Console()

# ── Seed demo data ────────────────────────────────────────────────────────────

DEMO_DOCS = [
    ("LangGraph is a library for building stateful, multi-actor applications with LLMs. "
     "It extends LangChain with cyclic graph support, enabling complex agent workflows "
     "with memory, branching, and human-in-the-loop capabilities.", "langgraph_docs"),

    ("CrewAI is a framework for orchestrating role-playing AI agents. Agents are assigned "
     "specific roles (researcher, writer, analyst) and collaborate on tasks using tools "
     "like web search, file I/O, and code execution.", "crewai_docs"),

    ("AutoGen 0.4 by Microsoft introduces an async, event-driven multi-agent framework. "
     "It supports decentralized agent communication, nested chats, and pluggable LLM backends. "
     "Ideal for long-running, parallel research pipelines.", "autogen_docs"),

    ("RAG (Retrieval-Augmented Generation) combines a vector database with an LLM to ground "
     "responses in retrieved documents. ChromaDB is a popular open-source vector store "
     "supporting cosine similarity search with persistent local storage.", "rag_overview"),

    ("OpenAI's Agents SDK (2025) provides native support for agent handoffs, guardrails, "
     "and tracing. It simplifies building multi-agent systems for OpenAI model users "
     "with built-in tool calling and structured outputs.", "openai_agents_sdk"),

    ("Tool calling (function calling) allows LLMs to invoke external tools — web search, "
     "calculators, databases — during inference. LangChain abstracts this with a unified "
     "Tool interface compatible with all major LLM providers.", "langchain_tools"),

    ("Vector embeddings convert text into dense numerical representations. Sentence-Transformers "
     "provides lightweight models (e.g. all-MiniLM-L6-v2) that run locally without API costs, "
     "making them ideal for RAG prototypes.", "embeddings_guide"),

    ("Agent memory architectures include: in-context (messages), external (vector DB / SQL), "
     "episodic (past runs), and semantic (knowledge base). Combining these enables agents "
     "that learn across sessions.", "agent_memory_patterns"),
]


def seed_memory() -> None:
    from rag.memory import ingest_texts
    console.print("[bold cyan]Seeding RAG memory with demo documents…[/bold cyan]")
    total = 0
    for text, source in DEMO_DOCS:
        n = ingest_texts([text], source=source)
        total += n
        console.print(f"  ✓ [{source}] → {n} chunk(s)")
    console.print(f"\n[green]Seeded {total} chunks from {len(DEMO_DOCS)} documents.[/green]\n")


# ── Run pipeline ──────────────────────────────────────────────────────────────

def run_research(query: str) -> str:
    from graph import research_graph

    console.print(Rule("[bold]Multi-Agent Research System[/bold]"))
    console.print(f"\n[bold yellow]Query:[/bold yellow] {query}\n")

    initial_state = {
        "query": query,
        "plan": [],
        "context": [],
        "findings": "",
        "report": "",
        "messages": [],
        "meta": {},
    }

    final_state = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        steps = [
            ("orchestrator", "🧭  Orchestrator — planning query…"),
            ("researcher",   "🔍  Researcher  — retrieving from RAG…"),
            ("analyst",      "🧠  Analyst     — synthesizing findings…"),
            ("writer",       "✍️   Writer      — drafting report…"),
        ]
        task_ids = {name: progress.add_task(desc, total=None) for name, desc in steps}

        current_node = None
        for event in research_graph.stream(initial_state, stream_mode="updates"):
            for node_name, state_update in event.items():
                # Mark previous done
                if current_node and current_node in task_ids:
                    progress.update(task_ids[current_node], description=f"✅  {current_node.capitalize()} — done")
                    progress.stop_task(task_ids[current_node])
                current_node = node_name
                if node_name in task_ids:
                    progress.update(task_ids[node_name], description=steps[[n for n,_ in steps].index(node_name)][1])
                final_state = state_update

        if current_node and current_node in task_ids:
            progress.update(task_ids[current_node], description=f"✅  {current_node.capitalize()} — done")

    # Fetch full state from last stream
    # Re-run to get final merged state (stream gives partial updates)
    full_state = research_graph.invoke(initial_state)

    # Print agent messages
    console.print(Rule("Agent Log"))
    for msg in full_state.get("messages", []):
        name = getattr(msg, "name", "agent")
        color = {"orchestrator": "cyan", "researcher": "green",
                 "analyst": "magenta", "writer": "yellow"}.get(name, "white")
        console.print(Panel(msg.content, title=f"[{color}]{name}[/{color}]",
                            border_style=color, expand=False))

    report = full_state.get("report", "No report generated.")

    # Print report
    console.print(Rule("Final Report"))
    console.print(Markdown(report))

    # Save to file
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = query[:40].replace(" ", "_").replace("/", "-")
    out_path = out_dir / f"report_{ts}_{safe_query}.md"
    out_path.write_text(report, encoding="utf-8")
    console.print(f"\n[dim]Report saved → {out_path}[/dim]")

    return report


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]

    if "--clear" in args:
        from rag.memory import clear_memory
        clear_memory()
        return

    if "--stats" in args:
        from rag.memory import memory_stats
        stats = memory_stats()
        console.print(stats)
        return

    if "--seed" in args:
        seed_memory()
        args = [a for a in args if a != "--seed"]

    query = " ".join(args).strip() if args else ""
    if not query:
        console.print("[bold]Enter your research query[/bold] (or Ctrl+C to exit):")
        query = input("> ").strip()

    if not query:
        console.print("[red]No query provided.[/red]")
        sys.exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]Error:[/red] OPENAI_API_KEY not set. Copy .env.example → .env and add your key.")
        sys.exit(1)

    run_research(query)


if __name__ == "__main__":
    main()
