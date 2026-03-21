# Agent Internals — How `create_agent` Works and How It Compares to LangGraph

## What `create_agent` Actually Is

`create_agent` is not a separate agent engine. It is a **thin wrapper over LangGraph** — confirmed by reading its source:

```python
# langchain/agents/factory.py
from langgraph.graph.state import StateGraph
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.constants import END, START
```

It builds a LangGraph `StateGraph` internally, compiles it, and returns the compiled graph. When you call `agent.invoke(...)`, you are running a LangGraph graph — you just didn't define it explicitly.

---

## The Fixed Loop Inside `create_agent`

The graph it builds has exactly two nodes and a conditional edge:

```
START
  │
  ▼
[llm_node] ── has tool calls? ──Yes──► [tool_node] ──► back to [llm_node]
                    │
                   No
                    │
                    ▼
                   END
```

In code terms, every iteration:

```python
while True:
    response = llm(messages, tools=tools)       # call the LLM
    messages.append(response)

    if response has no tool calls:
        break                                    # LLM decided it's done

    for tool_call in response.tool_calls:
        result = run_tool(tool_call)             # execute each tool
        messages.append(ToolMessage(result))     # append result to history
```

The loop structure is **hardcoded**. What varies each iteration is the LLM's decision: which tool to call, with what arguments, and when to stop.

---

## How Behaviour Is Controlled in `create_agent`

There is no code-level control over agent behaviour. Everything is driven by:

1. **The system prompt** — tells the LLM what role it plays, when to search, how many times, what format to answer in
2. **The tools available** — the LLM can only call tools it was given
3. **The LLM itself** — its trained behaviour shapes how it reasons

```python
agent = create_agent(
    model=llm,
    tools=[search_tool],
    system_prompt="""You are an HR assistant. Always search before answering.
    If the first search is insufficient, search again with different terms."""
)
```

The instruction "search at least once" and "search again if needed" are enforced only by the LLM's willingness to follow them — not by code. A poorly prompted agent or a weaker LLM may not comply reliably.

---

## The State Object

Every LangGraph graph (including `create_agent`'s internal one) passes a **state dict** between nodes. For `create_agent` the state is simply:

```python
{
    "messages": [SystemMessage, HumanMessage, AIMessage, ToolMessage, ...]
}
```

Each node reads from state and writes back to it. The full conversation history — including all tool calls and their results — accumulates in `messages`. This is what gets printed when you trace the agent's reasoning.

---

## `create_agent` vs LangGraph — Side by Side

| Dimension | `create_agent` | LangGraph |
|---|---|---|
| **Graph structure** | Fixed: LLM → tools → LLM → ... → end | You define nodes, edges, and conditions |
| **Behaviour control** | System prompt only — LLM self-governs | Code-level: guaranteed branches, loops, limits |
| **Branching** | LLM decides when to stop | Conditional edges in code: `if state["confidence"] < 0.8: retry` |
| **Loop control** | Runs until LLM stops calling tools | You can cap iterations, add timeouts, force exits |
| **State schema** | Fixed: just `messages` | You define a typed state dict with any fields |
| **Multiple paths** | Not possible | Yes — fan-out, fan-in, parallel branches |
| **Human-in-the-loop** | Not supported | Yes — `interrupt_before`/`interrupt_after` on any node |
| **Persistence** | None — stateless per invocation | Checkpointers save state to DB; agents can resume |
| **Multi-agent** | Not supported | Yes — graphs can call other graphs as subgraphs |
| **Predictability** | Lower — LLM can surprise you | Higher — graph topology is deterministic |
| **Complexity** | Low | Higher |

---

## When to Use Each

**Use `create_agent` when:**
- The task is open-ended and the LLM should self-direct
- You trust the LLM to decide when it has enough information
- You want fast iteration with minimal boilerplate
- The stakes of unpredictable behaviour are low

**Use LangGraph when:**
- You need guaranteed behaviour ("always validate before answering")
- You need to cap iterations or enforce timeouts
- You need branching based on intermediate results
- You need human approval before certain actions
- You need the agent to persist state across sessions
- You are building multi-agent systems
- You need to reason about or test the agent's control flow

---

## The Key Insight

`create_agent` **trusts the LLM** to govern its own behaviour. LangGraph **codifies** that behaviour so it is no longer up to the LLM.

This is the same tradeoff as:
- A junior developer given a verbal brief (prompt) vs. a spec with acceptance criteria (graph)
- A free-form conversation vs. a structured workflow

For simple agentic tasks, the verbal brief is fine and faster. For production systems where correctness matters, you want the spec.

---

## How This Relates to What You Built

`03_agent/main.py` uses `create_agent` — which compiles to this LangGraph internally:

```
START → call_llm → [tool calls?] → run_tools → call_llm → [no tool calls] → END
```

`04_langgraph` will make this graph **explicit** — you will write each node and edge yourself, then add branching, state fields beyond just messages, and persistence.
