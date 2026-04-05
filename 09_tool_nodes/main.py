import ssl
import requests as _requests

ssl._create_default_https_context = ssl._create_unverified_context

_original_send = _requests.Session.send
def _send_no_verify(self, request, **kwargs):
    kwargs["verify"] = False
    return _original_send(self, request, **kwargs)
_requests.Session.send = _send_no_verify

from dotenv import load_dotenv
import httpx

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated

http_client = httpx.Client(verify=False)
http_async_client = httpx.AsyncClient(verify=False)

# ---------------------------------------------------------------------------
# Tools — two ways to provide the LLM description:
#
# Pattern A (conventional): docstring as description.
#   The @tool decorator reads fn.__doc__ at decoration time.
#   Downside: the docstring serves two audiences (devs and the LLM) and
#   the coupling is implicit — a developer tidying the docstring can
#   accidentally break tool selection.
#
# Pattern B (explicit): description= keyword argument.
#   The LLM description is separate from the developer docstring.
#   More verbose but clearer about intent.
# ---------------------------------------------------------------------------

# Pattern A — docstring is the LLM description
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Examples: '2 + 2', '10 * 3.5', '100 / 4'."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

# Pattern B — explicit description separate from developer docstring
@tool(description="Get the current weather for a city. Examples: 'London', 'Tokyo', 'Sydney'.")
def get_weather(city: str) -> str:
    """Internal mock weather tool. Returns hardcoded data — replace with a real API call."""
    # Mocked — in a real app this would call a weather API
    weather_data = {
        "london": "12C, overcast",
        "sydney": "24C, sunny",
        "tokyo": "18C, partly cloudy",
    }
    return weather_data.get(city.lower(), f"No weather data for {city}")

tools = [calculate, get_weather]

# ---------------------------------------------------------------------------
# LLM with tools bound.
#
# .bind_tools() tells the LLM what tools are available. When the LLM decides
# to use one, it returns an AIMessage with tool_calls populated instead of
# plain text content.
# ---------------------------------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    http_client=http_client,
    http_async_client=http_async_client,
).bind_tools(tools)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ---------------------------------------------------------------------------
# Nodes
#
# "llm_node" — calls the LLM. If the LLM wants to use a tool, it returns
#              an AIMessage with tool_calls; ToolNode handles the rest.
#
# ToolNode   — inspects the last AIMessage for tool_calls, executes each
#              tool, appends ToolMessage results, hands back to the LLM.
#              You don't write this node — it's prebuilt.
# ---------------------------------------------------------------------------
def llm_node(state: State) -> dict:
    print("\n[llm] Thinking...")
    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant. Use tools when needed."),
    ] + state["messages"])

    if response.tool_calls:
        print(f"[llm] Wants to call: {[tc['name'] for tc in response.tool_calls]}")
    else:
        print(f"[llm] Final answer: {response.content[:80]}")

    return {"messages": [response]}

# ---------------------------------------------------------------------------
# Graph
#
#   START -> llm_node -> tools_condition -> ToolNode -> llm_node -> ... -> END
#                              |
#                              +-- (no tool calls) --> END
#
# tools_condition is a prebuilt routing function:
#   - if last message has tool_calls -> route to "tools"
#   - otherwise                      -> route to END
# ---------------------------------------------------------------------------
graph = StateGraph(State)

graph.add_node("llm", llm_node)
graph.add_node("tools", ToolNode(tools))   # prebuilt — no code needed

graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", tools_condition)  # prebuilt routing
graph.add_edge("tools", "llm")                       # always return to LLM after tool use

app = graph.compile()

# ---------------------------------------------------------------------------
# Run — a question that needs both tools
# ---------------------------------------------------------------------------
question = "What is 15 * 7, and what is the weather like in Tokyo?"
print(f"Question: {question}")
print("=" * 60)

result = app.invoke({"messages": [HumanMessage(question)]})

print("\n" + "=" * 60)
print("FULL MESSAGE TRACE")
print("=" * 60)
for msg in result["messages"]:
    role = msg.__class__.__name__.replace("Message", "")
    content = msg.content if msg.content else str(getattr(msg, "tool_calls", ""))
    print(f"\n[{role}]\n{content}")
