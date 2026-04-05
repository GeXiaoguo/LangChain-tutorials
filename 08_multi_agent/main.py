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
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from typing import TypedDict, Annotated, Literal

http_client = httpx.Client(verify=False)
http_async_client = httpx.AsyncClient(verify=False)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    http_client=http_client,
    http_async_client=http_async_client,
)

# ---------------------------------------------------------------------------
# State — shared across all agents. Each agent reads and appends to messages.
# ---------------------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ---------------------------------------------------------------------------
# Specialist agents — each does one job, then hands control back to supervisor
#
# They return Command(goto="supervisor") so the supervisor can decide
# whether to call another specialist or declare the task done.
# ---------------------------------------------------------------------------
def researcher(state: State) -> Command[Literal["supervisor"]]:
    print("\n[researcher] Answering factual question...")
    response = llm.invoke([
        SystemMessage(content=(
            "You are a factual researcher. Answer questions accurately and concisely. "
            "Prefix your response with 'RESEARCH: '"
        )),
    ] + state["messages"])
    print(f"[researcher] {response.content[:80]}...")
    return Command(
        update={"messages": [response]},
        goto="supervisor",
    )

def writer(state: State) -> Command[Literal["supervisor"]]:
    print("\n[writer] Drafting polished prose...")
    response = llm.invoke([
        SystemMessage(content=(
            "You are a professional writer. Take the research notes in the conversation "
            "and rewrite them as polished, engaging prose for a general audience. "
            "Prefix your response with 'DRAFT: '"
        )),
    ] + state["messages"])
    print(f"[writer] {response.content[:80]}...")
    return Command(
        update={"messages": [response]},
        goto="supervisor",
    )

# ---------------------------------------------------------------------------
# Supervisor — the routing brain.
#
# Reads the full conversation and decides which specialist to call next,
# or returns "FINISH" when the task is complete.
#
# Command(goto=...) is how it routes — LangGraph jumps to the named node.
# ---------------------------------------------------------------------------
SUPERVISOR_SYSTEM = SystemMessage(content="""
You are a supervisor managing a team of two specialists:
- "researcher": answers factual questions and gathers information
- "writer": takes research and rewrites it as polished prose

Given the conversation so far, decide who should act next.
Reply with ONLY one of these exact words: researcher, writer, FINISH

Rules:
- Call researcher first if factual information is needed
- Call writer after researcher has provided facts, to produce the final draft
- Reply FINISH when the writer has produced a final draft
""".strip())

def supervisor(state: State) -> Command[Literal["researcher", "writer", "__end__"]]:
    print("\n[supervisor] Deciding next step...")
    response = llm.invoke([SUPERVISOR_SYSTEM] + state["messages"])
    decision = response.content.strip()
    print(f"[supervisor] -> {decision}")

    if decision == "FINISH":
        return Command(goto=END)
    return Command(goto=decision)

# ---------------------------------------------------------------------------
# Graph
#
#   START → supervisor → researcher → supervisor → writer → supervisor → END
#                  ↑__________________________________________|
#
# The supervisor is the hub. Every specialist routes back to it.
# The supervisor decides when to stop.
# ---------------------------------------------------------------------------
graph = StateGraph(State)

graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)

graph.add_edge(START, "supervisor")
# No add_edge needed for specialist → supervisor — Command(goto=) handles routing

app = graph.compile()

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
task = "Write a short paragraph explaining how transformers work in AI."
print(f"Task: {task}")
print("=" * 60)

result = app.invoke({"messages": [HumanMessage(task)]})

print("\n" + "=" * 60)
print("FINAL CONVERSATION TRACE")
print("=" * 60)
for msg in result["messages"]:
    role = msg.__class__.__name__.replace("Message", "")
    print(f"\n[{role}]\n{msg.content}")
