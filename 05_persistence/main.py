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
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated

http_client = httpx.Client(verify=False)
http_async_client = httpx.AsyncClient(verify=False)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    http_client=http_client,
    http_async_client=http_async_client,
)

# ---------------------------------------------------------------------------
# State
#
# The `add_messages` reducer is the key difference from 04_langgraph.
# Instead of replacing `messages` on each update, it *appends* new messages
# to whatever the checkpointer already saved for this thread.
#
# This means you only pass the NEW message on each invocation — the full
# history is loaded automatically from the checkpoint.
# ---------------------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
SYSTEM = SystemMessage(content=(
    "You are a concise assistant. Keep answers to 2-3 sentences."
))

def chat(state: State) -> dict:
    # state["messages"] already contains the full history for this thread,
    # loaded from the checkpoint and merged with the new incoming message.
    response = llm.invoke([SYSTEM] + state["messages"])
    # Return only the new message — add_messages appends it to the history.
    return {"messages": [response]}

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
graph = StateGraph(State)
graph.add_node("chat", chat)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

# ---------------------------------------------------------------------------
# Run — compiled with SqliteSaver so state survives across process restarts.
#
# Each thread_id is an independent conversation. Passing the same thread_id
# a second time resumes exactly where it left off.
# ---------------------------------------------------------------------------
with SqliteSaver.from_conn_string("memory.db") as checkpointer:
    app = graph.compile(checkpointer=checkpointer)

    config_alice = {"configurable": {"thread_id": "alice"}}
    config_bob   = {"configurable": {"thread_id": "bob"}}

    print("=" * 60)
    print("THREAD: alice — turn 1")
    print("=" * 60)
    result = app.invoke(
        {"messages": [HumanMessage("What is RAG in one sentence?")]},
        config_alice,
    )
    print(f"Alice: {result['messages'][-1].content}")

    print("\n" + "=" * 60)
    print("THREAD: alice — turn 2  (should remember turn 1)")
    print("=" * 60)
    result = app.invoke(
        {"messages": [HumanMessage("What technique did I just ask you about?")]},
        config_alice,
    )
    print(f"Alice: {result['messages'][-1].content}")

    print("\n" + "=" * 60)
    print("THREAD: bob — turn 1  (fresh thread, no shared history)")
    print("=" * 60)
    result = app.invoke(
        {"messages": [HumanMessage("What technique did I just ask you about?")]},
        config_bob,
    )
    print(f"Bob: {result['messages'][-1].content}")

    # ---------------------------------------------------------------------------
    # Inspect saved state — this is what the checkpointer stored for alice.
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CHECKPOINT — alice's saved message history")
    print("=" * 60)
    snapshot = app.get_state(config_alice)
    for msg in snapshot.values["messages"]:
        role = msg.__class__.__name__.replace("Message", "")
        print(f"  [{role}] {msg.content}")
