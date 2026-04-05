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
from langchain_core.messages import HumanMessage, SystemMessage, AIMessageChunk
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated

http_client = httpx.Client(verify=False)
http_async_client = httpx.AsyncClient(verify=False)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    http_client=http_client,
    http_async_client=http_async_client,
)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chat(state: State) -> dict:
    response = llm.invoke([
        SystemMessage(content="You are a concise assistant."),
    ] + state["messages"])
    return {"messages": [response]}

graph = StateGraph(State)
graph.add_node("chat", chat)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)
app = graph.compile()

input_state = {"messages": [HumanMessage("Explain what an LLM is in 3 sentences.")]}

# ---------------------------------------------------------------------------
# Mode 1: stream() — yields state snapshots after each node completes.
#
# Each chunk is a dict: { "node_name": <full state update from that node> }
# You get one chunk per node. Useful for multi-node graphs where you want to
# know when each step finishes, not individual tokens.
# ---------------------------------------------------------------------------
print("=" * 60)
print("MODE 1: stream() — one update per node")
print("=" * 60)

for chunk in app.stream(input_state):
    for node_name, update in chunk.items():
        print(f"[{node_name}] produced: {update['messages'][-1].content[:80]}...")

# ---------------------------------------------------------------------------
# Mode 2: stream(stream_mode="messages") — yields individual LLM tokens.
#
# Each chunk is a tuple: (AIMessageChunk, metadata)
# AIMessageChunk.content is a partial token string — print without newline
# to watch the response build up word by word.
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MODE 2: stream(stream_mode='messages') — token by token")
print("=" * 60)

print("Response: ", end="", flush=True)
for chunk, metadata in app.stream(input_state, stream_mode="messages"):
    if isinstance(chunk, AIMessageChunk) and chunk.content:
        print(chunk.content, end="", flush=True)
print()  # newline after stream ends

# ---------------------------------------------------------------------------
# Mode 3: stream(stream_mode="updates") — yields only the fields that changed.
#
# Same as stream() but the dict contains only the delta, not the full state.
# Useful for large states where you don't want to receive everything each time.
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MODE 3: stream(stream_mode='updates') — changed fields only")
print("=" * 60)

for chunk in app.stream(input_state, stream_mode="updates"):
    for node_name, update in chunk.items():
        num_messages = len(update.get("messages", []))
        print(f"[{node_name}] delta: {num_messages} new message(s)")
