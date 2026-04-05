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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
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
# ---------------------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]
    draft: str        # email draft produced by the agent
    approved: bool    # human decision

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def draft_email(state: State) -> dict:
    """Ask the LLM to write an email based on the conversation so far."""
    print("\n[draft_email] Writing draft...")
    response = llm.invoke([
        SystemMessage(content=(
            "You are an assistant that drafts professional emails. "
            "Write a concise email (subject line + 3-4 sentences) based on the user's request. "
            "Format: Subject: ...\n\n<body>"
        )),
    ] + state["messages"])
    draft = response.content
    print(f"\n--- DRAFT ---\n{draft}\n-------------")
    return {"draft": draft}

def human_review(state: State) -> Command:
    """
    Pause here and hand the draft back to the caller.

    `interrupt()` saves the graph state at this exact point and raises an
    exception that LangGraph catches. The caller's `app.invoke()` returns
    with the interrupted value instead of a final result.

    The graph resumes when the caller calls `app.invoke(Command(resume=...))`.
    Whatever is passed to `resume=` becomes the return value of `interrupt()`.
    """
    decision = interrupt({
        "message": "Review the email draft. Approve or reject?",
        "draft": state["draft"],
    })

    # `decision` is whatever the human passed to Command(resume=...)
    approved = decision.strip().lower() == "approve"
    return Command(update={"approved": approved})

def send_email(state: State) -> dict:
    """Only reached if approved. In a real app, this would call an email API."""
    if state["approved"]:
        print("\n[send_email] Sending email...")
        print(state["draft"])
        return {"messages": [{"role": "assistant", "content": "Email sent successfully."}]}
    else:
        print("\n[send_email] Skipping — rejected by human.")
        return {"messages": [{"role": "assistant", "content": "Email was not sent. Draft discarded."}]}

# ---------------------------------------------------------------------------
# Graph
#
#   START → draft_email → human_review → send_email → END
#                              ↑
#                         pauses here, waits for Command(resume=...)
# ---------------------------------------------------------------------------
graph = StateGraph(State)

graph.add_node("draft_email", draft_email)
graph.add_node("human_review", human_review)
graph.add_node("send_email", send_email)

graph.add_edge(START, "draft_email")
graph.add_edge("draft_email", "human_review")
graph.add_edge("human_review", "send_email")
graph.add_edge("send_email", END)

# MemorySaver is enough here — we just need state to survive between the two
# invoke() calls within the same process. No need for SQLite.
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
config = {"configurable": {"thread_id": "demo"}}

print("=" * 60)
print("TURN 1 — Submit request, graph runs until interrupt")
print("=" * 60)

# First invoke — runs draft_email, hits interrupt() in human_review, pauses.
result = app.invoke(
    {
        "messages": [HumanMessage("Write an email to the team announcing we are moving to a 4-day work week starting next month.")],
        "draft": "",
        "approved": False,
    },
    config,
)

# At this point the graph is paused. `result` contains the interrupted value.
interrupted = result.get("__interrupt__")
if interrupted:
    print("\n[PAUSED] Graph is waiting for human review.")
    print(f"Message: {interrupted[0].value['message']}")

print("\n" + "=" * 60)
print("TURN 2a — Human approves, graph resumes")
print("=" * 60)

result = app.invoke(Command(resume="approve"), config)
print(f"\nFinal status: {result['messages'][-1].content}")

# ---------------------------------------------------------------------------
# Run again with a fresh thread to show rejection
# ---------------------------------------------------------------------------
config2 = {"configurable": {"thread_id": "demo-reject"}}

print("\n" + "=" * 60)
print("REJECTION DEMO — same flow, human rejects")
print("=" * 60)

app.invoke(
    {
        "messages": [HumanMessage("Write an email to the CEO asking for a pay rise.")],
        "draft": "",
        "approved": False,
    },
    config2,
)

result = app.invoke(Command(resume="reject"), config2)
print(f"\nFinal status: {result['messages'][-1].content}")
