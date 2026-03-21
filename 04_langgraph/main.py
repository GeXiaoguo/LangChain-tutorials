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

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

http_client = httpx.Client(verify=False)
http_async_client = httpx.AsyncClient(verify=False)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    http_client=http_client,
    http_async_client=http_async_client,
)

embeddings = OpenAIEmbeddings(
    http_client=http_client,
    http_async_client=http_async_client,
)

# --- Load and index documents ---
loader = TextLoader("sample.txt")
documents = loader.load()

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
    ("#", "Header1"),
    ("##", "Header2"),
    ("###", "Header3"),
])
chunks = splitter.split_text(documents[0].page_content)
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------------------------------------------------------------------------
# State — this is what flows between nodes. Unlike create_agent which only
# has "messages", we define exactly what fields we need.
# ---------------------------------------------------------------------------
class RAGState(TypedDict):
    question: str
    context: list[str]   # chunks retrieved so far
    answer: str
    search_attempts: int

# ---------------------------------------------------------------------------
# Nodes — each is a plain function: State -> partial State update
# ---------------------------------------------------------------------------
def retrieve(state: RAGState) -> dict:
    print(f"\n[retrieve] Attempt {state['search_attempts'] + 1}")
    docs = retriever.invoke(state["question"])
    context = [doc.page_content for doc in docs]
    print(f"[retrieve] Got {len(context)} chunks")
    return {
        "context": context,
        "search_attempts": state["search_attempts"] + 1,
    }

def generate(state: RAGState) -> dict:
    print(f"\n[generate] Building answer from {len(state['context'])} chunks")
    context_text = "\n\n".join(state["context"])
    response = llm.invoke([
        SystemMessage(content="Answer the question using only the provided context. Be concise."),
        HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {state['question']}"),
    ])
    return {"answer": response.content}

# ---------------------------------------------------------------------------
# Conditional edge — this is the routing logic create_agent hides from you.
# Returns the name of the next node to run.
# ---------------------------------------------------------------------------
def grade_and_route(state: RAGState) -> str:
    print(f"\n[grade] Evaluating context sufficiency...")
    context_text = "\n\n".join(state["context"])
    response = llm.invoke([
        SystemMessage(content="You are a relevance grader. Reply with only 'sufficient' or 'retry'."),
        HumanMessage(content=(
            f"Question: {state['question']}\n\n"
            f"Context:\n{context_text}\n\n"
            "Is this context sufficient to answer the question?"
        )),
    ])
    decision = response.content.strip().lower()
    print(f"[grade] Decision: {decision}")

    if state["search_attempts"] >= 3:
        print("[grade] Max attempts reached — proceeding to generate")
        return "generate"

    return "generate" if "sufficient" in decision else "retrieve"

# ---------------------------------------------------------------------------
# Graph — explicit nodes, edges, and conditional routing
#
#   START → retrieve → [grade] → generate → END
#                         │
#                         └──── retrieve (loop back if not sufficient)
# ---------------------------------------------------------------------------
graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.add_edge(START, "retrieve")
graph.add_conditional_edges(
    "retrieve",         # from this node
    grade_and_route,    # call this function to decide
    {
        "generate": "generate",   # if it returns "generate" → go to generate
        "retrieve": "retrieve",   # if it returns "retrieve" → loop back
    }
)
graph.add_edge("generate", END)

app = graph.compile()

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
question = "If I get a rating of 4 and resign, what happens to my bonus and unused leave?"
print(f"Question: {question}")

result = app.invoke({
    "question": question,
    "context": [],
    "answer": "",
    "search_attempts": 0,
})

print(f"\n{'='*60}")
print(f"Answer: {result['answer']}")
print(f"Total search attempts: {result['search_attempts']}")
