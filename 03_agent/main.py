import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import requests
import urllib3
urllib3.disable_warnings()
_original_send = requests.Session.send
def _send_no_verify(self, request, **kwargs):
    kwargs["verify"] = False
    return _original_send(self, request, **kwargs)
requests.Session.send = _send_no_verify

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents import create_agent

load_dotenv()

# --- Build vector store from sample.txt ---
loader = TextLoader("sample.txt")
docs = loader.load()
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "section")])
chunks = splitter.split_text(docs[0].page_content)

embeddings = OpenAIEmbeddings(http_client=httpx.Client(verify=False))
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- Define retriever as a tool ---
search_tool = create_retriever_tool(
    retriever,
    name="search_docs",
    description="Search the Acme Corp handbook for policies and facts. Use this before answering any question.",
)

# --- Build agent (LangChain 1.x API) ---
llm = ChatOpenAI(model="gpt-4o-mini", http_client=httpx.Client(verify=False))

agent = create_agent(
    model=llm,
    tools=[search_tool],
    system_prompt=(
        "You are a helpful HR assistant for Acme Corp. "
        "Use the search_docs tool to look up relevant handbook sections before answering. "
        "You may search multiple times if needed."
    ),
)

print("--- Agentic RAG ready (Acme Corp HR). Type 'quit' to exit. ---")
print("Try: 'If I get a rating of 4 and resign, what happens to my bonus and unused leave?'\n")

while True:
    question = input("You: ").strip()
    if question.lower() == "quit":
        print("Goodbye!")
        break
    if not question:
        continue

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    print("\n--- Full message trace ---")
    for i, msg in enumerate(result['messages']):
        print(f"\n[{i}] {type(msg).__name__}")
        print(f"    {msg.content[:300] if msg.content else '<tool call>'}")
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"    >> tool_call: {tc['name']}({tc['args']})")
    print("\n--- End trace ---")

    print(f"\nBot: {result['messages'][-1].content}\n")
