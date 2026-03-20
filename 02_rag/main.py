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
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- Phase 1: Index ---
# Load the document
loader = TextLoader("sample.txt")
docs = loader.load()

# Split into chunks (200 tokens, 40 token overlap)
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
chunks = splitter.split_documents(docs)

print(f"Document split into {len(chunks)} chunks\n")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk.page_content[:80]}...")

# Embed chunks and store in Chroma (in-memory)
embeddings = OpenAIEmbeddings(
    http_client=httpx.Client(verify=False)
)
vectorstore = Chroma.from_documents(chunks, embeddings)

# --- Phase 2: Retrieve + Generate ---
llm = ChatOpenAI(
    model="gpt-4o-mini",
    http_client=httpx.Client(verify=False)
)

print("\n--- RAG Q&A ready. Type 'quit' to exit. ---\n")

while True:
    question = input("You: ").strip()

    if question.lower() == "quit":
        print("Goodbye!")
        break

    if not question:
        continue

    # Find the 3 most relevant chunks
    retrieved = vectorstore.similarity_search(question, k=3)

    # Build context string from retrieved chunks
    context = "\n\n".join(doc.page_content for doc in retrieved)

    # Ask the LLM with context injected
    prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}"""

    response = llm.invoke(prompt)
    print(f"Bot: {response.content}\n")
