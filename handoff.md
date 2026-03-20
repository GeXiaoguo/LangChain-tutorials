# LangChain Learning — Handoff

## Goal
Build LangChain concepts step by step: chatbot → RAG → agentic RAG → (next TBD).

## Progress

### Completed
- [x] **01_chatbot** — CLI chatbot with manual message history (in-context memory)
- [x] **02_rag** — RAG pipeline: load → chunk → embed → Chroma → retrieve → generate

### Up Next
- [ ] **03_agent** — Agentic RAG: LLM uses a retriever as a tool, calls it multiple times via ReAct loop

## Environment Notes
- Python venv at `venv/` (not committed)
- API key in `.env` (not committed)
- **Corporate proxy does TLS inspection** — requires two SSL patches at the top of every script:
  1. `ssl._create_default_https_context = ssl._create_unverified_context` — covers Python's built-in urllib
  2. Monkeypatch `requests.Session.send` with `verify=False` — covers `tiktoken` and anything using `requests`
  3. `httpx.Client(verify=False)` passed explicitly to `ChatOpenAI` and `OpenAIEmbeddings`

## Structure
```
LangChain/
├── learnings.md
├── handoff.md
├── 01_chatbot/
│   └── main.py          ← complete, working
├── 02_rag/
│   ├── main.py          ← complete, working
│   └── sample.txt       ← sample document about LangChain/RAG concepts
└── 03_agent/
    ├── main.py          ← next step — agentic RAG with tool-calling agent
    └── sample.txt       ← Acme Corp employee handbook (multi-section, good for multi-hop questions)
```

## Key Files
- `.env` — OpenAI API key (`OPENAI_API_KEY`)
- `01_chatbot/main.py` — complete CLI chatbot with memory
- `02_rag/main.py` — complete RAG pipeline
- `03_agent/main.py` — agent skeleton, ready to run

## How to Resume
1. Activate venv: `venv\Scripts\activate`
2. Run `03_agent/main.py` — agentic RAG is the next step to explore
3. Good test question: *"If I get a rating of 4 and resign, what happens to my bonus and unused leave?"*
   (requires searching bonus policy + termination policy separately)

## Concepts Covered
- In-context memory (01_chatbot)
- Chunking, embeddings, vector stores, similarity search (02_rag)
- Five memory approaches: in-context, RAG, fine-tuning, tool use, external memory
- Agentic RAG / ReAct loop — LLM decides what to retrieve and when (03_agent)
