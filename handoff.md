# LangChain Learning — Handoff

## Goal
Build LangChain concepts step by step: chatbot → RAG → agentic RAG → LangGraph.

## Progress

### Completed
- [x] **01_chatbot** — CLI chatbot with manual message history (in-context memory)
- [x] **02_rag** — RAG pipeline: load → chunk → embed → Chroma → retrieve → generate
- [x] **03_agent** — Agentic RAG: LLM uses a retriever as a tool, calls it multiple times via ReAct loop

### Up Next
- [ ] **04_langgraph** — Explicit state graphs: nodes, edges, branching, loops, conditional routing

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
├── rag_limitations_non_text.md   ← deep dive on RAG limits for images/diagrams/code
├── 01_chatbot/
│   └── main.py                   ← complete, working
├── 02_rag/
│   ├── main.py                   ← complete, working
│   └── sample.txt                ← sample document about LangChain/RAG concepts
├── 03_agent/
│   ├── main.py                   ← complete, working — markdown-aware splitter + tool-calling agent
│   └── sample.txt                ← Acme Corp employee handbook (multi-section, good for multi-hop questions)
└── 04_langgraph/                 ← next
```

## Key Files
- `.env` — OpenAI API key (`OPENAI_API_KEY`)
- `01_chatbot/main.py` — complete CLI chatbot with memory
- `02_rag/main.py` — complete RAG pipeline
- `03_agent/main.py` — agentic RAG with markdown-aware splitter and full message trace
- `rag_limitations_non_text.md` — reference doc on RAG limits for non-text content

## How to Resume
1. Activate venv: `venv\Scripts\activate`
2. `03_agent` is complete and working — start `04_langgraph` next
3. Good test question for 03_agent: *"If I get a rating of 4 and resign, what happens to my bonus and unused leave?"*

## Concepts Covered
- In-context memory (01_chatbot)
- Chunking, embeddings, vector stores, similarity search (02_rag)
- Five memory approaches: in-context, RAG, fine-tuning, tool use, external memory
- Agentic RAG / ReAct loop — LLM decides what to retrieve and when (03_agent)
- Markdown-aware splitting — keep header + body together instead of arbitrary character chunks
- RAG limitations for non-text content: images, diagrams, Mermaid, code (rag_limitations_non_text.md)
- How Claude Code works — no RAG, pure tool-based navigation; CLAUDE.md as human-curated index

## What's Next — 04_langgraph

`create_agent` in 03_agent is actually a thin wrapper over LangGraph. LangGraph makes the graph explicit:

- **Nodes** — each step is a named node (call LLM, call tool, validate, route)
- **Edges** — define what runs after what
- **Conditional edges** — branch based on LLM output ("search again" vs "answer now")
- **State** — a typed dict passed between nodes, accumulates as the graph runs
- **Persistence** — checkpointers save state so agents can resume across sessions

This is the foundation for serious multi-step agents, human-in-the-loop workflows, and multi-agent systems.
