# LangChain Learning — Handoff

## Goal
Build LangChain concepts step by step: chatbot → RAG → agentic RAG → LangGraph.

## Progress

### Completed
- [x] **01_chatbot** — CLI chatbot with manual message history (in-context memory)
- [x] **02_rag** — RAG pipeline: load → chunk → embed → Chroma → retrieve → generate
- [x] **03_agent** — Agentic RAG: LLM uses a retriever as a tool, calls it multiple times via ReAct loop

- [x] **04_langgraph** — Explicit state graphs: nodes, edges, conditional routing, corrective RAG loop
- [x] **05_persistence** — LangGraph checkpointers: SqliteSaver, add_messages reducer, thread isolation

### Up Next
- [ ] **06_human_in_loop** — interrupt_before/after: pause graph execution for human review/approval before continuing

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
├── 04_langgraph/
│   ├── main.py                   ← complete, working — corrective RAG with grade_and_route loop
│   └── sample.txt
├── 05_persistence/
│   ├── main.py                   ← complete, working — SqliteSaver + thread isolation demo
│   └── memory.db                 ← SQLite checkpoint file (git-ignored)
└── 06_human_in_loop/             ← next
```

## Key Files
- `.env` — OpenAI API key (`OPENAI_API_KEY`)
- `01_chatbot/main.py` — complete CLI chatbot with memory
- `02_rag/main.py` — complete RAG pipeline
- `03_agent/main.py` — agentic RAG with markdown-aware splitter and full message trace
- `rag_limitations_non_text.md` — reference doc on RAG limits for non-text content

## How to Resume
1. Activate venv: `venv\Scripts\activate`
2. `05_persistence` is complete — start `06_human_in_loop` next
3. Good test question (works for 03_agent and 04_langgraph): *"If I get a rating of 4 and resign, what happens to my bonus and unused leave?"*

## Concepts Covered
- In-context memory (01_chatbot)
- Chunking, embeddings, vector stores, similarity search (02_rag)
- Five memory approaches: in-context, RAG, fine-tuning, tool use, external memory
- Agentic RAG / ReAct loop — LLM decides what to retrieve and when (03_agent)
- Markdown-aware splitting — keep header + body together instead of arbitrary character chunks
- RAG limitations for non-text content: images, diagrams, Mermaid, code (rag_limitations_non_text.md)
- How Claude Code works — no RAG, pure tool-based navigation; CLAUDE.md as human-curated index
- Explicit LangGraph: State, nodes, edges, conditional routing, corrective RAG retry loop (04_langgraph)
- LangGraph persistence: SqliteSaver, add_messages reducer, thread_id isolation, get_state() inspection (05_persistence)

## What's Next — 06_human_in_loop

With checkpointers in place, LangGraph can **pause mid-graph** and wait for a
human to approve or modify state before continuing.

Key concepts to cover:
- `interrupt_before=["node_name"]` — pause before a sensitive node runs
- `app.update_state(config, {...})` — inject human input into the checkpoint
- `app.invoke(None, config)` — resume from where it was interrupted
- Use case: agent wants to send an email / delete data — human must approve first
