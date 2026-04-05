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

### Up Next (Core)
- [x] **06_human_in_loop** — interrupt()/Command(resume=...): pause mid-node for human review, resume with decision
- [x] **07_streaming** — stream(), stream_mode="messages" (tokens), stream_mode="updates" (delta)
- [x] **08_multi_agent** — Supervisor pattern: Command(goto=...) routing, shared state as communication channel
- [ ] **09_subgraphs** — Graphs nested inside graphs; composing complex workflows

### Up Next (Advanced)
- [ ] **10_long_term_memory** — Store API: semantic memory across threads
- [x] **09_tool_nodes** — ToolNode prebuilt + tools_condition routing; @tool decorator; parallel tool calls
- [ ] **12_map_reduce** — Fan-out/fan-in: run a node in parallel over a list

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
├── 06_human_in_loop/
│   └── main.py                   ← complete, working — email draft with approve/reject pause
└── 07_multi_agent/               ← next
```

## Key Files
- `.env` — OpenAI API key (`OPENAI_API_KEY`)
- `01_chatbot/main.py` — complete CLI chatbot with memory
- `02_rag/main.py` — complete RAG pipeline
- `03_agent/main.py` — agentic RAG with markdown-aware splitter and full message trace
- `rag_limitations_non_text.md` — reference doc on RAG limits for non-text content

## How to Resume
1. Activate venv: `venv\Scripts\activate`
2. `09_tool_nodes` is complete — start `09_subgraphs` or `10_long_term_memory` next
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
- LangGraph design critique: thread_id is chatbot-biased naming; configurable dict shape is convention not types; compile() injects hidden behavior
- Human-in-the-loop: interrupt() pauses mid-node, Command(resume=...) resumes with human decision; MemorySaver vs SqliteSaver tradeoff (06_human_in_loop)
- Streaming: stream() per-node, stream_mode="messages" per-token, stream_mode="updates" delta (07_streaming)
- Multi-agent supervisor: Command(goto=...) routing, shared messages as agent communication channel (08_multi_agent)
- ToolNode + tools_condition: prebuilt tool execution, @tool decorator, parallel tool calls, bind_tools() (09_tool_nodes)

## What's Next — 09_subgraphs

Compose complex workflows by nesting graphs inside graphs:
- A subgraph is a compiled graph used as a node in a parent graph
- Each subgraph has its own state; parent and child share fields by name
- Enables reusable agent components and cleaner separation of concerns
