# RAG Limitations for Non-Text Content and How Agents Cope

## The core problem

RAG was designed for natural language text. Vector stores hold numbers derived from text. Everything non-text must be converted to text before it can be indexed — and that conversion often destroys the meaning.

Diagrams exist precisely because text fails to capture certain concepts. A network topology, a UML class diagram, a circuit schematic — these were drawn because **spatial and structural relationships cannot be expressed in text without losing meaning**. Converting them back to text to store in RAG is backwards.

---

## Content types and their RAG problems

### Raster images (screenshots, JPGs, PNGs)

Must be converted to text first via:
- **OCR** — works for screenshots of documents/UI, fails on diagrams
- **Vision LLM captioning** — GPT-4o or Claude describes the image; the caption becomes the chunk

The caption quality determines retrievability. A poor caption makes the image invisible to search.

### Vector diagrams (SVG)

SVG is XML text but raw tags are meaningless to an embedding model. Options:
- Render to PNG → treat as screenshot
- Extract text labels and relationships → embed those

### Mermaid diagrams

Mermaid is plain text and can be embedded directly, but:
- No LangChain splitter understands Mermaid syntax semantically — no splitter knows a `subgraph` boundary is more meaningful than a line break
- Even if retrieved perfectly, LLMs read Mermaid as token sequences, not as a graph
- Concepts like "left of", "branches back to", "parallel paths" are spatial — text linearises and destroys them
- Complex nested subgraphs, timing diagrams, ER diagrams with many cardinalities all degrade severely

### PDFs

Mixed — may contain text, images, tables, diagrams. Text layers extract cleanly; scanned pages require OCR; embedded diagrams hit all the image problems above.

---

## The retrieval problem still exists even with multimodal models

Multimodal models (GPT-4o, Claude 3.5 Sonnet) can *look at* a diagram and reason about it visually — genuinely better than text conversion. But the retrieval step still requires a text proxy:

```
diagram.png  →  caption (via vision LLM)  →  embed caption  →  stored in vector index

Query: "how does auth work?"
→ embed query → find closest caption → retrieve diagram.png
→ inject image bytes into context → multimodal model reads it
```

Retrieval is still text-based. Understanding is visual. They are separate problems.

This full pipeline — caption indexing + image retrieval + multimodal reasoning — is technically possible today but not productised cleanly in any standard RAG framework.

---

## What a better system would look like

```
Ingest pipeline:
  *.md, *.txt, *.py   → text splitter → embed
  *.png, *.jpg        → vision LLM caption + metadata → embed caption
  mermaid in .md      → render → caption → embed
  relationship data   → graph DB (nodes + edges)

Agent query:
  → searches text index (finds relevant code/docs)
  → searches image index by caption similarity (finds relevant diagram)
  → injects both text and image bytes into context
  → multimodal model reasons over everything together
```

The closer approach to human cognition is **GraphRAG** (Microsoft) — parses documents into entity/relationship graphs, answers from graph traversal, falls back to raw text only when needed. Maps much closer to how humans understand diagrams: as a web of relationships, not a paragraph.

---

## How Claude Code copes — and what it reveals about RAG

Claude Code does not use RAG at all. It uses **on-demand precise search tools**:

```
"Fix the auth bug"
→ Grep("auth")              — finds auth-related files by exact match
→ Glob("**/*auth*")         — finds auth files by name pattern
→ Read("src/auth/middleware.py")  — reads the exact file
→ understands enough to act
→ session ends → understanding gone
```

No embeddings, no vector store, no pre-indexing. This works because code has **exact, deterministic structure** — function names, class names, imports are always spelled the same way. Grep never misses a literal match. Vector similarity search is unnecessary and would be worse.

The intelligence is not in having memorised the codebase. It's in **knowing how to navigate** — deciding what to search for, reading the minimum necessary, building a local understanding on demand just enough to act.

This is why Claude Code can get things wrong on large codebases: it only read *enough* to act, and "enough" might have missed a relevant file three directories away.

---

## The non-text asset problem in Claude Code

Claude Code has no persistent index. Between sessions it knows nothing about your codebase. Non-text files — diagrams, architecture images, ERDs — are completely invisible unless explicitly surfaced.

**The only mechanism is `CLAUDE.md`** — a human-curated index that compensates for the lack of automatic indexing:

```markdown
## Architecture Diagrams

- `docs/architecture.png` — overall system architecture. Refer to this
  before adding new services. Shows how auth, API, and DB layers connect.

- `docs/data_flow.png` — payment request flow end to end.
  Relevant when touching anything in `src/payments/`.

- `docs/erd.png` — entity relationship diagram for the database schema.
  Check this before writing migrations.
```

Two things are required:
1. **The file path** — so Claude Code can read the file
2. **When it's relevant** — so Claude Code knows to reach for it

Without the "when relevant" hint, even a listed file may be ignored. `CLAUDE.md` is essentially a hand-written retrieval index — you are doing the work that a multimodal RAG pipeline would do automatically.

---

## Summary

| Content type | RAG quality | Better approach |
|---|---|---|
| Natural language text | Good | Standard RAG |
| Markdown with structure | Good with header-aware splitter | `MarkdownHeaderTextSplitter` |
| Mermaid diagrams | Poor — no semantic splitter, LLM can't visualise | Render → vision LLM caption |
| Screenshots / PNGs | Passable with vision LLM captioning | Caption → embed → retrieve image |
| Complex architecture diagrams | Poor | Multimodal model + GraphRAG |
| Code | RAG is the wrong tool entirely | Grep/Glob/Read — exact search |

The honest state of the field: RAG is one tool in the toolbox, not a universal answer. It works well for natural language corpora. For structured, visual, or relational knowledge, the industry is still figuring it out.
