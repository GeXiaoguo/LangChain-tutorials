# LangChain Learnings

## ChatPromptTemplate is heavy abstraction over simple string formatting

`ChatPromptTemplate` is essentially `str.format()` with extra ceremony. This works just as well and is more explicit:

```python
from langchain_core.messages import SystemMessage, HumanMessage

response = llm.invoke([
    SystemMessage(content=template.format(topic="space exploration")),
    HumanMessage(content="What was the first animal in space?")
])
```

Use `ChatPromptTemplate` only when the complexity justifies it:
- Reusing prompts across many inputs
- Composing prompts from partials
- Serializing/loading prompts from files
- Auto input validation via schema

For simple calls, raw message lists are clearer.

## Memory is an illusion — the LLM is stateless

The LLM has no memory between calls. You build memory yourself by maintaining a list of messages and sending the full history on every call:

```python
history = [SystemMessage(...)]
history.append(HumanMessage(content=user_input))
response = llm.invoke(history)          # full history sent every time
history.append(AIMessage(content=response.content))
```

Each call sends a longer and longer list. "Memory" is entirely client-side.

## HumanMessage / AIMessage / SystemMessage are just role-tagged strings

They map directly to OpenAI's `{"role": "user/assistant/system", "content": "..."}` format. No logic — just typed wrappers to avoid typos and make provider translation easier.

Most major LLMs (OpenAI, Anthropic, Gemini, Mistral, LLaMA) support the same three roles. This is where LangChain's abstraction genuinely helps — swap one line (`ChatOpenAI` → `ChatAnthropic`) and the rest of the code stays the same.

## RAG: vectors find the chunk, but the LLM sees the original text

The vector store holds two things per chunk: a vector (for similarity search) and the original text + metadata (sent to the LLM). The LLM never sees numbers — it gets the raw text stuffed into the prompt as context.

```
Vector store entry:
  vector:   [0.12, -0.87, 0.34, ...]              ← used only for similarity search
  document: "...became profitable in 2005..."      ← sent to LLM
  metadata: { source: "history.pdf", page: 3 }    ← optional, useful for citations
```

## RAG: chunking can break semantic meaning that spans boundaries

A sentence like "...and became profitable in 2005" at the start of a chunk is a fragment without its preceding context. Mitigations:

- **Overlapping chunks** — repeat the last N tokens of the previous chunk at the start of the next
- **Sentence-aware splitting** — never cut mid-sentence
- **Parent-child chunking** — index small chunks for precision, retrieve the larger parent for context

## Five ways to give an LLM memory

The LLM itself is always stateless. "Memory" is always something we build around it:

| Approach | How it works | Best for | Limitation |
|---|---|---|---|
| **In-context** | Append all messages to history, send every time | Short conversations | Context window limit; cost grows each turn |
| **RAG** | Embed documents, retrieve relevant chunks at query time | Large static document collections | Retrieval can miss things; stale if docs change |
| **Fine-tuning** | Train the model on your data, bake knowledge into weights | Stable domain knowledge or style | Expensive, slow, hard to update |
| **Tool use** | Give the LLM functions it can call (DB query, search API) | Live or dynamic data | Needs tool infrastructure |
| **External memory** | LLM reads/writes to a dedicated memory store (e.g. MemGPT) | Long-running agents across sessions | Complex to implement |

In real products these are often combined — e.g. RAG for documents + in-context for conversation history + tool use for live data.

## RAG: embedding model options

The embedding model is independent of the generation LLM. Common options:

| Embedding model | Cost | Requires |
|---|---|---|
| `OpenAIEmbeddings` | ~$0.0001/1K tokens | OpenAI API key |
| `HuggingFaceEmbeddings` | Free | Download model locally (~100MB+) |
| `OllamaEmbeddings` | Free | Ollama running locally |
| `CohereEmbeddings` | Paid | Cohere API key |

The vector DB is tied to the embedding model, not the generation LLM — you can swap GPT-4 for Claude freely, but changing embedding models requires rebuilding the vector DB. For local/free, `HuggingFaceEmbeddings` with `all-MiniLM-L6-v2` is the most common choice.

## Agentic RAG: the LLM drives its own retrieval

Standard RAG does one retrieval then generates. Agentic RAG lets the LLM decide what to search for, how many times, based on what it finds — like a researcher who looks up one thing and discovers they need to look up something else.

### ReAct loop (Reason + Act)

```
Question → Thought → Action (search) → Observation → Thought → Action (search again) → ... → Answer
```

The LLM writes its own reasoning steps and tool calls until it has enough context to answer.

### How tools replace hard-coded retrieval

In standard RAG, retrieval is always called once with the user's question. In an agent, retrieval is a **tool** the LLM can call with its own search query, multiple times, in any order:

```python
search_tool = create_retriever_tool(retriever, name="search_docs",
    description="Search docs for facts.")
# LLM decides when to call it and with what query
```

### Agentic patterns

| Pattern | How it discovers context |
|---|---|
| **ReAct** | Think-act-observe loop; LLM writes its own search queries |
| **Self-RAG** | LLM critiques retrieved chunks — re-retrieves if not relevant enough |
| **Multi-hop RAG** | Each chunk reveals new entities to look up next |
| **LangGraph** | Graph of nodes — each node can retrieve, reason, branch, or loop back |

### When agentic RAG wins over standard RAG

- Questions requiring multiple lookups ("compare X and Y", "how does A relate to B?")
- Unknown retrieval strategy upfront — the agent figures out what to search
- Tasks needing tools beyond retrieval (live data, calculation, DB queries)

In LangChain: `create_tool_calling_agent` + `AgentExecutor` is the modern approach. LangGraph is the next level for complex multi-step agents.

## LangGraph: graph structure, state, and persistence

### Graph definition is separate from execution

A graph is built from nodes (functions) and edges (wiring). `compile()` is the separate step that makes it executable — and is where cross-cutting behavior like checkpointing is injected. The same graph can be compiled different ways:

```python
app_dev  = graph.compile(checkpointer=MemorySaver())   # tests
app_prod = graph.compile(checkpointer=PostgresSaver())  # production
app_bare = graph.compile()                              # no persistence
```

### State reducers — the same concept as React/Redux

Each field in `State` can declare a **reducer** via `Annotated[type, reducer_fn]`:

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

The reducer controls how updates are merged into existing state. Default is replace; `add_messages` appends. The node just returns what's new — the reducer handles the merge. This is the same concept as a Redux reducer, just declared on the type rather than called explicitly.

### Conditional routing returns a string, not a boolean

A routing function returns the name of the next node to run. This is what makes multi-branch routing possible:

```python
def grade_and_route(state) -> str:
    return "generate" if sufficient else "retrieve"
```

### Checkpointing: `compile()` injects hidden load/save around every node

`compile(checkpointer=...)` wraps every node execution with load/save logic. Your node code stays clean — the checkpointer is wired in at compile time, not in the node itself.

The checkpointer is an **object** (not a function) because it needs to hold a DB connection and expose multiple operations: `get`, `put`, `list`.

### `thread_id` scopes state — but the name is chatbot-biased

`thread_id` is LangGraph's reserved key inside `config["configurable"]` for identifying which state to load/save. Same ID resumes; different ID starts fresh. It maps to whatever a "unit of work" means in your domain — conversation, document, order, game session.

The naming is a historical artefact — LangGraph grew out of LangChain's chatbot origins. The mechanism is general-purpose but the names (thread, messages) reflect the original use case. Classic tech debt: the abstraction outgrew its metaphor.

### The `{"configurable": {"thread_id": "..."}}` shape is a hardcoded convention

LangGraph literally does `config["configurable"]["thread_id"]` internally. The nesting exists so user-defined runtime config doesn't collide with LangGraph's own keys. Typos fail silently — nothing in the type system enforces the key name.
