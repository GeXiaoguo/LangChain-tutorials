# 07 — Streaming

## What This Example Does

Demonstrates the three streaming modes LangGraph supports. The graph is
intentionally simple (one `chat` node) so the focus stays on the streaming
API itself.

## The Three Modes

### Mode 1: `stream()` — one update per node

```python
for chunk in app.stream(input_state):
    for node_name, update in chunk.items():
        print(node_name, update)
```

- Yields a dict `{ "node_name": <state update> }` after **each node completes**
- You get the full state update, not individual tokens
- Useful for multi-node graphs where you want to know when each step finishes

### Mode 2: `stream(stream_mode="messages")` — token by token

```python
for chunk, metadata in app.stream(input_state, stream_mode="messages"):
    if isinstance(chunk, AIMessageChunk) and chunk.content:
        print(chunk.content, end="", flush=True)
```

- Yields `(AIMessageChunk, metadata)` tuples as the LLM generates each token
- `chunk.content` is a partial string — print without newline to see the
  response build up in real time
- This is what you use for any real UI (web, CLI, chat interface)

### Mode 3: `stream(stream_mode="updates")` — changed fields only

```python
for chunk in app.stream(input_state, stream_mode="updates"):
    for node_name, update in chunk.items():
        print(node_name, update)  # only the fields that changed
```

- Like Mode 1 but the dict contains only the **delta** — fields that changed
- Useful for large states where you don't want to receive the entire state
  on every node update

## Comparison

| Mode | Granularity | What you receive | Use case |
|---|---|---|---|
| `stream()` | Per node | Full state update | Knowing when each step finishes |
| `stream_mode="messages"` | Per token | `AIMessageChunk` | Real-time UI output |
| `stream_mode="updates"` | Per node | Changed fields only | Large state, efficiency |

## `invoke()` vs `stream()`

`invoke()` is just `stream()` that discards all intermediate chunks and
returns the final state. They run the same graph — the only difference is
what you do with the output as it arrives.
