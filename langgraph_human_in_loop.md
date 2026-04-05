# LangGraph: Human-in-the-Loop Design

## Main Execution Path vs Human as a Node

### What you might expect — human as a node

```
START → draft_email → [human_node] → send_email → END
```

A human approval step looks like any other node. The framework handles
pausing, waiting, and resuming transparently. You don't see the mechanism.

### What LangGraph actually does — interrupt in the execution path

```
START → draft_email → human_review → send_email → END
                            ↑
                       interrupt() called here
                       graph freezes, control returns to caller
                       caller collects human input externally
                       caller calls invoke(Command(resume=...)) to continue
```

The human interaction is **not hidden inside a node** — it leaks out to the
caller as two separate `invoke()` calls.

### Why a true blocking human node doesn't work

A node is a function the graph runner calls synchronously. It must return
before the graph can checkpoint and exit. A human takes minutes, hours, or
days to respond — the process would hang, and if it dies, the in-flight state
is lost.

`interrupt()` solves this by:
1. Checkpointing state to the store at the interrupt point
2. Raising a special exception the runner catches (not a crash)
3. Returning the interrupted value to the caller immediately
4. Waiting for `invoke(Command(resume=...))` — which can come from a
   completely separate process, minutes or days later

### The gap between the two designs

| Property | Human as node (ideal) | interrupt() (actual) |
|---|---|---|
| Caller awareness | None — framework handles it | Caller must handle two invoke() calls |
| External trigger | Framework owns it | You own it (webhook, polling, queue) |
| Process lifetime | Framework manages | Your process must stay alive or restart |
| Where it exists | Temporal, Durable Functions | LangGraph Platform (paid) |

LangGraph Platform (hosted) moves closer to the "human as node" ideal by
managing the re-invocation via webhooks. The open-source library leaves that
plumbing to you.

---

## LangGraph Limitations

### Reliability

| Issue | What happens | Mitigation |
|---|---|---|
| Node crashes (exception) | Checkpoint preserved from before the node; manual retry re-runs the node | Implement idempotent nodes; wrap invoke() in try/except |
| Node hangs | Process hangs indefinitely | Add timeouts yourself (httpx timeout, asyncio.wait_for, etc.) |
| Non-idempotent node retried | Side effect runs twice (email sent twice, record inserted twice) | Design nodes to be idempotent; use deduplication keys |
| Process killed mid-node | Checkpoint from before the node is intact; re-run on next invoke | Same as crash recovery |

LangGraph has **no built-in timeout, watchdog, or cancellation mechanism**.

### Scalability

- Graphs run synchronously within a single process by default
- No built-in task queue or worker pool
- No horizontal scaling of individual nodes
- High-volume pipelines require wrapping in your own async/queue infrastructure

### API design

- `{"configurable": {"thread_id": "..."}}` is a convention enforced by prose,
  not types — typos fail silently
- `thread_id` is chatbot-biased naming for a general-purpose isolation mechanism
- `compile()` injects hidden load/save behavior — explicitness disappears once
  you add a checkpointer
- `interrupt()` leaks the human-in-the-loop mechanism to the caller rather than
  encapsulating it

### What LangGraph open-source is good for

- Learning agentic patterns (nodes, edges, state, routing)
- Prototyping and internal tools
- Low-stakes automation where occasional failures are acceptable
- Workflows where you own the surrounding infrastructure

### What it is not, without LangGraph Platform or wrapping infrastructure

- A durable execution engine
- A production-grade workflow orchestrator for enterprise/regulated use cases

### The honest comparison

| Requirement | LangGraph OSS | LangGraph Platform | Temporal / Step Functions |
|---|---|---|---|
| Crash recovery | Partial (checkpoint before node) | Yes | Yes |
| Hang detection / timeout | No | Yes | Yes |
| Idempotent retries | Your problem | Managed | Managed |
| Human-in-the-loop | Leaks to caller | Webhooks, managed | Native human tasks |
| Audit trail | Basic checkpoint history | Yes | Yes |
| Horizontal scaling | No | Yes | Yes |
| Cost | Free | Paid SaaS | Infra + complexity |

For serious production workflows, the pragmatic pattern is: use a proven
execution engine (Temporal, AWS Step Functions, Airflow) for durability and
reliability, and call LLMs from within that — not the other way around.
LangGraph is one component in that stack, not the foundation of it.
