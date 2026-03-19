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
