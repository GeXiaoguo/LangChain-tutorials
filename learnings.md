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
