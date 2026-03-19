import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    http_client=httpx.Client(verify=False)
)

# Memory is just a list of messages — the full conversation history
history = [
    SystemMessage(content="You are a helpful assistant that answers questions about space exploration. Keep answers concise.")
]

# Turn 1
history.append(HumanMessage(content="What was the first animal in space?"))
response = llm.invoke(history)
history.append(AIMessage(content=response.content))  # save the response to history
print(f"User:      What was the first animal in space?")
print(f"Assistant: {response.content}\n")

# Turn 2 — the LLM can refer back to turn 1
history.append(HumanMessage(content="How long did it survive?"))
response = llm.invoke(history)
history.append(AIMessage(content=response.content))
print(f"User:      How long did it survive?")
print(f"Assistant: {response.content}\n")

# Turn 3
history.append(HumanMessage(content="What country sent it?"))
response = llm.invoke(history)
history.append(AIMessage(content=response.content))
print(f"User:      What country sent it?")
print(f"Assistant: {response.content}\n")
