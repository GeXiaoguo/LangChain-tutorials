import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    http_client=httpx.Client(verify=False)
)

history = [
    SystemMessage(content="You are a helpful assistant. Keep answers concise.")
]

print("Chatbot ready. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "quit":
        print("Goodbye!")
        break

    if not user_input:
        continue

    history.append(HumanMessage(content=user_input))
    response = llm.invoke(history)
    history.append(AIMessage(content=response.content))

    print(f"Bot: {response.content}\n")
