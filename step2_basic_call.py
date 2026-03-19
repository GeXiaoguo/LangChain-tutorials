import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    http_client=httpx.Client(verify=False)
)

# Send a message and get a response
response = llm.invoke("What is the capital of France?")

print(response.content)
