import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    http_client=httpx.Client(verify=False)
)

# Define the prompt structure
# SystemMessage sets the chatbot's personality/role
# HumanMessage is the user's input
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions about {topic}. Keep answers concise."),
    ("human", "{question}"),
])

# Chain the prompt and the model together using LCEL (|)
chain = prompt | llm

# Invoke the chain with variable values
response = chain.invoke({
    "topic": "space exploration",
    "question": "What was the first animal in space?"
})

print(response.content)
