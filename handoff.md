# LangChain Q&A Chatbot — Learning Session Handoff

## Goal
Build a Q&A chatbot step by step using LangChain and OpenAI.

## Progress

### Completed
- [x] **Step 1: Setup** — venv created, dependencies installed (`langchain`, `langchain-openai`, `python-dotenv`)
- [x] **Step 2: Basic LLM call** — `step2_basic_call.py` working
- [x] **Step 3: Prompt Templates** — `ChatPromptTemplate` added to `step2_basic_call.py`
- [x] **Step 4: Memory** — manual message history list in `step2_basic_call.py`

### Up Next
- [x] **Step 5: CLI chat loop** — interactive terminal chatbot in `step2_basic_call.py`

## Environment Notes
- Python venv at `venv/` (not committed)
- API key in `.env` (not committed)
- Corporate proxy does TLS inspection — `httpx.Client(verify=False)` used in all scripts to bypass SSL verification

## Key Files
- `.env` — OpenAI API key (`OPENAI_API_KEY`)
- `step2_basic_call.py` — basic LLM call, confirmed working

## How to Resume
1. Activate venv: `venv\Scripts\activate`
2. All 5 steps complete — chatbot is fully working
