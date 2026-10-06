# Hopscotch Support Chatbot - Session 2: LangChain

> **The series**
>
> | # | Repo | Topic |
> |---|------|-------|
> | 1 | [hopscotch-s1-openai-chatbot](https://github.com/niti007/hopscotch-s1-openai-chatbot) | Raw `openai` SDK + Streamlit |
> | **2** | **[hopscotch-s2-langchain-chatbot](https://github.com/niti007/hopscotch-s2-langchain-chatbot)  <- you are here** | Chat models, messages, prompt templates, chains, chat history |
> | 3 | [hopscotch-s3-lcel-chatbot](https://github.com/niti007/hopscotch-s3-lcel-chatbot) | Runnables, LCEL, parallel + branching, streaming |

A return-policy support assistant for **Hopscotch**, a premium kids' fashion retailer in Mumbai. The policy lives in `hopscotch_policy.md` (30-day standard returns, 90-day manufacturing defects, and a Section 4 list of cases a human must always review).

## Where we left off (Session 1)

Session 1 was a single-file Streamlit app calling the raw `openai` SDK (pointed at OpenRouter) with `MODEL="openai/gpt-4o-mini"` hardcoded, the policy pasted into an f-string, and history as a list of `{"role", "content"}` dicts in `st.session_state`.

**The motivating problem:** *what if tomorrow we want another model or provider?* With the raw SDK, that means rewriting the client, the message format and every call site. Session 2 introduces **LangChain** so the model is one swappable object and everything else is written against LangChain's interfaces.

## Concepts

| Concept | What it is | Where |
|---|---|---|
| Chat model | `ChatOpenAI` pointed at OpenRouter; swap one string to change model | `llm.py` |
| Messages | Typed `SystemMessage` / `HumanMessage` / `AIMessage` instead of dicts | `app.py`, history store |
| `ChatPromptTemplate` | A prompt with declared `{variables}`, filled at call time | `prompts.py` |
| `MessagesPlaceholder` | A slot that receives a *list* of past messages (the chat history) | `SUPPORT_PROMPT` |
| `.partial(...)` | Pre-fill some variables (here `policy`) so callers supply fewer | `prompts.py` |
| Chain `\|` | `prompt \| llm \| parser`: output of one step feeds the next | `chains.py` |
| `StrOutputParser` | Turns the model's `AIMessage` into a plain string | `chains.py` |
| `InMemoryChatMessageHistory` | Stores one conversation's messages | `chains.py` |
| `RunnableWithMessageHistory` + `session_id` | Loads/saves history automatically per session | `build_chatbot()` |

## Before / after vs Session 1

| | Session 1 | Session 2 |
|---|---|---|
| Prompt | f-string with policy pasted in | `ChatPromptTemplate` + `.partial(policy=POLICY)` |
| History | list of dicts in `st.session_state` | `InMemoryChatMessageHistory` per `session_id` |
| Multi-user | one list per browser tab | one history per `session_id` (try the session switcher) |
| Model switching | rewrite SDK calls | change one string: `get_llm(model)` |
| Call site | build messages by hand, append reply by hand | `chatbot.invoke({"input": ...}, config={"configurable": {"session_id": sid}})` |

## Project structure

```
app.py                 Streamlit UI (3 tabs + sidebar)
llm.py                 get_llm(), AVAILABLE_MODELS - the only provider-aware file
prompts.py             ALL prompts as ChatPromptTemplates + PROMPT_CATALOG
chains.py              history store, build_chatbot, triage chains, SAMPLE_TICKETS
hopscotch_policy.md    the return policy (loaded into prompts.POLICY)
requirements.txt
.env.example
```

## Setup

Python 3.10+ (3.11 recommended). You need an [OpenRouter](https://openrouter.ai) API key.

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt
# pip alternative:  python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

cp .env.example .env        # then paste your OPENROUTER_API_KEY into .env
```

## Run

```bash
.venv/bin/streamlit run app.py     # or: streamlit run app.py with the venv activated
```

## Teaching flow (2h)

| Time | Topic | Where in the app |
|---|---|---|
| 0:00-0:15 | Recap Session 1, the "switch model tomorrow" problem | Sidebar model picker, `llm.py` |
| 0:15-0:35 | Chat models and Message types | `llm.py`, Chat tab "Stored history objects" |
| 0:35-1:05 | `ChatPromptTemplate`, `MessagesPlaceholder`, `.partial`, validation, same template on two models | **Prompt Templates tab** |
| 1:05-1:30 | Chat history and `session_id`; two customers, isolated | Chat tab + session switcher, "Prompt sent this turn" |
| 1:30-1:55 | Chains with `\|`: 4-step ticket triage | **Ticket triage tab**, `run_triage_chain` |
| 1:55-2:00 | Limitations, teaser for Session 3 | below |

Note: recent LangChain versions print a deprecation warning for `RunnableWithMessageHistory` / `InMemoryChatMessageHistory` (LangGraph persistence is the long-term replacement). They still work and are the clearest way to teach `session_id`-scoped history.

## Limitations -> why Session 3

- Triage is **imperative Python**: four sequential LLM calls, one after another.
- **No streaming**: the user waits for the full reply.
- Classification and extraction are independent but are **not run in parallel**.
- **No declarative routing**: Section 4 human-review cases (child safety, allergy, orders over INR 5,000, unclear facts) should branch to a human, but there is no `RunnableBranch` yet.
- The calling code is **not reusable or batchable** (`.batch()`, `.stream()`, async) because it is a hand-written function, not a Runnable.

Session 3 rewrites all of this with **Runnables and LCEL**.
