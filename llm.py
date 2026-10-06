"""llm.py - the ONE place that knows which model / provider we talk to.

Session 1 hardcoded the OpenAI SDK + model name inside the app. Here, every other
file (prompts, chains, app) just calls `get_llm(model)` and gets back a LangChain
chat model. Swapping provider = changing this file only.
"""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()  # reads OPENROUTER_API_KEY from .env into os.environ

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"

# Display name -> OpenRouter model id (format: provider/model)
AVAILABLE_MODELS = {
    "OpenAI · GPT-4o mini": "openai/gpt-4o-mini",
    "Anthropic · Claude Haiku 4.5": "anthropic/claude-haiku-4.5",
    "Google · Gemini 2.5 Flash Lite": "google/gemini-2.5-flash-lite",
    "Meta · Llama 3.3 70B": "meta-llama/llama-3.3-70b-instruct",
}


def get_llm(model: str = DEFAULT_MODEL, temperature: float = 0.3) -> ChatOpenAI:
    """Return a LangChain chat model for `model`, served through OpenRouter.

    OpenRouter speaks the OpenAI wire format, so ChatOpenAI works for ALL the
    models above - we only change the `model` string.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found. Copy .env.example to .env and add your key."
        )

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,  # redirects the OpenAI client to OpenRouter
        temperature=temperature,       # low = consistent, policy-like answers
    )

    # ------------------------------------------------------------------
    # Want to call a provider DIRECTLY instead of via OpenRouter? Only THIS
    # function changes - prompts, history and chains stay exactly the same,
    # because they are written against LangChain's interface, not a provider's.
    #
    # from langchain_anthropic import ChatAnthropic
    # return ChatAnthropic(model="claude-haiku-4-5", temperature=temperature)
    #
    # from langchain_google_genai import ChatGoogleGenerativeAI
    # return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=temperature)
    # ------------------------------------------------------------------
