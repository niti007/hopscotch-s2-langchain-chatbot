"""chains.py - wires prompts + models + history together.

Chain = steps joined with `|`: the output of one step is the input of the next.
"""
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

from prompts import (
    CLASSIFICATION_PROMPT,
    EXTRACTION_PROMPT,
    PRIORITY_PROMPT,
    RESPONSE_PROMPT,
    SUPPORT_PROMPT,
)

# ---------------------------------------------------------------------------
# Chat history: session_id -> InMemoryChatMessageHistory
# (production: Redis / Postgres behind the same interface)
# ---------------------------------------------------------------------------
store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Look up (or create) the history for one customer session.

    Note: this uses the module-level `store`. Streamlit re-runs the script on every
    interaction, but modules are imported once, so `store` survives reruns; app.py
    also exposes it via @st.cache_resource to make that explicit.
    """
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def build_chatbot(llm):
    """prompt | llm | parser, wrapped so history is loaded/saved per session_id."""
    chain = SUPPORT_PROMPT | llm | StrOutputParser()
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",       # which input key is the new human message
        history_messages_key="chat_history",  # which prompt slot receives past messages
    )


def build_triage_chains(llm) -> dict:
    """The four small chains used by ticket triage."""
    parser = StrOutputParser()
    return {
        "classification": CLASSIFICATION_PROMPT | llm | parser,
        "extraction": EXTRACTION_PROMPT | llm | parser,
        "priority": PRIORITY_PROMPT | llm | parser,
        "response": RESPONSE_PROMPT | llm | parser,
    }


def run_triage_chain(ticket_text: str, llm) -> dict:
    """Run the 4 triage steps IMPERATIVELY: plain Python, one LLM call after another.

    Session 3 rewrites this declaratively with LCEL (RunnableParallel, RunnableBranch,
    streaming) - compare the two!
    """
    chains = build_triage_chains(llm)

    # Step 1: classify
    category = chains["classification"].invoke({"ticket_text": ticket_text}).strip()
    # Step 2: extract structured details (a JSON string)
    details = chains["extraction"].invoke({"ticket_text": ticket_text}).strip()
    # Step 3: priority, needs steps 1 + 2
    priority = chains["priority"].invoke({"category": category, "details": details}).strip()
    # Step 4: draft reply, needs steps 1 + 2
    draft_reply = chains["response"].invoke(
        {"category": category, "details": details, "ticket_text": ticket_text}
    ).strip()

    return {
        "category": category,
        "priority": priority,
        "details": details,
        "draft_reply": draft_reply,
    }


SAMPLE_TICKETS = {
    "Ananya - peeling sneaker sole (defect)": (
        "Hi, I'm Ananya. I bought the Hopscotch Dino-Print Sneakers (Order #HS88213) for my son "
        "two weeks ago. The sole started peeling off after just three wears, even though he only "
        "wore them to school. This seems like a manufacturing issue. Can I get a replacement pair? "
        "The order was INR 2,499."
    ),
    "Winter jacket - worn, wrong size (policy)": (
        "I ordered the wrong size for the Hopscotch Winter Jacket last month and my daughter has "
        "already worn it a few times outdoors. It's a bit snug now - can I return it for a bigger "
        "size? Order #HS77102, delivered 40 days ago, INR 3,200."
    ),
    "Loose button - child safety": (
        "Hello, this is Rohan. A button on my 2-year-old's romper (Order #HS90455) came off after "
        "one wash and she put it in her mouth. Thankfully she was fine but I'm worried it's a "
        "choking hazard. Delivered 10 days ago. INR 1,899."
    ),
    "Rs 7,200 order - zip failure": (
        "Hi, I'm Meera. The zip on the Hopscotch Party Dress set (Order #HS91777, INR 7,200) broke "
        "the second time my daughter wore it. It was delivered 25 days ago. I would like a refund."
    ),
}
