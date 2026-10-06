"""prompts.py - ALL prompts live here as documented, module-level ChatPromptTemplates.

Session 1 pasted the policy into an f-string. Now each prompt is a reusable object
with declared input variables, so it can be inspected, validated, versioned and
rendered without calling any LLM.
"""
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# The policy text, loaded once from the markdown file next to this module.
POLICY = (Path(__file__).parent / "hopscotch_policy.md").read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# 1. Support chatbot prompt: system + chat history slot + new human message
# ---------------------------------------------------------------------------
SUPPORT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a customer support assistant for Hopscotch, a premium kids' fashion "
        "retailer based in Mumbai. Be warm, concise and solution-oriented.\n\n"
        "Use ONLY the return policy below to answer.\n\n"
        "=== RETURN POLICY ===\n{policy}\n=== END POLICY ===\n\n"
        "Rules:\n"
        "1. Answer from the policy. Never invent rules, deadlines or refunds.\n"
        "2. If you are missing the delivery date, the item's condition (worn / washed / tags) "
        "or the order value, ask the customer for it before giving an answer.\n"
        "3. For anything in Section 4 (child-safety concern, skin reaction or allergy, order "
        "worth more than INR 5,000, or unclear / conflicting facts) NEVER decide the outcome. "
        "Say that a human agent will review the case.\n"
        "4. For a suspected manufacturing defect, ask the customer to share photos of the "
        "problem before a replacement or refund can be arranged."
    )),
    MessagesPlaceholder("chat_history"),  # past turns (a LIST of messages) go here
    ("human", "{input}"),
]).partial(policy=POLICY)  # pre-fill {policy} so callers only supply chat_history + input

# ---------------------------------------------------------------------------
# 2. Ticket triage prompts
# ---------------------------------------------------------------------------
CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Classify the customer's return/complaint message into exactly one category:\n"
        "- 'policy_violation': a standard return (change of mind, wrong size, colour) that the "
        "policy does not allow - e.g. outside the 30-day window, or worn/washed/tags removed - "
        "or damage from misuse, normal wear and tear, or ignoring the care label.\n"
        "- 'genuine_defect': a manufacturing defect reported within 90 days of delivery "
        "(seams or stitching failing, sole peeling under normal use, zip/button failing, colour "
        "bleeding when washed as instructed, print cracking), even if worn or washed per the care label.\n"
        "- 'other': anything else (delivery delay, billing question, general query).\n"
        "Respond with ONLY the category label, nothing else."
    )),
    ("human", "{ticket_text}"),
])

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Extract these fields from the customer message as a compact JSON object: "
        "customer_name, order_id, product, issue (one-sentence summary), "
        "days_since_delivery (integer; convert weeks/months to days), "
        "order_value_inr (number, no currency symbol). "
        "Use null for any field that is not mentioned. "
        "Respond with ONLY valid JSON, no markdown fences, no extra text."
    )),
    ("human", "{ticket_text}"),
])

PRIORITY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Given a ticket's category and extracted details, assign a priority: "
        "'High', 'Medium' or 'Low'.\n"
        "High = child-safety concern, skin reaction/allergy, or a genuine_defect on an order "
        "above INR 5,000.\n"
        "Medium = routine genuine_defect, or an 'other' ticket that needs follow-up.\n"
        "Low = policy_violation with no safety concern.\n"
        "Respond with ONLY the priority label."
    )),
    ("human", "Category: {category}\nDetails: {details}"),
])

RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are drafting a reply on behalf of Hopscotch customer support. You get the ticket "
        "category and extracted details as context. Write a warm reply of 3-4 sentences that "
        "acknowledges the issue and proposes a next step according to the policy below: "
        "genuine_defect -> ask for photos, then replacement/refund; policy_violation -> politely "
        "explain the policy; other -> route to the right team. If the case falls under Section 4 "
        "(child safety, allergy, order above INR 5,000, unclear facts) do NOT promise an outcome - "
        "say a human agent will review it.\n\n"
        "=== RETURN POLICY ===\n{policy}\n=== END POLICY ==="
    )),
    ("human", "Category: {category}\nDetails: {details}\nOriginal message: {ticket_text}"),
]).partial(policy=POLICY)

# ---------------------------------------------------------------------------
# Catalog used by the "Prompt Templates" tab in app.py
# ---------------------------------------------------------------------------
PROMPT_CATALOG: dict[str, dict] = {
    "Support chatbot": {
        "prompt": SUPPORT_PROMPT,
        "description": "System rules + the full return policy, then the running chat history, then the customer's new message.",
        "used_in": "Chat tab - build_chatbot() in chains.py",
    },
    "Ticket classification": {
        "prompt": CLASSIFICATION_PROMPT,
        "description": "Labels a ticket as policy_violation, genuine_defect or other using the 30/90-day rules.",
        "used_in": "Triage step 1 - run_triage_chain() in chains.py",
    },
    "Detail extraction": {
        "prompt": EXTRACTION_PROMPT,
        "description": "Pulls customer, order, product, issue, days since delivery and order value out of free text as JSON.",
        "used_in": "Triage step 2 - run_triage_chain() in chains.py",
    },
    "Priority scoring": {
        "prompt": PRIORITY_PROMPT,
        "description": "Turns category + extracted details into High / Medium / Low.",
        "used_in": "Triage step 3 - run_triage_chain() in chains.py",
    },
    "Reply drafting": {
        "prompt": RESPONSE_PROMPT,
        "description": "Writes a warm 3-4 sentence draft reply, grounded in the policy (partial variable).",
        "used_in": "Triage step 4 - run_triage_chain() in chains.py",
    },
}
