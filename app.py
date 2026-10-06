"""Hopscotch Support (Session 2) - Streamlit UI on top of LangChain.

Files:  llm.py (model) | prompts.py (all prompts) | chains.py (chains + history) | app.py (this UI)
"""
import html
import json
import os
import re
import time
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder

import chains
from chains import SAMPLE_TICKETS, build_chatbot, run_triage_chain
from llm import AVAILABLE_MODELS, DEFAULT_MODEL, get_llm
from prompts import POLICY, PROMPT_CATALOG, SUPPORT_PROMPT

load_dotenv()
st.set_page_config(page_title="Hopscotch Support (Session 2)", page_icon="🧒", layout="wide")

if not os.getenv("OPENROUTER_API_KEY"):
    st.error("OPENROUTER_API_KEY is not set. Copy `.env.example` to `.env` and add your key, then restart.")
    st.stop()

MODEL_NAME_BY_ID = {v: k for k, v in AVAILABLE_MODELS.items()}


@st.cache_resource
def get_store() -> dict:
    """Session store that survives Streamlit reruns (and is shared by all browser tabs).

    It is the SAME dict object that chains.get_session_history reads/writes.
    """
    return chains.store


store = get_store()

# ---------------------------------------------------------------------------
# Styling: role-coloured cards that work in light AND dark themes
# (semi-transparent backgrounds, solid left border, inherited text colour)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.hs-card{border-left:5px solid var(--c);background:var(--bg);border-radius:6px;padding:10px 14px;margin:8px 0;color:inherit;line-height:1.5;font-size:0.93rem}
.hs-role{font-size:0.72rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--c);margin-bottom:4px}
.hs-system{--c:#8b5cf6;--bg:rgba(139,92,246,.12)}
.hs-human{--c:#0ea5e9;--bg:rgba(14,165,233,.12)}
.hs-ai{--c:#f59e0b;--bg:rgba(245,158,11,.14)}
.hs-ph{border:2px dashed rgba(128,128,128,.6);background:rgba(128,128,128,.10);border-radius:6px;padding:10px 14px;margin:8px 0;color:inherit}
.hs-pill{display:inline-block;padding:0 7px;border-radius:999px;background:rgba(236,72,153,.22);border:1px solid rgba(236,72,153,.7);font-family:monospace;font-size:0.85em}
.hs-badge{display:inline-block;padding:2px 12px;border-radius:999px;font-weight:700;font-size:0.9rem;border:1px solid var(--c);background:var(--bg);color:inherit}
.hs-chip{display:inline-block;padding:1px 9px;margin:2px 3px 2px 0;border-radius:999px;border:1px solid rgba(128,128,128,.6);background:rgba(128,128,128,.12);font-family:monospace;font-size:.85em}
.hs-red{--c:#ef4444;--bg:rgba(239,68,68,.15)}
.hs-green{--c:#22c55e;--bg:rgba(34,197,94,.15)}
.hs-amber{--c:#f59e0b;--bg:rgba(245,158,11,.15)}
.hs-blue{--c:#0ea5e9;--bg:rgba(14,165,233,.15)}
.hs-grey{--c:#9ca3af;--bg:rgba(156,163,175,.15)}
.hs-step{border:1px solid rgba(128,128,128,.4);border-top:4px solid #22c55e;border-radius:6px;padding:8px 12px;color:inherit}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def pills(text: str) -> str:
    """HTML-escape `text`, then turn {variable} into coloured pill chips."""
    safe = html.escape(text).replace("\n", "<br>")
    return re.sub(r"\{(\w+)\}", r'<span class="hs-pill">{\1}</span>', safe)


def role_of(msg_type: str) -> str:
    return {"human": "human", "ai": "ai"}.get(msg_type, "system")


def card(role: str, body_html: str) -> str:
    return f'<div class="hs-card hs-{role}"><div class="hs-role">{role}</div>{body_html}</div>'


def show_template_structure(prompt):
    """Iterate prompt.messages and draw one card per message template."""
    out = []
    for m in prompt.messages:
        if isinstance(m, MessagesPlaceholder):
            out.append(
                f'<div class="hs-ph">📥 <span class="hs-pill">{m.variable_name}</span> - '
                "a LIST of past messages is inserted here at runtime</div>"
            )
        else:
            # SystemMessagePromptTemplate / HumanMessagePromptTemplate / AIMessagePromptTemplate
            role = m.__class__.__name__.replace("MessagePromptTemplate", "").lower()
            role = role if role in ("system", "human", "ai") else "system"
            out.append(card(role, pills(m.prompt.template)))
    st.markdown("".join(out), unsafe_allow_html=True)


def show_messages(messages: list[BaseMessage], key: str, truncate_system: bool = True):
    """Render concrete messages (what the LLM actually receives) as role cards."""
    full = st.toggle("Show full policy / long text", key=f"full_{key}") if truncate_system else True
    out = []
    for m in messages:
        text = m.content
        if not full and m.type == "system" and len(text) > 500:
            text = text[:500] + "\n[... truncated - toggle above to see everything ...]"
        out.append(card(role_of(m.type), html.escape(text).replace("\n", "<br>")))
    st.markdown("".join(out), unsafe_allow_html=True)


def friendly_error(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Session state + sidebar
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8]
if "answered_by" not in st.session_state:
    st.session_state.answered_by = {}  # (session_id, message_index) -> model id
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = {}  # session_id -> messages sent last turn


def new_session():
    st.session_state.session_id = uuid.uuid4().hex[:8]


def switch_session():
    st.session_state.session_id = st.session_state.session_picker


with st.sidebar:
    st.header("🧒 Hopscotch Support")
    st.subheader("Model")
    model_label = st.selectbox("Model (via OpenRouter)", list(AVAILABLE_MODELS), index=list(AVAILABLE_MODELS.values()).index(DEFAULT_MODEL))
    model = AVAILABLE_MODELS[model_label]
    st.info("Switching provider = changing ONE string - the prompt, history and chain code don't change.")
    st.code("llm = get_llm(model)", language="python")

    st.subheader("Session")
    sid = st.session_state.session_id
    st.write(f"Current session_id: `{sid}`")
    st.button("➕ New session", on_click=new_session)
    options = list(store.keys())
    if sid not in options:
        options.append(sid)
    st.selectbox("Switch session", options, index=options.index(sid), key="session_picker", on_change=switch_session,
                 help="Each session_id has its own isolated history - like separate customers. (Demo only: the store is shared by everyone using this server, so do not use this switcher in a real deployment.)")

sid = st.session_state.session_id

try:
    llm = get_llm(model)
except Exception as e:  # e.g. missing key
    st.error(friendly_error(e))
    st.stop()

tab_chat, tab_prompts, tab_triage = st.tabs(["💬 Chat", "🧩 Prompt Templates", "🎫 Ticket triage (chains)"])

# ===========================================================================
# TAB 1: Chat
# ===========================================================================
with tab_chat:
    st.caption(f"Session `{sid}` · model: {model_label}")
    history_box = st.container()
    user_msg = st.chat_input("Ask about a return, refund or defect...")

    if user_msg:
        history_obj = chains.get_session_history(sid)
        before = list(history_obj.messages)
        # Exactly what the prompt template will produce for this turn:
        st.session_state.last_prompt[sid] = SUPPORT_PROMPT.invoke(
            {"chat_history": before, "input": user_msg}
        ).to_messages()
        try:
            with st.spinner("Thinking..."):
                build_chatbot(llm).invoke(
                    {"input": user_msg}, config={"configurable": {"session_id": sid}}
                )
            # the AI reply is the last message in the store
            st.session_state.answered_by[(sid, len(history_obj.messages) - 1)] = model
        except Exception as e:
            st.error(f"LLM call failed - {friendly_error(e)}")

    with history_box:
        msgs = store.get(sid).messages if sid in store else []
        if not msgs:
            st.info("No messages yet in this session. Try: *My son's sneaker sole peeled off after 3 wears.*")
        for i, m in enumerate(msgs):
            with st.chat_message("user" if isinstance(m, HumanMessage) else "assistant"):
                st.markdown(m.content)
                if isinstance(m, AIMessage):
                    who = st.session_state.answered_by.get((sid, i))
                    if who:
                        st.caption(f"answered by {MODEL_NAME_BY_ID.get(who, who)}")

    with st.expander("🔍 Prompt sent to the model this turn"):
        last = st.session_state.last_prompt.get(sid)
        if last:
            st.caption("= SUPPORT_PROMPT.invoke({'chat_history': <history before this turn>, 'input': <message>}).to_messages()")
            show_messages(last, key="chat_prompt")
        else:
            st.write("Send a message first.")

    with st.expander("🗂️ Stored history objects"):
        for m in (store[sid].messages if sid in store else []):
            st.markdown(f"`{type(m).__name__}` - {m.content[:200]}{'...' if len(m.content) > 200 else ''}")
        if sid not in store:
            st.write("Nothing stored yet for this session.")

# ===========================================================================
# TAB 2: Prompt Templates showcase
# ===========================================================================
SESSION1_CODE = '''# Session 1: f-string + raw dicts
SYSTEM_PROMPT = f"""You are a Hopscotch support assistant.
Use only this policy:
{POLICY}
"""
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
messages += st.session_state.messages          # list of dicts
messages.append({"role": "user", "content": user_text})
client.chat.completions.create(model=MODEL, messages=messages)'''

SESSION2_CODE = '''# Session 2: ChatPromptTemplate
SUPPORT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are ... {policy} ... rules ..."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
]).partial(policy=POLICY)

SUPPORT_PROMPT.invoke({"chat_history": history, "input": user_text})'''


def example_values(name: str, prompt) -> dict:
    """Sensible default text for each input variable."""
    ticket = next(iter(SAMPLE_TICKETS.values()))
    examples = {
        "ticket_text": ticket,
        "input": "Hi, my son's sneakers (INR 2,499) have a peeling sole after 3 wears. Delivered 2 weeks ago.",
        "category": "genuine_defect",
        "details": '{"customer_name": "Ananya", "order_id": "HS88213", "product": "Dino-Print Sneakers", '
                   '"issue": "sole peeling", "days_since_delivery": 14, "order_value_inr": 2499}',
    }
    return {v: examples.get(v, "") for v in prompt.input_variables if v != "chat_history"}


def parse_demo_history(text: str) -> list[BaseMessage]:
    out = []
    for line in text.splitlines():
        if line.lower().startswith("human:"):
            out.append(HumanMessage(content=line[6:].strip()))
        elif line.lower().startswith("ai:"):
            out.append(AIMessage(content=line[3:].strip()))
    return out


with tab_prompts:
    name = st.selectbox("Choose a prompt", list(PROMPT_CATALOG), key="prompt_choice")
    entry = PROMPT_CATALOG[name]
    prompt = entry["prompt"]
    st.write(entry["description"])
    st.caption(f"Used in: {entry['used_in']}")

    st.subheader("1 · Structure")
    c1, c2, c3 = st.columns([3, 3, 1])
    c1.markdown("**Input variables**<br>" + "".join(f'<span class="hs-chip">{v}</span>' for v in prompt.input_variables), unsafe_allow_html=True)
    partial_keys = list(prompt.partial_variables)
    c2.markdown("**Pre-filled (partial)**<br>" + ("".join(f'<span class="hs-chip">{v}</span>' for v in partial_keys) or "none"), unsafe_allow_html=True)
    c3.metric("Messages", len(prompt.messages))
    show_template_structure(prompt)

    st.subheader("2 · Fill & render")
    st.caption("Fill the variables, then see exactly what the LLM would receive - no LLM call is made.")
    values: dict = {}
    for var, default in example_values(name, prompt).items():
        values[var] = st.text_area(var, default, key=f"val_{name}_{var}", height=100)
    if "chat_history" in prompt.input_variables:
        demo = st.text_area(
            "chat_history (demo: one `human:` / `ai:` line per message)",
            "human: Hi, I want to return a jacket.\nai: Happy to help! When was it delivered, and is it unworn with tags?",
            key=f"hist_{name}", height=100,
        )
        values["chat_history"] = parse_demo_history(demo)

    if st.button("▶️ Render", key=f"render_{name}"):
        try:
            rendered = prompt.invoke(values).to_messages()
            st.success("This is exactly what the LLM receives:")
            show_messages(rendered, key=f"render_{name}")
        except Exception as e:
            st.error(friendly_error(e))

    st.subheader("3 · Before vs after")
    b1, b2 = st.columns(2)
    b1.markdown("**Session 1 - f-string**")
    b1.code(SESSION1_CODE, language="python")
    b2.markdown("**Session 2 - ChatPromptTemplate**")
    b2.code(SESSION2_CODE, language="python")
    st.markdown(
        "- **Reuse and versioning**: a prompt is a named object in `prompts.py` you can import, diff and test.\n"
        "- **Validation**: missing variables fail loudly *before* any LLM call is paid for.\n"
        "- **Provider-agnostic**: output is typed messages, not provider-specific dicts, so any chat model accepts it."
    )
    if st.button("🧪 Demo: invoke with a missing variable", key=f"missing_{name}"):
        try:
            prompt.invoke({})
            st.warning("No error raised (prompt has no required variables).")
        except Exception as e:
            st.error(f"Caught {type(e).__name__}: {e}")

    st.subheader("4 · Same template, different models")
    m1, m2 = st.columns(2)
    labels = list(AVAILABLE_MODELS)
    left = m1.selectbox("Model A", labels, index=0, key="cmp_a")
    right = m2.selectbox("Model B", labels, index=1, key="cmp_b")
    if st.button("⚖️ Compare", key=f"compare_{name}"):
        for col, label in ((m1, left), (m2, right)):
            with col:
                try:
                    chain = prompt | get_llm(AVAILABLE_MODELS[label]) | StrOutputParser()
                    t0 = time.time()
                    with st.spinner(label):
                        out = chain.invoke(values)
                    st.markdown(f"**{label}** · {time.time() - t0:.1f}s")
                    st.write(out)
                except Exception as e:
                    st.error(f"{label}: {friendly_error(e)}")

# ===========================================================================
# TAB 3: Ticket triage
# ===========================================================================
CATEGORY_COLORS = {"policy_violation": "amber", "genuine_defect": "red", "other": "grey"}
PRIORITY_COLORS = {"high": "red", "medium": "amber", "low": "green"}


def badge(text: str, color: str) -> str:
    return f'<span class="hs-badge hs-{color}">{html.escape(text)}</span>'


def try_parse_json(text: str):
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(cleaned)
    except ValueError:
        return None


with tab_triage:
    st.markdown("Raw ticket in → category, details, priority and a draft reply out.")
    choice = st.selectbox("Sample ticket", list(SAMPLE_TICKETS))
    ticket = st.text_area("Ticket text (editable)", SAMPLE_TICKETS[choice], height=140, key=f"ticket_{choice}")
    st.code("classification_prompt | llm | StrOutputParser()", language="python")
    st.caption("4 sequential LLM calls in plain Python - Session 3 makes this declarative + parallel.")

    if st.button("Run 4-step chain", type="primary"):
        try:
            with st.spinner("Running 4 LLM calls..."):
                st.session_state.triage = (choice, run_triage_chain(ticket, llm))
        except Exception as e:
            st.error(f"Triage failed - {friendly_error(e)}")

    if "triage" in st.session_state:
        res = st.session_state.triage[1]
        s1, s2, s3, s4 = st.columns(4)
        cat = res["category"].strip().lower()
        pri = res["priority"].strip().lower()
        with s1:
            st.markdown('<div class="hs-step"><b>1 · Classification</b></div>', unsafe_allow_html=True)
            st.markdown(badge(res["category"], CATEGORY_COLORS.get(cat, "grey")), unsafe_allow_html=True)
        with s2:
            st.markdown('<div class="hs-step"><b>2 · Extraction</b></div>', unsafe_allow_html=True)
            parsed = try_parse_json(res["details"])
            if parsed is not None:
                st.json(parsed)
            else:
                st.text(res["details"])
        with s3:
            st.markdown('<div class="hs-step"><b>3 · Priority</b></div>', unsafe_allow_html=True)
            st.markdown(badge(res["priority"], PRIORITY_COLORS.get(pri, "grey")), unsafe_allow_html=True)
        with s4:
            st.markdown('<div class="hs-step"><b>4 · Draft reply</b></div>', unsafe_allow_html=True)
            st.write(res["draft_reply"])
