from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.rag_engine import RAGEngine


st.set_page_config(
    page_title="LegaLease - Legal AI Assistant",
    layout="wide",
)

st.markdown(
    """
    <style>
        .hero-card {
            background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
            border: 1px solid #dbe4ff;
            border-radius: 20px;
            padding: 24px 28px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
        }
        .answer-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-left: 6px solid #1d4ed8;
            border-radius: 16px;
            padding: 18px 20px;
            margin: 12px 0 18px 0;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        }
        .summary-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 18px 20px;
            min-height: 140px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
        }
        .summary-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #64748b;
            margin-bottom: 8px;
        }
        .summary-value {
            font-size: 1rem;
            color: #0f172a;
            font-weight: 600;
            line-height: 1.5;
        }
        .risk-card {
            border-radius: 16px;
            padding: 18px 20px;
            margin: 12px 0;
            border: 1px solid transparent;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
        }
        .risk-high {
            background: #fef2f2;
            border-color: #fecaca;
        }
        .risk-medium {
            background: #fff7ed;
            border-color: #fdba74;
        }
        .risk-low {
            background: #f0fdf4;
            border-color: #86efac;
        }
        .risk-badge {
            display: inline-block;
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 0.8rem;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .badge-high {
            background: #dc2626;
            color: white;
        }
        .badge-medium {
            background: #ea580c;
            color: white;
        }
        .badge-low {
            background: #16a34a;
            color: white;
        }
    </style>
    <div class="hero-card">
        <h1 style="margin:0; color:#0f172a;">⚖️ LegaLease - Legal AI Assistant</h1>
        <p style="margin:10px 0 0 0; color:#475569; font-size:1.05rem;">
            Rental-law guidance, lease summarization, and clause risk checks in one clean workspace.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


def render_answer_card(answer: str) -> None:
    """Render a styled answer block."""
    st.markdown(
        f"""
        <div class="answer-card">
            <div style="font-size:0.85rem; text-transform:uppercase; letter-spacing:0.06em; color:#64748b; margin-bottom:8px;">
                Answer
            </div>
            <div style="font-size:1rem; color:#111827; line-height:1.7;">
                {answer}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_card(label: str, value: str) -> None:
    """Render a styled summary field card."""
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-label">{label}</div>
            <div class="summary-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_card(clause: str, risk: str, reason: str) -> None:
    """Render a styled clause-risk card."""
    risk_lower = risk.lower()
    card_class = f"risk-card risk-{risk_lower}"
    badge_class = f"risk-badge badge-{risk_lower}"
    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="{badge_class}">{risk.title()} Risk</div>
            <div style="font-weight:700; color:#0f172a; margin-bottom:8px;">Clause</div>
            <div style="color:#1f2937; line-height:1.7; margin-bottom:14px;">{clause}</div>
            <div style="font-weight:700; color:#0f172a; margin-bottom:6px;">Reason</div>
            <div style="color:#334155; line-height:1.6;">{reason}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

tab_chat, tab_summary, tab_clauses = st.tabs(
    ["💬 Legal Chat", "📄 Lease Summarizer", "⚠️ Clause Checker"]
)

with tab_chat:
    query = st.text_input("Enter your legal query", key="legal_chat_query").strip()

    col1, col2 = st.columns([1, 1])
    with col1:
        ask_clicked = st.button("Ask", type="primary", key="ask_button")
    with col2:
        clear_clicked = st.button("Clear Chat", key="clear_button")

    if clear_clicked:
        st.session_state.chat_history = []

    if ask_clicked:
        if not query:
            st.warning("Please enter a legal query.")
        else:
            with st.spinner("Analyzing legal query..."):
                result = st.session_state.rag_engine.answer_question(query)

            st.session_state.chat_history.insert(
                0,
                {
                    "query": query,
                    "answer": result.get("answer", "No answer available."),
                    "chunks": result.get("retrieved_chunks", []),
                },
            )

    for item in st.session_state.chat_history:
        chunks = item.get("chunks", [])
        st.markdown("### Question")
        st.write(item.get("query", ""))
        render_answer_card(item.get("answer", "No answer available."))
        st.caption(f"Chunks used: {len(chunks)}")

        with st.expander("View Sources"):
            if not chunks:
                st.write("No sources available.")
            else:
                for index, chunk in enumerate(chunks, start=1):
                    source = chunk.get("source", f"Source {index}")
                    text = chunk.get("text", "")[:200]
                    st.markdown(f"**{source}**")
                    st.write(text)
                    st.divider()

with tab_summary:
    lease_text = st.text_area(
        "Paste the lease text to summarize",
        height=240,
        key="lease_summary_input",
    )
    summarize_clicked = st.button("Summarize", type="primary", key="summarize_button")

    if summarize_clicked:
        if not lease_text.strip():
            st.warning("Please enter lease text to summarize.")
        else:
            summary = st.session_state.rag_engine.summarise_document(lease_text)
            col1, col2 = st.columns(2)
            with col1:
                render_summary_card("Rent", summary.get("rent", "Not found"))
                render_summary_card("Notice Period", summary.get("notice_period", "Not found"))
            with col2:
                render_summary_card("Deposit", summary.get("deposit", "Not found"))
                render_summary_card("Maintenance", summary.get("maintenance", "Not found"))

with tab_clauses:
    clause_text = st.text_area(
        "Paste lease clauses or agreement text to analyze",
        height=240,
        key="clause_checker_input",
    )
    analyze_clicked = st.button("Analyze Clauses", type="primary", key="analyze_button")

    if analyze_clicked:
        if not clause_text.strip():
            st.warning("Please enter lease text to analyze.")
        else:
            findings = st.session_state.rag_engine.flag_clauses(clause_text)
            for finding in findings:
                render_risk_card(
                    finding.get("clause", "Clause not available."),
                    finding.get("risk", "low"),
                    finding.get("reason", "No reason provided."),
                )
