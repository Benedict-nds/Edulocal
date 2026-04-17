# EduLocal Assistant — Streamlit UI: note-grounded Q&A with optional file uploads.

from __future__ import annotations

from typing import Any

import streamlit as st
from dotenv import load_dotenv
from openai import APIError, OpenAIError

from utils.ai_client import api_key_configured, chat_completion
from utils.file_parser import extract_text_from_bytes
from utils.grounding import disclaimer_text, top_keyword_excerpts, weak_question_note_match
from utils.prompt_builder import ResponseMode, build_messages, ensure_twi_output_structure

load_dotenv()

# -------------------------
# UI CONSTANTS
# -------------------------

# Human-readable labels mapped to internal mode keys used by prompt_builder.
MODE_OPTIONS: list[tuple[str, ResponseMode]] = [
    ("Normal Answer", "normal"),
    ("Simple Explanation", "simple"),
    ("Summary", "summary"),
    ("Twi-Supported Explanation", "twi"),
]

# Streamlit session_state keys: widgets bind to notes/question; upload_id resets file input.
SESSION_NOTES = "lecture_notes"
SESSION_QUESTION = "study_question"
SESSION_UPLOAD_ID = "upload_id"
SESSION_RESPONSE = "response_bundle"


# Ensure session_state exists so text widgets and upload reset behave predictably on first load.
def _init_session() -> None:
    if SESSION_NOTES not in st.session_state:
        st.session_state[SESSION_NOTES] = ""
    if SESSION_QUESTION not in st.session_state:
        st.session_state[SESSION_QUESTION] = ""
    if SESSION_UPLOAD_ID not in st.session_state:
        st.session_state[SESSION_UPLOAD_ID] = 0


# Merge pasted notes with decoded upload text; return one string for the model plus per-file warnings.
def _combine_lecture_notes(
    pasted_notes: str,
    uploaded_files: list[Any] | None,
) -> tuple[str, list[str]]:
    chunks: list[str] = []
    warnings: list[str] = []
    pasted = (pasted_notes or "").strip()
    if pasted:
        chunks.append(pasted)

    for f in uploaded_files or []:
        try:
            raw = f.getvalue()
            text = extract_text_from_bytes(f.name, raw)
            if text:
                # File marker helps the model see which passage came from which upload.
                chunks.append(f"### From file: {f.name}\n{text}")
            else:
                warnings.append(f"No extractable text from “{f.name}”.")
        except Exception as exc:  # noqa: BLE001 — per-file failure should not crash the app
            warnings.append(f"Could not read “{f.name}”: {exc}")

    combined = "\n\n".join(chunks).strip()
    return combined, warnings


# Return a user-facing error string if notes, question, or API key are missing; else None.
def _validation_message(notes: str, question: str) -> str | None:
    if not notes.strip():
        return "Add lecture notes: paste text and/or upload a .txt or .pdf file."
    if not question.strip():
        return "Enter a study question."
    if not api_key_configured():
        return "Missing OpenAI API key. Set OPENAI_API_KEY in your .env file."
    return None


# Reset inputs and bump upload_id so Streamlit creates a fresh file_uploader widget (clears files).
def _clear_all() -> None:
    st.session_state[SESSION_NOTES] = ""
    st.session_state[SESSION_QUESTION] = ""
    st.session_state[SESSION_UPLOAD_ID] = int(st.session_state[SESSION_UPLOAD_ID]) + 1
    st.session_state.pop(SESSION_RESPONSE, None)


# Draw the last successful response: answer, optional keyword-based excerpts, and disclaimer caption.
def _render_response_block(bundle: dict[str, Any]) -> None:
    st.divider()
    st.subheader("Response")
    with st.container(border=True):
        answer = bundle.get("answer") or ""
        if answer.strip():
            st.markdown(answer)
        else:
            st.info("The model returned an empty response. Try again or shorten the notes.")

    excerpts: list[str] = bundle.get("excerpts") or []
    if excerpts:
        st.markdown("##### Grounding / Source Excerpts")
        st.caption("Short passages from your notes that best match your question (keyword overlap).")
        for i, ex in enumerate(excerpts, start=1):
            with st.expander(f"Excerpt {i}", expanded=(i == 1)):
                st.markdown(f"> {ex}")

    st.caption(bundle.get("disclaimer", disclaimer_text(False)))


def main() -> None:
    st.set_page_config(page_title="EduLocal Assistant", page_icon="📚", layout="centered")
    _init_session()

    # -------------------------
    # INPUT HANDLING
    # -------------------------

    st.title("EduLocal Assistant")
    st.markdown(
        "Study companion for **Ghanaian university** learners—ask questions grounded in "
        "**your** lecture notes. Responses follow what you provide; if the notes are not "
        "enough, the assistant will say so."
    )

    st.text_area(
        "Lecture notes",
        height=220,
        placeholder="Paste your lecture notes here…",
        key=SESSION_NOTES,
    )

    # Dynamic key ties to upload_id so "Clear All" forces a new uploader instance (files cleared).
    uploads = st.file_uploader(
        "Or upload notes (.txt or .pdf)",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        key=f"notes_files_{st.session_state[SESSION_UPLOAD_ID]}",
    )

    if uploads:
        st.markdown("**Uploaded files**")
        for f in uploads:
            st.markdown(f"- `{f.name}`")

    st.text_input(
        "Your study question",
        placeholder="e.g. What is the main idea of week 3?",
        key=SESSION_QUESTION,
    )

    mode_label = st.selectbox(
        "Response style",
        options=[m[0] for m in MODE_OPTIONS],
        index=0,
    )
    mode_by_label: dict[str, ResponseMode] = dict(MODE_OPTIONS)
    mode = mode_by_label[mode_label]

    col_clear, col_gen = st.columns(2)
    with col_clear:
        if st.button("Clear All", use_container_width=True):
            _clear_all()
            st.rerun()
    with col_gen:
        generate = st.button("Generate Response", type="primary", use_container_width=True)

    # -------------------------
    # PROCESSING (generate branch)
    # -------------------------

    if generate:
        pasted = str(st.session_state.get(SESSION_NOTES, ""))
        question = str(st.session_state.get(SESSION_QUESTION, ""))
        notes_text, file_warnings = _combine_lecture_notes(pasted, uploads)
        for w in file_warnings:
            st.warning(w)

        err = _validation_message(notes_text, question)
        if err:
            st.error(err)
        else:
            # Grounding UI: overlap heuristic chooses disclaimer strength; excerpts are not sent to OpenAI.
            weak = weak_question_note_match(notes_text, question)
            excerpts = top_keyword_excerpts(notes_text, question, n=3)
            disc = disclaimer_text(weak)
            messages = build_messages(question, context=notes_text, mode=mode)
            try:
                with st.spinner("Generating…"):
                    answer = chat_completion(messages)
            except RuntimeError as e:
                st.error(str(e))
            except APIError as e:
                st.error(f"OpenAI API error: {getattr(e, 'message', e)}")
            except OpenAIError as e:
                st.error(f"OpenAI error: {e}")
            except Exception as e:  # noqa: BLE001 — last-resort message for unexpected failures
                st.error(f"Something went wrong: {e}")
            else:
                # Twi mode: enforce visible two-part headings if the model skipped them (demo-safe).
                if mode == "twi":
                    answer = ensure_twi_output_structure(answer)
                st.session_state[SESSION_RESPONSE] = {
                    "answer": answer,
                    "excerpts": excerpts,
                    "disclaimer": disc,
                }
                st.rerun()

    # -------------------------
    # OUTPUT (persisted until Clear All or new successful run)
    # -------------------------

    if SESSION_RESPONSE in st.session_state:
        _render_response_block(st.session_state[SESSION_RESPONSE])


if __name__ == "__main__":
    main()
