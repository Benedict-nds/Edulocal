# Assembles OpenAI chat messages: shared grounding rules plus mode-specific system text.

from typing import Literal

ResponseMode = Literal["normal", "simple", "summary", "twi"]

# Twi mode: exact heading lines the model must reproduce (prompts + post-display safeguard).
TWI_HEADING_ENGLISH = "ENGLISH (Simple Explanation):"
TWI_HEADING_TW = "TWI-SUPPORTED EXPLANATION:"

# Shared system prefix for every mode: note-only answers, admit gaps, forbid hallucinations.
_GROUNDING = (
    "You are EduLocal Assistant, a study helper for students at Ghanaian universities. "
    "You MUST base your response ONLY on the lecture notes the user provides. "
    "If the notes do not contain enough information to answer fully, say so clearly "
    "in your own words—explain what is missing—and do NOT invent facts, examples, "
    "or course details that are not supported by the notes."
)

# Mode-specific instructions appended after _GROUNDING (normal/simple/summary/twi).
_MODE_SUFFIX: dict[ResponseMode, str] = {
    "normal": "Give a clear, accurate answer that directly addresses the question.",
    "simple": (
        "Explain in simple language suitable for someone new to the topic. "
        "Use short sentences and define jargon briefly when it appears in the notes."
    ),
    "summary": (
        "Focus on summarizing only the parts of the notes that are relevant to the question. "
        "If the question is broad, summarize the key ideas present in the notes."
    ),
    "twi": (
        "You are answering in TWI-SUPPORTED EXPLANATION mode for Ghanaian university learners.\n"
        "GROUNDING: Base BOTH sections ONLY on the lecture notes. Do not invent facts, examples, "
        "or claims that are not supported by the notes. Do not imply the notes contain "
        "information they do not. If the notes are incomplete for the question, say so clearly "
        "in BOTH sections—do not guess.\n"
        "MANDATORY STRUCTURE — EVERY reply MUST follow exactly this two-part layout, in this order. "
        "Use these two heading lines EXACTLY as written below (same spelling, capitalization, "
        "parentheses, and trailing colon). Put each heading on its own line. Do not omit either "
        "heading. Do not rename them, do not wrap them in bold/markdown, do not number them, and "
        "do not merge both sections into one paragraph—use two clearly separated blocks with a "
        "blank line between them.\n\n"
        f"{TWI_HEADING_ENGLISH}\n"
        "<clear, simple English explanation grounded only in the notes>\n\n"
        f"{TWI_HEADING_TW}\n"
        "<same ideas explained again in learner-friendly Twi-supported style: simple English "
        "mixed with short Twi phrases or glosses where appropriate; native-perfect Twi is NOT "
        "required>\n\n"
        "Do not write any preamble before the first heading. Do not add other section titles."
    ),
}


# Concatenate global grounding with the chosen mode's suffix (unless a custom system is passed in).
def system_prompt_for_mode(mode: ResponseMode) -> str:
    return f"{_GROUNDING} {_MODE_SUFFIX[mode]}"


# If the model omitted Twi section headings, wrap the raw text so the UI still shows two labeled blocks.
def ensure_twi_output_structure(text: str) -> str:
    body = (text or "").strip()
    lower = body.lower()
    # Case-insensitive check so minor capitalization drift still counts as compliant.
    has_en = "english (simple explanation):" in lower
    has_tw = "twi-supported explanation:" in lower
    if has_en and has_tw:
        return body
    return (
        f"{TWI_HEADING_ENGLISH}\n"
        f"{body}\n\n"
        f"{TWI_HEADING_TW}\n"
        "Twi-supported explanation was not formatted correctly. Please regenerate."
    )


# Build [system, optional notes message, user question]; Twi mode appends a formatting reminder to the user turn.
def build_messages(
    user_question: str,
    context: str | None = None,
    *,
    mode: ResponseMode = "normal",
    system_instruction: str | None = None,
) -> list[dict[str, str]]:
    system = system_instruction or system_prompt_for_mode(mode)
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    if context and context.strip():
        messages.append(
            {
                "role": "user",
                "content": f"--- Lecture notes ---\n{context.strip()}\n--- End lecture notes ---",
            }
        )
    user_content = user_question.strip()
    if mode == "twi":
        # Extra nudge on the user message reinforces exact headings (system prompt already mandates them).
        user_content = (
            f"{user_content}\n\n"
            f"Reminder: Start your answer with the exact line {TWI_HEADING_ENGLISH} then a blank "
            f"line after the English section, then the exact line {TWI_HEADING_TW} "
            "for the second section. Do not merge into one paragraph."
        )
    messages.append({"role": "user", "content": user_content})
    return messages
