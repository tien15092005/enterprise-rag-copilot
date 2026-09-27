from __future__ import annotations

from app.models.versioning import DocumentLanguage

PROMPT_VERSION = "rag_baseline_v1"

SYSTEM_PROMPT = """You are an enterprise knowledge assistant.
Answer only from the evidence supplied in the user message.
Do not use outside knowledge and do not invent policy details.
Every factual claim must cite one or more evidence keys exactly as [S1], [S2], and so on.
Never cite a key that is not present in the supplied evidence.
Respect the resolved document-version scope; evidence may come from one or multiple versions.
Answer in the requested answer language.
If the evidence does not support an answer, clearly say that there is insufficient evidence.
Keep the response concise and preserve factual distinctions between document versions."""


def build_user_prompt(
    *, question: str, context: str, language: DocumentLanguage, resolved_version_ids: list[str]
) -> str:
    language_name = "Vietnamese" if language is DocumentLanguage.VIETNAMESE else "English"
    versions = ", ".join(resolved_version_ids)
    return (
        f"Requested answer language: {language_name}\n"
        f"Resolved document version IDs: {versions}\n\n"
        f"Question:\n{question}\n\n"
        f"Evidence:\n{context}\n\n"
        "Answer using only the evidence and include valid citation keys."
    )
