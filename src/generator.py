"""
Generator: take a question + retrieved context, ask Claude to answer.
"""
from dataclasses import dataclass

from anthropic import Anthropic

from src.config import ANTHROPIC_API_KEY, GENERATION_MODEL, TOP_K
from src.retriever import format_for_prompt, RetrievalResult
from src.unified_retriever import retrieve_unified


_client = Anthropic(api_key=ANTHROPIC_API_KEY)


SYSTEM_PROMPT = """You are a Friends TV show expert assistant. Answer questions about the show using ONLY the provided scene excerpts as your source of truth.

Rules:
1. Base your answer strictly on the provided sources. Do not use outside knowledge about Friends, even if you know it.
2. When you state a fact, cite the source like this: "Ross yells 'PIVOT' while moving a couch (S05E16, Scene 3)."
3. If the sources do not contain enough information to answer the question, say so plainly: "I couldn't find enough information in the retrieved scenes to answer this." Do not guess or fill gaps from prior knowledge.
4. Keep answers concise. Quote dialogue sparingly and only when it directly answers the question.
5. If sources contradict each other or describe similar moments across episodes, note that."""


@dataclass
class RAGAnswer:
    """The complete output of a RAG query: the answer plus the sources it used."""
    question: str
    answer: str
    sources: list[RetrievalResult]
    mode: str


def build_user_prompt(question: str, results: list[RetrievalResult]) -> str:
    context = format_for_prompt(results)
    return (
        f"Here are some scene excerpts from Friends:\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the scenes above. Cite sources as you go."
    )


def answer_question(question: str, mode: str = "vector", top_k: int = TOP_K) -> RAGAnswer:
    """
    Retrieve relevant scenes using the given mode, then ask Claude to answer.
    """
    sources = retrieve_unified(question, mode=mode, top_k=top_k)
    user_prompt = build_user_prompt(question, sources)

    response = _client.messages.create(
        model=GENERATION_MODEL,
        max_tokens=1024,
        temperature=0.3,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    answer_text = response.content[0].text

    return RAGAnswer(question=question, answer=answer_text, sources=sources, mode=mode)


if __name__ == "__main__":
    import sys
    question = sys.argv[1] if len(sys.argv) > 1 else "What does Ross yell when moving a couch?"
    mode = sys.argv[2] if len(sys.argv) > 2 else "vector"

    print(f"Q: {question}\nMode: {mode}\n")
    result = answer_question(question, mode=mode)

    print(f"A: {result.answer}\n")
    print(f"--- Sources used ---")
    for i, src in enumerate(result.sources, start=1):
        print(f"{i}. {src.citation}  (distance: {src.distance:.3f})")
