"""Query utilities for retrieving document-aware context and LLM answers."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence

from .datamodel import ChunkMatch, TextChunk
from .llm_client import LLMClient

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def _normalise_tokens(text: str) -> List[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [token for token in tokens if token not in _STOPWORDS]


def _format_figure_list(metadata: Dict) -> List[str]:
    figures = metadata.get("figure_identifiers") or metadata.get("figure_identifier") or []
    if isinstance(figures, str):
        figures = [figures]
    return [fig for fig in figures if fig]


def _format_citation(metadata: Dict, fallback_page_index: int) -> str:
    page_number = metadata.get("page_number") or (fallback_page_index + 1)
    section_title = metadata.get("section_title") or f"Page {page_number}"
    figures = _format_figure_list(metadata)
    if not figures:
        return f"Page {page_number}, Section \"{section_title}\""
    if len(figures) == 1:
        return f"Page {page_number}, Section \"{section_title}\", {figures[0]}"
    joined = ", ".join(figures)
    return f"Page {page_number}, Section \"{section_title}\", Figures {joined}"


def _format_context_header(chunk: TextChunk) -> str:
    metadata = chunk.metadata or {}
    page_number = metadata.get("page_number") or (chunk.page_index + 1)
    section_title = metadata.get("section_title") or f"Page {page_number}"
    figures = _format_figure_list(metadata)
    figures_display = ", ".join(figures) if figures else "None"
    return f"Page {page_number} | Section: {section_title} | Figures: {figures_display}"


class TextQueryEngine:
    """Simple keyword-based query engine over chunked document text."""

    def __init__(self, chunks: Sequence[TextChunk | Dict]) -> None:
        self.chunks: List[TextChunk] = [
            chunk if isinstance(chunk, TextChunk) else TextChunk.from_dict(chunk)
            for chunk in chunks
        ]
        self._token_sets: List[set[str]] = []
        self._idf: Dict[str, float] = {}
        self._build_index()

    def _build_index(self) -> None:
        doc_freq: Counter[str] = Counter()
        token_sets: List[set[str]] = []

        for chunk in self.chunks:
            tokens = set(_normalise_tokens(chunk.text))
            token_sets.append(tokens)
            doc_freq.update(tokens)

        total_docs = max(1, len(self.chunks))
        idf: Dict[str, float] = {}
        for token, freq in doc_freq.items():
            idf[token] = math.log((1 + total_docs) / (1 + freq)) + 1.0

        self._token_sets = token_sets
        self._idf = idf

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, question: str, *, top_k: int = 3) -> List[ChunkMatch]:
        query_tokens = set(_normalise_tokens(question))
        if not query_tokens:
            return []

        scored: List[ChunkMatch] = []
        for chunk, tokens in zip(self.chunks, self._token_sets):
            overlap = tokens & query_tokens
            if not overlap:
                continue
            score = sum(self._idf.get(token, 1.0) for token in overlap)
            scored.append(ChunkMatch(chunk=chunk, score=score))

        scored.sort(key=lambda match: match.score, reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Question answering
    # ------------------------------------------------------------------
    def answer(
        self,
        question: str,
        *,
        llm_client: LLMClient,
        llm_model: str | None = None,
        llm_options: Dict | None = None,
        top_k: int = 3,
    ) -> Dict:
        matches = self.retrieve(question, top_k=top_k)
        context_blocks = [
            f"[{_format_context_header(match.chunk)}] {match.chunk.text}"
            for match in matches
        ]
        context_text = "\n\n".join(context_blocks)

        if context_text:
            prompt = (
                "You are an assistant that answers questions about financial reports. "
                "Use the provided context to answer the question.\n\n"
                "When answering, cite the relevant context using the format "
                "'Page X, Section \"Section Title\"' and include figure numbers when "
                "available (e.g. 'Page 5, Section \"Overview\", Figure 2'). "
                "List all citations at the end under 'Sources:'.\n\n"
                f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
            )
        else:
            prompt = (
                "You are an assistant that answers questions about financial reports. "
                "There was no relevant context retrieved, so rely on general reasoning.\n\n"
                "If you make claims, state that no document sources were found.\n\n"
                f"Question: {question}\nAnswer:"
            )

        response = llm_client.complete(
            prompt,
            model=llm_model,
            extra_payload=llm_options,
        )

        unique_citations: List[str] = []
        seen: set[str] = set()
        for match in matches:
            citation = _format_citation(match.chunk.metadata or {}, match.chunk.page_index)
            if citation not in seen:
                seen.add(citation)
                unique_citations.append(citation)

        return {
            "prompt": prompt,
            "response": response,
            "matches": [match.to_dict() for match in matches],
            "sources": unique_citations,
        }


def build_query_engine_from_data(data: Dict) -> TextQueryEngine:
    """Convenience helper to build a query engine from parsed report data."""

    chunks = data.get("text_chunks", [])
    return TextQueryEngine(chunks)

