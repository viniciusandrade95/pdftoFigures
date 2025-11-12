import re
from typing import Any, Dict, List, Optional
from .datamodel import ReaderConfig, TextChunk
from .extractor import extract_from_pages, group_paragraphs
from .llm_client import LLMClient
from .converter import get_pdf_pages


CHUNK_SIZE_WORDS = 120
CHUNK_OVERLAP_WORDS = 30


def _split_into_chunks(
    paragraph: str,
    *,
    chunk_size_words: int = CHUNK_SIZE_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> List[str]:
    """Split text into overlapping word-based chunks.

    The helper keeps context overlap between consecutive chunks so that
    downstream retrieval has sufficient surrounding information while
    avoiding extremely long prompts.
    """

    words = paragraph.split()
    if not words:
        return []

    size = max(1, chunk_size_words)
    overlap = max(0, min(overlap_words, size - 1))
    step = max(1, size - overlap)

    chunks: List[str] = []
    start = 0
    total = len(words)
    while start < total:
        end = min(total, start + size)
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if end >= total:
            break
        start += step
    return chunks

def parse_financial_report(
    path: str,
    force_ocr: bool = False,
    *,
    llm_client: Optional[LLMClient] = None,
    llm_model: Optional[str] = None,
    llm_options: Optional[Dict[str, Any]] = None,
) -> Dict:
    cfg = ReaderConfig()
    pages = extract_from_pages(get_pdf_pages(path, cfg, force_ocr), cfg)
    data = {"metadata": {}, "sections": []}
    all_text = " ".join(p.all_text() for p in pages)
    match = re.search(r"(\b\d{4}\b)", all_text)
    if match: data["metadata"]["year"] = int(match.group(1))
    data["metadata"]["chunking"] = {
        "chunk_size_words": CHUNK_SIZE_WORDS,
        "overlap_words": CHUNK_OVERLAP_WORDS,
    }
    text_chunks: List[TextChunk] = []
    next_chunk_index = 0
    for page in pages:
        paragraphs = group_paragraphs(page.elements, cfg)
        page_chunks: List[TextChunk] = []
        for paragraph_index, paragraph in enumerate(paragraphs):
            for chunk_text in _split_into_chunks(
                paragraph,
                chunk_size_words=CHUNK_SIZE_WORDS,
                overlap_words=CHUNK_OVERLAP_WORDS,
            ):
                chunk = TextChunk(
                    page_index=page.index,
                    chunk_index=next_chunk_index,
                    text=chunk_text,
                    metadata={"paragraph_index": paragraph_index},
                )
                page_chunks.append(chunk)
                text_chunks.append(chunk)
                next_chunk_index += 1
        page.text_chunks = page_chunks
        if llm_client:
            for chunk_index, paragraph in enumerate(paragraphs):
                try:
                    response = llm_client.complete(
                        paragraph,
                        model=llm_model,
                        extra_payload=llm_options,
                    )
                except Exception as exc:  # pragma: no cover - safety net for external calls
                    response = {"error": str(exc)}
                page.llm_responses.append(
                    {
                        "page_index": page.index,
                        "chunk_index": chunk_index,
                        "prompt": paragraph,
                        "response": response,
                    }
                )
        for t in paragraphs:
            if re.search(r"assets|liabilities|equity|profit|income|capital", t.lower()):
                section = {"title": "Financial Summary", "text": t[:500]}
                data["sections"].append(section)
    if llm_client:
        data["llm_responses"] = [
            response
            for page in pages
            for response in page.llm_responses
        ]
    data["text_chunks"] = [chunk.to_dict() for chunk in text_chunks]
    return data
