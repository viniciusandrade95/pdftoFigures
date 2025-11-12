import re
from typing import Any, Dict, List, Optional
from .datamodel import ParsedPage, ReaderConfig
from .extractor import extract_from_pages, group_paragraphs
from .llm_client import LLMClient
from .converter import get_pdf_pages

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
    for page in pages:
        paragraphs = group_paragraphs(page.elements, cfg)
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
    return data
