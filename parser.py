import re
from typing import Dict, List
from .datamodel import ParsedPage, ReaderConfig
from .extractor import extract_from_pages, group_paragraphs
from .converter import get_pdf_pages

def parse_financial_report(path: str, force_ocr: bool = False) -> Dict:
    cfg = ReaderConfig()
    pages = extract_from_pages(get_pdf_pages(path, cfg, force_ocr), cfg)
    data = {"metadata": {}, "sections": []}
    all_text = " ".join(p.all_text() for p in pages)
    match = re.search(r"(\b\d{4}\b)", all_text)
    if match: data["metadata"]["year"] = int(match.group(1))
    for page in pages:
        paragraphs = group_paragraphs(page.elements, cfg)
        for t in paragraphs:
            if re.search(r"assets|liabilities|equity|profit|income|capital", t.lower()):
                section = {"title": "Financial Summary", "text": t[:500]}
                data["sections"].append(section)
    return data
