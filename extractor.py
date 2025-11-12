from typing import List
from .datamodel import ParsedPage, ParsedTable, ParsedElement, ReaderConfig
import re

def group_paragraphs(elements: List[ParsedElement], cfg: ReaderConfig) -> List[str]:
    lines = sorted([e for e in elements if e.text], key=lambda e: -e.y0)
    paragraphs, current = [], []
    last_y = None
    for el in lines:
        if last_y is not None and abs(last_y - el.y0) > cfg.line_break_distance:
            paragraphs.append(" ".join(e.text for e in current)); current = []
        current.append(el); last_y = el.y0
    if current: paragraphs.append(" ".join(e.text for e in current))
    return paragraphs

def detect_tables(elements: List[ParsedElement], cfg: ReaderConfig) -> List[ParsedTable]:
    tables = []
    text_lines = [e.text for e in elements if e.text and re.search(r"\d", e.text)]
    for text in text_lines:
        if re.search(r"\d{3}[\.,]\d{3}", text):  # likely a table row
            y = [e for e in elements if text in e.text][0].y0
            tables.append(ParsedTable(x0=0, y0=y, x1=1000, y1=y+15, rows=[[text]]))
    return tables

def extract_from_pages(pages: List[ParsedPage], cfg: ReaderConfig) -> List[ParsedPage]:
    for page in pages:
        page.tables = detect_tables(page.elements, cfg)
    return pages
