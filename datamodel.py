from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class ReaderConfig:
    space_max_distance: int = 6
    line_break_distance: int = 14
    column_tolerance: int = 10
    min_table_cols: int = 2
    min_table_rows: int = 2


@dataclass
class Rectangle:
    x0: float
    y0: float
    x1: float
    y1: float

    def width(self) -> float: return max(0, self.x1 - self.x0)
    def height(self) -> float: return max(0, self.y1 - self.y0)

    def v_overlap(self, other: "Rectangle", tol: float = 0) -> bool:
        return not (self.y1 < other.y0 - tol or self.y0 > other.y1 + tol)

    def h_overlap(self, other: "Rectangle", tol: float = 0) -> bool:
        return not (self.x1 < other.x0 - tol or self.x0 > other.x1 + tol)

    def collides_with(self, other: "Rectangle", tol: float = 0) -> bool:
        return self.v_overlap(other, tol) and self.h_overlap(other, tol)

    def to_dict(self) -> Dict[str, float]:
        return dict(x0=self.x0, y0=self.y0, x1=self.x1, y1=self.y1)


@dataclass
class ParsedElement(Rectangle):
    text: Optional[str] = None
    has_bold: bool = False
    element_type: str = "text"

    def is_empty(self) -> bool:
        return not self.text or self.text.strip() == ""


@dataclass
class ParsedTable(Rectangle):
    rows: List[List[str]] = field(default_factory=list)
    element_type: str = "table"
    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.rows)


@dataclass
class ParsedPage:
    index: int
    width: float
    height: float
    elements: List[ParsedElement] = field(default_factory=list)
    tables: List[ParsedTable] = field(default_factory=list)
    llm_responses: List[Dict[str, Any]] = field(default_factory=list)
    text_chunks: List["TextChunk"] = field(default_factory=list)

    def all_text(self) -> str:
        return " ".join(el.text for el in self.elements if el.text)


@dataclass
class TextChunk:
    """Chunk of parsed text associated with a page in the source document."""

    page_index: int
    chunk_index: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_index": self.page_index,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TextChunk":
        return cls(
            page_index=payload.get("page_index", 0),
            chunk_index=payload.get("chunk_index", 0),
            text=payload.get("text", ""),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class ChunkMatch:
    """Result of matching a chunk to a query, including its relevance score."""

    chunk: TextChunk
    score: float

    def to_dict(self) -> Dict[str, Any]:
        payload = self.chunk.to_dict()
        payload["score"] = self.score
        return payload
