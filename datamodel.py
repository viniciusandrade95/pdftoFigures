from dataclasses import dataclass, field
from typing import List, Optional, Dict

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

    def all_text(self) -> str:
        return " ".join(el.text for el in self.elements if el.text)
