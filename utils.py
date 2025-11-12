import re, os
import pytesseract
from PIL import Image
from pdf2image import convert_from_path, pdfinfo_from_path
from typing import List, Dict, Optional, Tuple

def clean_text(text: str) -> str:
    if not text: return ""
    return re.sub(r"\s+", " ", text).strip()

def parse_numeric_value(value: str):
    if not value: return None
    try:
        v = value.strip().replace(" ", "")
        if re.match(r"^\d{1,3}(\.\d{3})*,\d+$", v):
            v = v.replace(".", "").replace(",", ".")
        elif re.match(r"^\d{1,3}(,\d{3})*\.\d+$", v):
            v = v.replace(",", "")
        return float(v)
    except Exception:
        return None

def detect_and_correct_rotation(image: Image.Image) -> Image.Image:
    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        angle = osd.get("rotate", 0)
        if osd.get("orientation_conf", 0) > 1.0 and angle != 0:
            return image.rotate(-angle, expand=True)
        return image
    except Exception:
        return image

def pdf_to_images(path_to_pdf: str, output_dir: str, target_size: int = 1500, page_index_only: Optional[int] = None) -> Dict[int, str]:
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    info = pdfinfo_from_path(path_to_pdf)
    max_pages = info["Pages"]
    mapping = {}
    pages = range(1, max_pages + 1) if page_index_only is None else [page_index_only + 1]
    for p in pages:
        img = convert_from_path(path_to_pdf, first_page=p, last_page=p, fmt="jpg")[0]
        img = detect_and_correct_rotation(img)
        w, h = img.size
        if w > h: new_w, new_h = target_size, int(h * (target_size / w))
        else: new_w, new_h = int(w * (target_size / h)), target_size
        img = img.resize((new_w, new_h))
        out = os.path.join(output_dir, f"page_{p-1}.jpg")
        img.save(out, "JPEG")
        mapping[p - 1] = out
    return mapping
