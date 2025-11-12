from typing import List, Optional, Tuple
import os, tempfile
from subprocess import call

import pypdf
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument, PDFEncryptionError
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTFigure
from pdfminer.pdfpage import PDFPage
from pytesseract import image_to_string
from PIL import Image

from .datamodel import ReaderConfig, ParsedPage, ParsedElement
from .utils import clean_text, pdf_to_images

def is_image_file(path: str) -> bool:
    return os.path.splitext(path.lower())[1] in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

def decrypt_pdf_with_qpdf(path: str):
    repaired = path.replace(".pdf", "_repaired.pdf")
    call(f'qpdf --password="" --decrypt "{path}" "{repaired}"', shell=True)
    os.remove(path)
    os.rename(repaired, path)

def open_pdf(pdf_path: str) -> Tuple[PDFDocument, PDFPageInterpreter, PDFPageAggregator, any, pypdf.PdfReader]:
    fp = open(pdf_path, "rb")
    parser = PDFParser(fp)
    try:
        document = PDFDocument(parser, password="")
    except PDFEncryptionError:
        fp.close(); decrypt_pdf_with_qpdf(pdf_path)
        fp = open(pdf_path, "rb"); parser = PDFParser(fp)
        document = PDFDocument(parser, password="")
    rsrcmgr = PDFResourceManager()
    device = PDFPageAggregator(rsrcmgr, laparams=LAParams())
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    return document, interpreter, device, fp, pypdf.PdfReader(pdf_path)

def parse_layout(layout_obj) -> List:
    items = []
    for obj in layout_obj:
        if isinstance(obj, (LTTextBox, LTTextLine)): items.append(obj)
        elif isinstance(obj, LTFigure): items.extend(parse_layout(obj._objs))
    return items

def get_pdf_pages(pdf_path: str, config: Optional[ReaderConfig] = None, force_ocr: bool = False) -> List[ParsedPage]:
    config = config or ReaderConfig()
    if is_image_file(pdf_path):
        text = image_to_string(Image.open(pdf_path))
        return [ParsedPage(index=0, width=1000, height=1000,
                elements=[ParsedElement(x0=0, y0=0, x1=1000, y1=1000, text=text)])]
    document, interpreter, device, fp, _ = open_pdf(pdf_path)
    pages = []
    for i, page in enumerate(PDFPage.create_pages(document)):
        interpreter.process_page(page)
        layout = device.get_result()
        elements = []
        for obj in parse_layout(layout):
            try:
                text = clean_text(obj.get_text())
                x0, y0, x1, y1 = obj.bbox
                elements.append(ParsedElement(x0=x0, y0=y0, x1=x1, y1=y1, text=text))
            except Exception: continue
        if force_ocr or not elements:
            with tempfile.TemporaryDirectory() as tmp:
                imgs = pdf_to_images(pdf_path, tmp, page_index_only=i)
                path = imgs.get(i)
                if path:
                    text = image_to_string(Image.open(path))
                    elements = [ParsedElement(x0=0, y0=0, x1=1000, y1=1000, text=clean_text(text))]
        mb = page.mediabox; width, height = mb[2]-mb[0], mb[3]-mb[1]
        pages.append(ParsedPage(index=i, width=width, height=height, elements=elements))
    fp.close()
    return pages
