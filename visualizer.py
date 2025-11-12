from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont
from .datamodel import ParsedPage, Rectangle

def draw_elements(page: ParsedPage, out_path: str, color: Tuple[int,int,int]=(255,0,0)):
    img = Image.new("RGB", (int(page.width), int(page.height)), (255,255,255))
    draw = ImageDraw.Draw(img)
    for e in page.elements:
        draw.rectangle([(e.x0, page.height - e.y1), (e.x1, page.height - e.y0)], outline=color, width=1)
    img.save(out_path)
