"""Crop figure PDFs to their content. Usage: uv run --with pillow --with pypdf python figures/crop_pdf.py <name> [<name>...]"""
import sys, subprocess
from PIL import Image, ImageChops
from pypdf import PdfReader, PdfWriter
DPI = 200; PAD = 4  # points of padding around content
for name in sys.argv[1:]:
    pdf = f"figures/{name}.pdf"
    subprocess.run(["pdftoppm", "-r", str(DPI), "-png", "-singlefile", pdf, f"/tmp/crop_{name}"], check=True)
    im = Image.open(f"/tmp/crop_{name}.png").convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bbox = ImageChops.difference(im, bg).getbbox()
    x0, y0, x1, y1 = [v * 72 / DPI for v in bbox]
    r = PdfReader(pdf); page = r.pages[0]
    H = float(page.mediabox.height)
    box = (x0 - PAD, H - y1 - PAD, x1 + PAD, H - y0 + PAD)
    page.mediabox.lower_left = (box[0], box[1]); page.mediabox.upper_right = (box[2], box[3])
    page.cropbox.lower_left = (box[0], box[1]); page.cropbox.upper_right = (box[2], box[3])
    w = PdfWriter(); w.add_page(page); w.write(pdf)
    print(name, "cropped to %.2f x %.2f in" % ((box[2]-box[0])/72, (box[3]-box[1])/72))
