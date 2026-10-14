"""Gabungkan screenshot di folder screenshots/ menjadi 1 PDF (<1 MB) untuk form pengumpulan.

Jalankan dari root repo:  pip install pillow  &&  python tools/gabung_screenshot.py
Hasil: UI_chatbot.pdf
"""
import glob
import os
import sys

from PIL import Image

BATAS = 950_000  # byte, aman di bawah 1 MB
files = sorted(f for ext in ("png", "jpg", "jpeg") for f in glob.glob(f"screenshots/*.{ext}"))
if not files:
    sys.exit("Taruh screenshot (png/jpg) di folder screenshots/ dulu.")

for lebar in (1200, 1000, 800, 600):
    for kualitas in (85, 70, 55, 40):
        halaman = []
        for f in files:
            im = Image.open(f).convert("RGB")
            if im.width > lebar:
                im = im.resize((lebar, int(im.height * lebar / im.width)))
            halaman.append(im)
        halaman[0].save("UI_chatbot.pdf", save_all=True, append_images=halaman[1:], quality=kualitas)
        ukuran = os.path.getsize("UI_chatbot.pdf")
        if ukuran <= BATAS:
            sys.exit(print(f"OK: UI_chatbot.pdf ({len(files)} halaman, {ukuran // 1024} KB)"))
sys.exit("Masih lebih dari 1 MB; kurangi jumlah screenshot.")
