"""Run this once before building the installer to generate installer\app_icon.ico.

Usage (from the installer\ folder):
    python create_icon.py
"""
from pathlib import Path
from PIL import Image

src = Path(__file__).parent.parent / "whoilogo.png"
dst = Path(__file__).parent / "app_icon.ico"

img = Image.open(src).convert("RGBA")
img.save(
    dst,
    format="ICO",
    sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
)
print(f"Written: {dst}")
