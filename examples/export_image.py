"""Render a ptspy color-scheme palette to a PNG file.

Usage:
    python examples/export_image.py                  # 300x300 default
    python examples/export_image.py --size 600       # 600x600
"""

import argparse
from pathlib import Path

from ptspy import P, colorCube, colorScheme, frame, randomizer


def main():
    parser = argparse.ArgumentParser(description="Render a color-scheme palette image.")
    parser.add_argument("--size", type=int, default=300, help="Image width & height (square). Default: 300")
    args = parser.parse_args()
    size = args.size

    # This is the same demo code as shown in the Start guide
    # ---
    @frame(bg="000", size=(size, size))
    def draw(form, intensity=0.5, hues=9, swatches=9, variantSeed=0):
        cube = colorCube(steps=24)
        variant = randomizer(variantSeed)
        palette = colorScheme(cube, hues, swatches, intensity, variant)
        form.colorMap(palette[:, :, 0], P([0, 0], form.size), style="circle")

    img = draw(variantSeed=8)
    # ---

    out = Path(__file__).parent / "ptspy_sample_image.png"
    out.write_bytes(img.encodeToData())
    print(f"Saved {out} ({size}x{size})")


if __name__ == "__main__":
    main()
