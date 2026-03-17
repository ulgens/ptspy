"""Render a ptspy animation and export it as an MP4 file.

Usage:
    python examples/export_video.py                  # 200x200, 60 frames
    python examples/export_video.py --size 400       # 400x400
"""

import argparse
from pathlib import Path

import mediapy
from ptspy import frames


def main():
    parser = argparse.ArgumentParser(description="Export a ptspy animation as MP4.")
    parser.add_argument("--size", type=int, default=300, help="Video width & height (square). Default: 300")
    args = parser.parse_args()
    size = args.size

    # This is the same demo code as shown in the Start guide
    # ---
    @frames(numFrames=60, size=(size, size))
    def draw(form, index, radius):
        form.fillOnly("000").point(form.size / 2, radius + index)

    video = draw(radius=10)
    # ---

    out = Path(__file__).parent / "ptspy_sample_video.mp4"
    mediapy.write_video(str(out), video, codec="h264", fps=30)
    print(f"Saved {out} ({size}x{size}, {len(video)} frames)")


if __name__ == "__main__":
    main()
