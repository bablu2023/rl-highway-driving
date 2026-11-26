"""Turn a result video (.mp4) into a GIF that plays directly in the GitHub README.

    python make_gif.py results/best_table/best_agent.mp4                 # -> best_agent.gif
    python make_gif.py input.mp4 output.gif --fps 5 --max-frames 120

Needs: pip install "imageio[ffmpeg]" pillow
"""
import argparse
import os

import imageio
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("gif", nargs="?", help="output path (default: same name with .gif)")
    p.add_argument("--fps", type=float, default=5, help="playback speed of the GIF")
    p.add_argument("--max-frames", type=int, default=150, help="keep the GIF small for GitHub")
    args = p.parse_args()

    out = args.gif or os.path.splitext(args.video)[0] + ".gif"
    frames = imageio.mimread(args.video, memtest=False)[: args.max_frames]
    if not frames:
        raise SystemExit(f"No frames found in {args.video}")
    images = [Image.fromarray(f) for f in frames]
    images[0].save(out, save_all=True, append_images=images[1:],
                   duration=int(1000 / args.fps), loop=0)
    print(f"Saved {out} ({len(images)} frames, {os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
