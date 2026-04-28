#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from PIL import Image


def main():
    ap = argparse.ArgumentParser(description='Make a 2x2 collage from four images')
    ap.add_argument('--images', nargs=4, required=True, help='Four input images (in order: top-left, top-right, bottom-left, bottom-right)')
    ap.add_argument('--out', required=True, help='Output image path')
    ap.add_argument('--pad', type=int, default=10, help='Padding between images')
    args = ap.parse_args()

    # Load images
    imgs = [Image.open(p).convert('RGB') for p in args.images]
    # Normalize sizes to the smallest width/height among the four to keep aspect
    min_w = min(im.width for im in imgs)
    min_h = min(im.height for im in imgs)
    imgs = [im.resize((min_w, min_h)) for im in imgs]

    pad = args.pad
    w = min_w * 2 + pad * 3
    h = min_h * 2 + pad * 3
    canvas = Image.new('RGB', (w, h), color=(255, 255, 255))

    positions = [
        (pad, pad),
        (pad * 2 + min_w, pad),
        (pad, pad * 2 + min_h),
        (pad * 2 + min_w, pad * 2 + min_h),
    ]
    for im, (x, y) in zip(imgs, positions):
        canvas.paste(im, (x, y))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    canvas.save(args.out)
    print(f"[done] collage -> {args.out}")


if __name__ == '__main__':
    main()
