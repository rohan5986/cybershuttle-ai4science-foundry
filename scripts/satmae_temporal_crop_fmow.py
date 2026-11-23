#!/usr/bin/env python3
"""Rebuild SatMAE's unpublished *_rgb_crop_0.jpg chips from raw fMoW-RGB.

SatMAE val_62classes.csv and CustomDatasetFromImagesTemporal both assume
filenames ending in `_rgb_crop_0.jpg` (the loader globs the last 15
characters). The public SpaceNet fMoW-RGB dump only has full scenes
(`*_rgb.jpg`) plus a sidecar JSON whose `bounding_boxes[0].box` is
`[x, y, width, height]`.

Default mode is a **tight** JSON box. `--pad-context` instead follows the
fMoW baseline context buffer in
https://github.com/fMoW/baseline/blob/master/code/data_ml_functions/dataFunctions.py
(the snippet SatMAE's README points at). `--resize 224` matches that
baseline's final JPEG size.

`--out-root` writes a parallel tree so a padded experiment does not
overwrite the tight chips used for the 62.60% result.
"""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

_OVERWRITE = False
_PAD = False
_RESIZE = 0
_VAL_ROOT = None
_OUT_ROOT = None


def _init_worker(overwrite: bool, pad: bool, resize: int, val_root: str, out_root: str) -> None:
    global _OVERWRITE, _PAD, _RESIZE, _VAL_ROOT, _OUT_ROOT
    _OVERWRITE = overwrite
    _PAD = pad
    _RESIZE = resize
    _VAL_ROOT = Path(val_root)
    _OUT_ROOT = Path(out_root) if out_root else None
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True


def _context_mult(ratio: float) -> float:
    # Copied from fMoW baseline dataFunctions.py (mutually exclusive bands).
    if ratio < 0.1:
        return 2.0
    if 0.1 <= ratio < 0.2:
        return 1.0
    if 0.2 <= ratio < 0.3:
        return 0.5
    if 0.3 <= ratio < 0.4:
        return 0.3
    if 0.4 <= ratio < 0.5:
        return 0.2
    return 0.15


def padded_window(x: int, y: int, w: int, h: int, img_w: int, img_h: int):
    """Return (left, upper, right, lower) with fMoW baseline context.

    The baseline uses OpenCV `img.shape` as (H, W, C) and sets
    `wRatio = box_w / H`, `hRatio = box_h / W`. Reproduce that, including
    the swapped denominators.
    """
    w_ratio = float(w) / float(img_h)
    h_ratio = float(h) / float(img_w)
    width_buffer = int((w * _context_mult(w_ratio)) / 2.0)
    height_buffer = int((h * _context_mult(h_ratio)) / 2.0)
    left = max(0, x - width_buffer)
    upper = max(0, y - height_buffer)
    right = min(img_w, x + w + width_buffer)
    lower = min(img_h, y + h + height_buffer)
    return left, upper, right, lower


def crop_one(rgb_path: str) -> str:
    src = Path(rgb_path)
    rel = src.relative_to(_VAL_ROOT)
    dest_dir = (_OUT_ROOT / rel.parent) if _OUT_ROOT is not None else src.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    dst = dest_dir / src.name.replace("_rgb.jpg", "_rgb_crop_0.jpg")
    if dst.is_file() and not _OVERWRITE:
        return "exists"

    meta = src.with_name(src.name.replace("_rgb.jpg", "_rgb.json"))
    if not meta.is_file():
        return "no_json"
    try:
        info = json.loads(meta.read_text())
    except (OSError, json.JSONDecodeError):
        return "bad_json"

    boxes = info.get("bounding_boxes") or []
    if not boxes:
        return "no_box"
    box = boxes[0].get("box")
    if not box or len(box) < 4:
        return "no_box"

    x, y, w, h = (int(round(float(v))) for v in box[:4])
    if w <= 0 or h <= 0:
        return "empty_box"

    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            if _PAD:
                left, upper, right, lower = padded_window(x, y, w, h, im.width, im.height)
            else:
                left = max(0, x)
                upper = max(0, y)
                right = min(im.width, x + w)
                lower = min(im.height, y + h)
            if right <= left or lower <= upper:
                return "empty_box"
            crop = im.crop((left, upper, right, lower))
            if _RESIZE > 0:
                crop = crop.resize((_RESIZE, _RESIZE), Image.BILINEAR)
            crop.save(dst, format="JPEG", quality=95)
    except (OSError, ValueError):
        return "bad_image"
    return "wrote"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", required=True, help="Directory of fMoW-RGB val scenes")
    parser.add_argument(
        "--out-root",
        default="",
        help="Optional parallel tree for chips (does not overwrite scenes under --val-root)",
    )
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--pad-context",
        action="store_true",
        help="fMoW baseline context pad (15 percent up to 2x) instead of a tight box",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=0,
        help="If >0, resize the chip to this square (baseline uses 224)",
    )
    args = parser.parse_args()

    val_root = Path(args.val_root).resolve()
    out_root = str(Path(args.out_root).resolve()) if args.out_root else ""
    rgb_files = sorted(str(p) for p in val_root.rglob("*_rgb.jpg") if "_msrgb" not in p.name)
    if not rgb_files:
        raise SystemExit(f"no *_rgb.jpg under {val_root}")

    counts = {
        "wrote": 0,
        "exists": 0,
        "no_json": 0,
        "bad_json": 0,
        "no_box": 0,
        "empty_box": 0,
        "bad_image": 0,
    }
    workers = max(1, args.workers)
    print(
        f"val_root={val_root} out_root={out_root or '(beside rgb)'} "
        f"pad={args.pad_context} resize={args.resize} n_rgb={len(rgb_files)} "
        f"workers={workers}",
        flush=True,
    )

    def consume(statuses):
        for i, status in enumerate(statuses, 1):
            counts[status] = counts.get(status, 0) + 1
            if i == 1 or i % 1000 == 0 or i == len(rgb_files):
                print(f"progress {i}/{len(rgb_files)} {counts}", flush=True)

    init_args = (args.overwrite, args.pad_context, args.resize, str(val_root), out_root)
    if workers == 1:
        _init_worker(*init_args)
        consume(crop_one(path) for path in rgb_files)
    else:
        with Pool(processes=workers, initializer=_init_worker, initargs=init_args) as pool:
            consume(pool.imap_unordered(crop_one, rgb_files, chunksize=16))

    print("crop_counts", counts, flush=True)
    if counts["wrote"] + counts["exists"] == 0:
        raise SystemExit("no AOI crops written")


if __name__ == "__main__":
    main()
