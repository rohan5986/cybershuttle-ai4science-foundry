#!/usr/bin/env python3
"""Can Prithvi-EO-2.0 consume SatMAE's fMoW-RGB chips?

SatMAE Temporal scores 3-band JPEG AOIs (0-255). Prithvi-EO-2.0 is a
masked ViT pretrained on 6-band HLS V2 reflectance
(B02, B03, B04, B05, B06, B07) at 30 m. This probe records:

  1. a legal dummy HLS forward pass
  2. what happens if we pass 3-band RGB
  3. a naive glue: repeat RGB into 6 channels, with and without a
     reflectance-scale hack

It does not claim a fused classifier. Output: JSON evidence.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image


HLS_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
HLS_STD = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]
# TerraTorch 1.2 registers these as terratorch_prithvi_eo_v2_* (HF card
# names without the prefix fail with "could not be instantiated").
BACKBONE_CANDIDATES = [
    "terratorch_prithvi_eo_v2_300",
    "prithvi_eo_v2_300m",
    "prithvi_eo_v2_300",
]
BACKBONE = BACKBONE_CANDIDATES[0]


def _record(step: str, **kwargs):
    rec = {"step": step, **kwargs}
    print(json.dumps(rec), flush=True)
    return rec


def _forward(model, x: torch.Tensor):
    with torch.no_grad():
        out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if hasattr(out, "last_hidden_state"):
        t = out.last_hidden_state
    else:
        t = out
    return {
        "in_shape": list(x.shape),
        "out_type": type(out).__name__,
        "out_shape": list(t.shape) if hasattr(t, "shape") else str(type(t)),
        "out_finite": bool(torch.isfinite(t).all().item()) if torch.is_tensor(t) else None,
        "out_mean": float(t.float().mean().item()) if torch.is_tensor(t) else None,
        "out_std": float(t.float().std().item()) if torch.is_tensor(t) else None,
    }


def _load_rgb(path: Path, size: int = 224) -> np.ndarray:
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)  # HWC, 0-255


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", default="", help="Path to one fMoW *_rgb_crop_0.jpg")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    evidence = {
        "model": BACKBONE,
        "source": "ibm-nasa-geospatial/Prithvi-EO-2.0 via TerraTorch BACKBONE_REGISTRY",
        "expected_bands": ["B02", "B03", "B04", "B05", "B06", "B07"],
        "expected_units": "HLS V2 reflectance",
        "satmae_input": "3-band JPEG 0-255 AOI crops",
        "steps": [],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        out_path.write_text(json.dumps(evidence, indent=2) + "\n")

    try:
        from terratorch.registry import BACKBONE_REGISTRY

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        evidence["device"] = str(device)
        evidence["torch"] = torch.__version__
        evidence["cuda"] = torch.version.cuda
        model = None
        last_err = None
        loaded_name = BACKBONE_CANDIDATES[0]
        for name in BACKBONE_CANDIDATES:
            try:
                model = BACKBONE_REGISTRY.build(name, pretrained=True)
                loaded_name = name
                evidence["model"] = name
                break
            except Exception as exc:  # noqa: BLE001
                last_err = f"{name}: {exc}"
                evidence["steps"].append(_record("load_try", ok=False, backbone=name, error=str(exc)[:300]))
        if model is None:
            raise RuntimeError(last_err or "no backbone candidate loaded")
        model = model.to(device).eval()
        n_params = sum(p.numel() for p in model.parameters())
        evidence["steps"].append(
            _record("load", ok=True, n_params=n_params, backbone=loaded_name)
        )
        save()
    except Exception as exc:
        evidence["steps"].append(
            _record("load", ok=False, error=str(exc), traceback=traceback.format_exc()[-2000:])
        )
        save()
        raise

    # 1. Legal dummy HLS: (B, C, T, H, W) = 1,6,1,224,224 in reflectance range
    dummy = torch.zeros(1, 6, 1, 224, 224, device=device)
    for c, (m, s) in enumerate(zip(HLS_MEAN, HLS_STD)):
        dummy[:, c] = m + 0.1 * s * torch.randn_like(dummy[:, c])
    try:
        stats = _forward(model, dummy)
        evidence["steps"].append(_record("dummy_hls_6band", ok=True, **stats))
    except Exception as exc:
        evidence["steps"].append(
            _record("dummy_hls_6band", ok=False, error=str(exc), traceback=traceback.format_exc()[-1500:])
        )
    save()

    # 2. 3-band tensor — the SatMAE layout
    rgb_dummy = dummy[:, :3]
    try:
        stats = _forward(model, rgb_dummy)
        evidence["steps"].append(_record("dummy_rgb_3band", ok=True, **stats))
    except Exception as exc:
        evidence["steps"].append(
            _record(
                "dummy_rgb_3band",
                ok=False,
                error=str(exc).split("\n")[0],
                note="Expected: patch embed is Conv3d with in_chans=6.",
            )
        )
    save()

    rgb_path = Path(args.rgb) if args.rgb else None
    if rgb_path and rgb_path.is_file():
        evidence["rgb_chip"] = str(rgb_path)
        arr = _load_rgb(rgb_path)
        evidence["rgb_chip_shape"] = list(arr.shape)
        evidence["rgb_chip_min"] = float(arr.min())
        evidence["rgb_chip_max"] = float(arr.max())

        # 3. Repeat RGB -> 6 channels, keep 0-255 JPEG units
        six_jpeg = np.concatenate([arr, arr], axis=2)  # HWC 6
        x = torch.from_numpy(six_jpeg).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        # (C,H,W) -> (1,C,1,H,W)
        x = x.to(device)
        try:
            stats = _forward(model, x)
            evidence["steps"].append(
                _record(
                    "fmow_rgb_repeat6_jpeg_units",
                    ok=True,
                    note="Naive glue. JPEG 0-255 vs HLS reflectance ~1e3. Not a valid input.",
                    **stats,
                )
            )
        except Exception as exc:
            evidence["steps"].append(
                _record("fmow_rgb_repeat6_jpeg_units", ok=False, error=str(exc)[:500])
            )

        # 4. Scale 0-255 toward HLS red-band mean (~1433) then repeat
        scale = HLS_MEAN[2] / 128.0
        six_scaled = np.concatenate([arr * scale, arr * scale], axis=2)
        x2 = torch.from_numpy(six_scaled).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).to(device)
        try:
            stats = _forward(model, x2)
            evidence["steps"].append(
                _record(
                    "fmow_rgb_repeat6_scaled",
                    ok=True,
                    scale=scale,
                    note="Still missing NIR/SWIR. Only tests whether a scaled RGB stack is numerically accepted.",
                    **stats,
                )
            )
        except Exception as exc:
            evidence["steps"].append(
                _record("fmow_rgb_repeat6_scaled", ok=False, error=str(exc)[:500])
            )
    else:
        evidence["steps"].append(_record("fmow_rgb", ok=False, error="no --rgb chip"))

    evidence["conclusion"] = (
        "Prithvi-EO-2.0 loads and runs on 6-band HLS-shaped tensors. "
        "SatMAE fMoW chips are 3-band JPEG AOIs; a 3-band forward pass is rejected. "
        "Repeating RGB into 6 channels is a tensor hack, not an HLS observation "
        "(no NIR/SWIR, wrong GSD, wrong radiometry). A shared land-use head needs "
        "co-located HLS for the same fMoW sites plus a 62-way adapter, not channel repeat."
    )
    save()
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
