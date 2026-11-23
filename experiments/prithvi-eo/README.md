# PoC: Prithvi-EO-2.0 next to SatMAE Temporal

SatMAE Temporal is RGB, three dates, 62-class fMoW land-use.
[Prithvi-EO-2.0](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)
(IBM / NASA / JSC) is a 6-band HLS ViT, not in the Foundry.

Question: **can a Prithvi encoder consume the chips this SatMAE recipe
already scores, or what adapter is missing?**

## Run

Separate env (do not reuse the SatMAE torch 1.12 prefix):

```bash
cd ~/scratch/cybershuttle-ai4science-foundry
git checkout satmae-temporal-padded-poc
sbatch scripts/prithvi_setup_env.sbatch          # wait: COMPLETED
sbatch scripts/prithvi_probe.sbatch              # wait: COMPLETED
```

Evidence: `$HOME/scratch/satmae-temporal/results/prithvi-probe/evidence.json`
(copy in this folder: [evidence.json](evidence.json)).

## Observed on ICE (2026-11-22)

| Stage | Job | Partition | State | Elapsed | Exit |
|---|---|---|---|---|---|
| Env (`terratorch` 1.2.11, torch 2.5.1+cu121, Python 3.11) | 7245699 | `coc-cpu` | COMPLETED | 00:06:37 | 0:0 |
| Probe, HF card id `prithvi_eo_v2_300m` | 7245812 | `coc-gpu` 1× L40S | FAILED | 00:04:15 | 1:0 |
| Probe retry, `terratorch_prithvi_eo_v2_300` | 7246491 | `coc-gpu` 1× V100 | COMPLETED | 00:04:48 | 0:0 |

Job 7245812 is a documented miss, not a cluster crash:
`BACKBONE_REGISTRY.build("prithvi_eo_v2_300m")` raises
“could not be instantiated from any source.” TerraTorch 1.2 registers
`terratorch_prithvi_eo_v2_300`. The Hugging Face card name is not the
registry key. First-failure JSON:
[evidence-7245812-name-miss.json](evidence-7245812-name-miss.json).

Job 7246491 loaded **303,886,336** parameters as
`terratorch_prithvi_eo_v2_300` (`pretrained=True`). Chip:
`crop_field_1021_0_rgb_crop_0.jpg` (224×224×3, JPEG 6–236).

| Step | ok | What happened |
|---|---|---|
| `dummy_hls_6band` `(1,6,1,224,224)` | yes | tokens `[1, 197, 1024]`, all finite |
| `dummy_rgb_3band` `(1,3,1,224,224)` | **no** | `Conv3d` weight `[1024, 6, 1, 16, 16]` vs 3 channels |
| `fmow_rgb_repeat6_jpeg_units` | yes* | tensor accepted; JPEG 0–255 vs HLS ~1e3 |
| `fmow_rgb_repeat6_scaled` | yes* | tensor accepted; still no NIR/SWIR |

\*Numerical accept is not a valid HLS observation.

## What this is not

Not a 62-way fused classifier, not co-located HLS for fMoW sites, and not
a claim that channel-repeated RGB is HLS. A shared land-use head needs
HLS for the same fMoW AOIs plus a 62-way adapter.
