# PoC: fMoW baseline context padding vs tight AOI crops

Research question: **does rebuilding `*_rgb_crop_0.jpg` with the official
fMoW baseline context pad (the preprocess SatMAE's README points at)
move fMoW-Temporal Top-1 off the 61–63% tight-crop plateau toward the
79.99% repository number?**

This is the adapter we claimed was required before pairing SatMAE Temporal
with another EO model (Prithvi-EO): a shared AOI definition. It does not
retrain the checkpoint.

## What changed

Tight crops (Assignment 2) used `bounding_boxes[0].box` with no buffer.
Median chip was 204×162; 64% had a short side under 224.

Padded crops copy
[`dataFunctions.py`](https://github.com/fMoW/baseline/blob/master/code/data_ml_functions/dataFunctions.py)
(context 15% up to 2×, including the baseline's width/height ratio swap)
and resize to 224×224. Chips land in `$SATMAE_TEMPORAL_WORKDIR/data/padded/`
so the original tight tree is untouched.

## Run on PACE ICE

From the Foundry clone, after the Assignment 2 env/weights/S3 val already
exist:

```bash
cd ~/scratch/cybershuttle-ai4science-foundry
git checkout satmae-temporal-padded-poc
export SATMAE_TEMPORAL_WORKDIR="$HOME/scratch/satmae-temporal"
mkdir -p logs

sbatch scripts/satmae_temporal_crop_padded.sbatch          # wait: COMPLETED
sbatch scripts/satmae_temporal_eval_subset_padded.sbatch   # wait: COMPLETED
```

Same checkpoint, same `--seed 0` stratified 500-row draw, same evaluator.

## Observed on ICE (2026-11-22)

| Crop | Job | Partition | Elapsed | n | Top-1 | Top-5 | GPU |
|---|---|---|---|---|---|---|---|
| tight | 6197584 | `coc-gpu` 1× L40S | (Assignment 2) | 500 | **61.00%** | 77.80% | NVIDIA L40S |
| padded | 7245811 | `coc-gpu` 1× L40S | 00:02:58 | 500 | **65.20%** | 84.60% | NVIDIA L40S |

Padded crop job **7245680** (`coc-cpu`, 00:08:00, exit 0:0): 53,041 chips
written, 0 missing, `pad=True resize=224`. Parallel tree
`$SATMAE_TEMPORAL_WORKDIR/data/padded/`. Evaluator: missing keys 0,
unexpected 0, torch 1.12.1+cu113, peak GPU 2.05 GB.

Net: **+4.2 Top-1 / +6.8 Top-5** vs the tight subset. Still **14.8 points**
below the 79.99% repository number. Context padding is a real but partial
piece of the gap — not the whole miss.

Per-class (n=500, ~8 examples/class): padding helped context-heavy labels
(`multi-unit_residential` 0 → 0.625, `car_dealership` 0.375 → 1.0,
`police_station` 0 → 0.50) and hurt compact distinctive objects
(`nuclear_powerplant` 0.625 → 0, `airport` 0.625 → 0.25, `solar_farm`
0.875 → 0.50). 25 classes up, 22 down, 15 unchanged.

Artifacts in this folder: [metrics.json](metrics.json),
[subset_padded_7245811.out](subset_padded_7245811.out),
[nvidia-smi](subset_padded_7245811_nvidia-smi.txt),
[versions](subset_padded_7245811_versions.txt).

Full padded val was not run. The subset already falsifies “padding alone
recovers 79.99%.”
