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

Compare:

- tight subset: `$SATMAE_TEMPORAL_WORKDIR/results/subset500/metrics.json` (61.00%, job 6197584)
- padded subset: `$SATMAE_TEMPORAL_WORKDIR/results/subset500-padded/metrics.json`

Same checkpoint, same `--seed 0` stratified 500-row draw, same evaluator.

## Expected vs observed

Filled after the ICE jobs finish.
