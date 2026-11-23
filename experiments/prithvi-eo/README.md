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

## What this is not

Not a 62-way fused classifier, not co-located HLS for fMoW sites, and not
a claim that channel-repeated RGB is HLS.
