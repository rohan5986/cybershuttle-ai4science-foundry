# Execution evidence — SatMAE Temporal on PACE ICE

Copied from ICE after the 2026-10-11 runs. Weights and fMoW imagery are
**not** here; see the model README for DOI / S3 / SHA-256.

| Artifact | Path |
|---|---|
| Smoke metrics + confusion + nvidia-smi / versions / sacct / Slurm `.out` | [smoke/](smoke/) |
| Subset (500, seed 0) | [subset500/](subset500/) |
| Full val (53,041) | [full/](full/) |
| Resolved conda lock + pip freeze | [env/](env/) |
| Failure excerpts (then fixed) | [failures/](failures/) |

Passing GPU jobs: smoke **6196203**, subset **6197584**, full **6197773**.
All on `coc-gpu`, QOS `coc-ice`, 1× NVIDIA L40S.
