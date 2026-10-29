# Stage 1: reproducing the AETHER paper's SatBird result

Notebook `09_aether_reconstruction.ipynb` checks that this project's port of the AETHER method
(`askmap.aether`) matches the authors' own code, by re-running their SatBird alignment experiment and comparing
the retrieval index with the paper's 0.24. The authors' code and data are not part of this repository; these
scripts fetch them and run the experiment in the authors' own environment.

The published code cannot be run as published: every alignment experiment names two configuration files that
are not in the repository. `configs/aether_reconstruction/` holds this project's reconstruction of them, with
every guessed value and its source recorded in the files.

## What you need

- About 5 GB of disk under `$ASKMAP_DATA_ROOT/vendor/` (the code, its own virtual environment, and the Zenodo
  data) and about 7 hours of CPU for the three seeds (roughly 2.3 hours each on a Mac mini).
- `uv`, `git`, and a Hugging Face login (the authors' code downloads CLIP).

## Steps

```bash
# 1. Fetch the authors' code (commit 7b382c5, MIT licence) and data (Zenodo record 23336313), and build their
#    environment under $ASKMAP_DATA_ROOT/vendor/
scripts/aether_stage1/setup.sh

# 2. Run the three seeds, keep each seed's best checkpoint, and run notebook 09.
#    AETHER_RUN_DIR is the folder that holds the logs and kept checkpoints.
export AETHER_RUN_DIR=$ASKMAP_DATA_ROOT/aether_run
scripts/aether_stage1/stage1.sh
```

`stage1.sh` runs `run_seeds.sh` for seeds 12345, 1 and 2 (skipping any seed that already finished), copies each
seed's best checkpoint aside with `keep_best.sh` as soon as the log reports it (the seeds share checkpoint file
names), checks that every seed finished, and then executes notebook 09.

Notebook 09 calls `r1b_export.py` inside the authors' environment to reload each checkpoint, run their test
loop, and save the tensors this project's port needs to recompute the same loss and index.

## Reading the result

Over the three seeds this project's run scored 0.236 against the paper's 0.24. The notebook's rule, fixed before
it ran: a three-seed mean within 0.02 of the paper is weak support that the set-up matches the authors'; a bigger
gap means the set-up is uncertain; it is not called a reproduction either way.
