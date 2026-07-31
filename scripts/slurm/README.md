# scripts/slurm/

These are cluster-specific SLURM `sbatch` templates, carried over from the five source repos
that were consolidated into this one (file names are prefixed by origin: `pretrain_*` from the
core pretraining repo, `tagger_*`, `syntax_*`, `meter_*`, `insc_*` from the four downstream
repos). They were written for and run on one specific SLURM cluster (GH200
nodes, apptainer containers, a particular SLURM account) and have been genericized just enough
to not contain that cluster's hardcoded account or absolute paths — they are **not**
drop-in-runnable on another cluster without editing.

Before submitting any of these, you MUST:

1. **Edit the SLURM account.** Every file has:
   ```
   #SBATCH -A YOUR_ACCOUNT   # EDIT: your SLURM account
   ```
   Replace `YOUR_ACCOUNT` with your own allocation/project account.

2. **Check the partition name.** Every file has:
   ```
   #SBATCH -p gpu   # EDIT: your GPU partition name
   ```
   `gpu` was this cluster's GPU partition name — yours may differ (`gpu-a100`, `mig`, etc).

3. **Check node/GPU counts and wall-time limits** (`--nodes`, `--gpus-per-node`, `-t`) against
   your own cluster's node shapes and queue limits — these were tuned for 4x GH200/node.

4. **Set `CHARDIFF_SIF`** (or otherwise adapt the `apptainer exec --nv $SIF ...` calls) to your
   own container image — the original NGC-based training container is not included in this
   repo. `env.sh` defaults `CHARDIFF_SIF` to this cluster's own shared container
   (`GreekCharBERT/containers/swift-megatron.sif`, the same one the sibling repos already use)
   so a submission that forgets to override it doesn't silently end up with `$SIF` empty (see
   point 7) — on another cluster, override `CHARDIFF_SIF` instead of relying on that default.
   If you're not using a container at all, replace the `apptainer exec ...` wrapper with a
   plain shell invocation of the same inner command.

5. **Set `CHARDIFF_DATA`** in your login-shell environment (or export it via `--export` on
   `sbatch`) before submitting — every script assumes `$CHARDIFF_ROOT/env.sh` (sourced inside
   the job) can resolve `CHARDIFF_DATA` for shard/checkpoint/run paths.

6. **Submit from the repo root, with a `logs/` directory already there** (`mkdir -p logs`).
   Every script's `#SBATCH -o`/`-e` is a relative path (`logs/<name>-%j.out`) precisely
   because `#SBATCH` directives are parsed statically by `sbatch` *before* the script body
   runs — a script-computed variable like `$CHARDIFF_ROOT` is never expanded there, so an
   earlier version of this template that used `-o $CHARDIFF_ROOT/logs/...` silently wrote
   logs into a literal directory named `$CHARDIFF_ROOT` instead of the real path. `sbatch`
   resolves a relative `-o`/`-e` against the directory you ran `sbatch` from, which is
   reliable as long as that's the repo root and `logs/` exists first (`sbatch` does not
   create it, and job launch fails outright if it doesn't exist).

7. **`CHARDIFF_DATA` must be exported in the *submitting* shell, or passed via `--export`**
   -- not just set inside your own login profile after the fact; `env.sh` has no default for
   it and sourcing fails outright without one. `CHARDIFF_SIF` now has a real default (point 4)
   so forgetting it no longer breaks silently, but the underlying failure mode is worth
   knowing about if you override it: a job whose apptainer/container step runs with an empty
   `$SIF` will try to interpret the next token on the command line (e.g. the literal word
   `bash`) as the image path and fail near-instantly with a cryptic "could not open image
   .../bash" error -- a real failure mode hit in practice (repeatedly, before the default was
   added), not a hypothetical one.

## What was and wasn't changed automatically

`scripts/genericize_sbatch.py` (kept in `scripts/`, alongside these, for provenance) did the
following mechanically when this repo was first consolidated: replaced the hardcoded SLURM
account line, annotated the partition line, replaced every absolute path into one of the five
old per-project repos with `$CHARDIFF_ROOT` (computed from the script's own location at the
top of each file — this also fixed the `source .../env.sh` lines), retargeted the
`insc_train/` / `insc_eval/` / `insc_data/` script paths to the merged `insc/` package layout,
dropped the (out-of-scope) `LemmaDiff-grc` side-repo from a few `PYTHONPATH` lines, and fixed
the `configs/` subdirectory prefixes (`configs/pretrain/`, `configs/syntax/`, `configs/insc/`)
to match this repo's `configs/` layout. A few remaining `/nobackup/...` data-tier literals
(samples files, run directories) were fixed by hand afterward to use the `$GCB_DATA` /
`$INS_DATA` conventions from `env.sh`.

The `-o`/`-e` log paths were an exception: the original mechanical fix replaced the old
absolute prefix with `$CHARDIFF_ROOT`, which looks reasonable but never actually works in an
`#SBATCH` directive (see point 6 above) -- confirmed in practice when 10 real jobs run this
way all wrote their logs into a literal `./$CHARDIFF_ROOT/logs/` directory instead of the repo's
own `logs/`. Fixed across all 28 affected scripts to use a relative path instead.

What it did **not** do: change wall-time budgets, node counts, or any cluster-topology-specific
tuning (rendezvous ports, `SNIC_TMP` staging assumptions in `stage_shards.sh`, etc.) — those
are exactly the kind of thing you should review before running on a different cluster.
