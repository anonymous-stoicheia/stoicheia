#!/usr/bin/env python3
"""One-time genericization pass over scripts/slurm/*.sbatch, run once during the
CharDiff-grc consolidation (see the top-level README / repo history for context). Not meant
to be re-run against already-genericized files (it is not idempotent for the #SBATCH -A/-p
comment-appending steps). Kept for provenance / reproducibility of that pass.

Transforms, across every scripts/slurm/*.sbatch file:
  1. Insert a CHARDIFF_ROOT bootstrap line (derived from the script's own location) right
     after the trailing #SBATCH directive block, so every subsequent $CHARDIFF_ROOT
     reference below resolves without a hardcoded path.
  2. #SBATCH -A <acct>                -> #SBATCH -A YOUR_ACCOUNT   # EDIT: your SLURM account
  3. #SBATCH -p gpu                   -> same + "# EDIT: your GPU partition name"
  4. Any absolute path into one of the 5 pre-consolidation source repos -> $CHARDIFF_ROOT
     (this one substring rule handles -o/-e log paths, hardcoded *_ROOT= assignments, and
     "source .../env.sh" lines uniformly, since they're all textually that same prefix).
  5. scripts/stage_shards.sh -> scripts/pretrain/stage_shards.sh (moved during consolidation).
  6. insc_train/ , insc_eval/ , insc_data/ script-invocation paths -> insc/train/, insc/eval/,
     insc/data/ (those dirs were merged into the insc/ package during consolidation).
  7. Drop $LDF_ROOT from PYTHONPATH construction (the LemmaDiff-grc side-repo is out of scope
     for this release; see parser/model.py's module docstring).
  8. Per-origin configs/ prefix fixes, since configs/ now has pretrain/syntax/tagger/meter/insc
     subdirectories instead of being flat:
       pretrain_*.sbatch:  $GCB_ROOT/configs/  -> $GCB_ROOT/configs/pretrain/
       syntax_*.sbatch:    $SYN_ROOT/configs/  -> $SYN_ROOT/configs/syntax/
       insc_*.sbatch:      configs/finetune.json (arg default) -> configs/insc/finetune.json
     tagger_*.sbatch / meter_*.sbatch take their config as a plain CLI arg (no in-script
     configs/ path to fix), so only cosmetic usage-comment updates apply to them (see below).
  9. Cosmetic: update "Usage: sbatch scripts/X.sbatch" / "scripts/X.json" comments to the new
     scripts/slurm/<prefix>_X.sbatch / configs/<subsystem>/X.json paths, purely for readability.
"""
import re
import sys
from pathlib import Path

SLURM_DIR = Path(__file__).resolve().parent / "slurm"

OLD_REPO_RE = re.compile(
    r"$CHARDIFF_DATA"
    r"(?:GreekCharBERT|CharDiff-grc_tagger|CharDiff-grc_syntax|CharDiff-grc_meter|CharDiff-grc_inscriptions)"
)

BOOTSTRAP = (
    'CHARDIFF_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)\n'
    "export CHARDIFF_ROOT\n"
)


def genericize(text: str, origin: str) -> str:
    lines = text.split("\n")

    # 1) insert bootstrap right after the last leading #SBATCH line
    last_sbatch = -1
    for i, l in enumerate(lines):
        if l.startswith("#SBATCH"):
            last_sbatch = i
    if last_sbatch >= 0:
        lines.insert(last_sbatch + 1, BOOTSTRAP.rstrip("\n"))
    text = "\n".join(lines)

    # 2/3) account + partition
    text = re.sub(r"^#SBATCH -A \S+.*$", "#SBATCH -A YOUR_ACCOUNT   # EDIT: your SLURM account",
                  text, flags=re.MULTILINE)
    text = re.sub(r"^#SBATCH -p gpu\s*$", "#SBATCH -p gpu   # EDIT: your GPU partition name",
                  text, flags=re.MULTILINE)

    # 4) absolute source-repo paths -> $CHARDIFF_ROOT
    text = OLD_REPO_RE.sub("$CHARDIFF_ROOT", text)

    # 5) stage_shards.sh moved under scripts/pretrain/
    text = text.replace("$GCB_ROOT/scripts/stage_shards.sh",
                         "$GCB_ROOT/scripts/pretrain/stage_shards.sh")

    # 6) insc_{train,eval,data}/ -> insc/{train,eval,data}/
    text = text.replace("insc_train/", "insc/train/")
    text = text.replace("insc_eval/", "insc/eval/")
    text = text.replace("insc_data/", "insc/data/")

    # 7) drop the dropped LemmaDiff-grc side-repo from PYTHONPATH
    text = text.replace("$SYN_ROOT:$TAGGER_ROOT:$LDF_ROOT:$GCB_ROOT",
                         "$SYN_ROOT:$TAGGER_ROOT:$GCB_ROOT")

    # 8) per-origin configs/ prefix fixes
    if origin == "pretrain":
        text = text.replace("$GCB_ROOT/configs/", "$GCB_ROOT/configs/pretrain/")
    elif origin == "syntax":
        text = text.replace("$SYN_ROOT/configs/", "$SYN_ROOT/configs/syntax/")
    elif origin == "insc":
        text = re.sub(r'(\$\{1:-)configs/', r"\1configs/insc/", text)

    # 9) cosmetic usage-comment updates: scripts/X.sbatch -> scripts/slurm/<origin>_X.sbatch,
    #    configs/X.json -> configs/<origin>/X.json (only inside comment lines, best-effort)
    prefix = {"pretrain": "pretrain_", "tagger": "tagger_", "syntax": "syntax_",
              "meter": "meter_", "insc": "insc_"}[origin]

    def fix_comment_line(m):
        line = m.group(0)
        line = re.sub(r"\bscripts/(?!slurm/)([A-Za-z0-9_]+\.sbatch)",
                      lambda mm: f"scripts/slurm/{prefix}{mm.group(1)}", line)
        line = re.sub(rf"\bconfigs/(?!{origin}/)([A-Za-z0-9_]+\.json)",
                      lambda mm: f"configs/{origin}/{mm.group(1)}", line)
        return line

    text = "\n".join(
        fix_comment_line(re.match(r".*", l)) if l.strip().startswith("#") else l
        for l in text.split("\n")
    )

    return text


def main():
    origin_prefixes = {
        "pretrain_": "pretrain", "tagger_": "tagger", "syntax_": "syntax",
        "meter_": "meter", "insc_": "insc",
    }
    changed = []
    for f in sorted(SLURM_DIR.glob("*.sbatch")):
        origin = next((o for p, o in origin_prefixes.items() if f.name.startswith(p)), None)
        if origin is None:
            print(f"WARNING: no origin prefix recognized for {f.name}, skipping", file=sys.stderr)
            continue
        orig = f.read_text()
        new = genericize(orig, origin)
        if new != orig:
            f.write_text(new)
            changed.append(f.name)
    print(f"genericized {len(changed)} sbatch files:")
    for n in changed:
        print(" ", n)


if __name__ == "__main__":
    main()
