"""Compare epigraphers' bracket-restorations in PHI editions against our model's own
high-confidence predictions, using leak-proof fold checkpoints (each inscription is
scored by the fold whose held-out test digit matches its own PHI_ID's last digit, so
the model provably never trained on it).

For each clean, single-run restoration span:
  - EXACT: beam-20 restore at the epigrapher's own gap width, plus a teacher-forced
    score of the epigrapher's actual restored text at that same width.
  - ELASTIC (gaps > 3 letters only): beam-20 restore at width-1/width/width+1/width+2
    (a +/-2 sweep around the epigrapher's width), scored by mean log-prob per letter
    so different widths are comparable; best-scoring width reported alongside exact.

Ranked by CONFIDENCE GAP = model's own top-beam score - model's teacher-forced score
for the epigrapher's restoration (mean log-prob per letter, so gap widths compare
fairly). High gap + model's own pick != epigrapher's pick = the interesting cases.

  python insc/eval/phi_disagree.py --out disagree.json --limit 30
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import torch

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "data"))
from data.normalize import ALPHABET
from eval.intrinsic import load_model
sys.path.insert(2, str(Path(__file__).resolve().parent))
from restore import beam_restore, _char_logp

ALIST = list(ALPHABET)
A_IDX = {c: i for i, c in enumerate(ALIST)}
MASK, UNK_BND, UNK_DIA, UNK_PUNCT = 24, 3, 48, 6
RAW_JSONL = os.path.expandvars("$INS_DATA/raw/iphi.jsonl")
FOLD = {"ς": "σ", "ϲ": "σ", "ϙ": "κ", "ϛ": "σ"}

CKPT_BY_DIGIT = {
    0: "$INS_DATA/runs/both_ft_docclean_t0v1/best.pt",
    1: "$INS_DATA/runs/both_ft_docclean_t1v2/best.pt",
    2: "$INS_DATA/runs/both_ft_docclean_t2v3/best.pt",
    3: "$INS_DATA/runs/both_ft_docclean_t3v4/best.pt",
    4: "$INS_DATA/runs/both_ft_docclean_t4v5/best.pt",
    5: "$INS_DATA/runs/both_ft_docclean_t5v6/best.pt",
    6: "$INS_DATA/runs/both_ft_docclean_t6v7/best.pt",
    7: "$INS_DATA/runs/both_ft_docclean_t7v8/best.pt",
    8: "$INS_DATA/runs/both_ft_docclean_t8v9/best.pt",
    9: "$INS_DATA/runs/both_ft_docclean_t9v0/best.pt",
}

# ------------------------------------------------------------------ Leiden markup parsing

TAG_RE = re.compile(r"<[^>]+>")
ANGLE_RE = re.compile(r"&lt;|&gt;")
CURLY_RE = re.compile(r"\{[^{}]*\}")
BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")
GREEK_RANGES = [(0x0370, 0x03FF), (0x1F00, 0x1FFF)]
REJECT_CHARS = set("-—–.•‧◌?")


def is_greek_letter(ch):
    base = unicodedata.normalize("NFD", ch)[0]
    cp = ord(base)
    return any(lo <= cp <= hi for lo, hi in GREEK_RANGES) and unicodedata.category(base).startswith("L")


def fold_letter(ch):
    return unicodedata.normalize("NFD", ch)[0].lower()


class _Builder:
    def __init__(self):
        self.chars = []
        self.ordinal = -1
        self.pending_space = False
        self.ok = True

    def feed(self, ch):
        if is_greek_letter(ch):
            if self.pending_space and self.chars:
                self.chars.append(" ")
            self.pending_space = False
            self.ordinal += 1
            self.chars.append(fold_letter(ch))
            return self.ordinal
        elif ch.isspace():
            if self.chars:
                self.pending_space = True
            return None
        else:
            self.ok = False
            return None


def parse_record(edition):
    """-> (clean_text, [(start_ord, end_ord_inclusive, restored_text)]) or None.
    Only accepts records where every [...] span is a single contiguous letter-run
    (no internal spaces/dashes/uncertainty marks) and no stray lacuna markers exist
    anywhere else in the text either -- i.e. a fully, cleanly restored edition."""
    s = TAG_RE.sub(" ", edition)
    s = ANGLE_RE.sub("", s)
    s = CURLY_RE.sub("", s)
    b = _Builder()
    spans = []
    pos = 0
    for m in BRACKET_RE.finditer(s):
        for ch in s[pos:m.start()]:
            b.feed(ch)
        inner = m.group(1)
        if not inner.strip():
            b.ok = False
        start_ord = end_ord = None
        restored = []
        for ch in inner:
            o = b.feed(ch)
            if o is not None:
                if start_ord is None:
                    start_ord = o
                end_ord = o
                restored.append(b.chars[-1])
            elif ch.isspace():
                b.ok = False   # multi-word restoration span -- skip for this pass
        if start_ord is not None:
            if end_ord - start_ord + 1 != len(restored):
                b.ok = False
            spans.append((start_ord, end_ord, "".join(restored)))
        pos = m.end()
    for ch in s[pos:]:
        b.feed(ch)
    if not b.ok or not spans:
        return None
    return "".join(b.chars), spans


def normalize_ithaca(s):
    return " ".join(s.split())


def canon(s):
    return "".join(FOLD.get(c, c) for c in s)


def ordinal_context(text, start, end, n_letters=15):
    """text may contain spaces; start/end are LETTER-ordinal positions (spaces don't
    get an ordinal -- same convention as text_to_ids_bnd/full_ids). Returns (before,
    after) substrings of the surrounding text, each covering up to n_letters actual
    letters (plus their interleaved spaces), found via an explicit ordinal -> string-
    index map -- NOT a naive character-index slice, which silently mismatches the
    two indexing schemes and produces garbled, misleading context."""
    ord_to_idx = []   # ord_to_idx[k] = string index of the k-th letter
    for i, ch in enumerate(text):
        if ch != " ":
            ord_to_idx.append(i)
    n = len(ord_to_idx)

    before_from_ord = max(0, start - n_letters)
    before = text[ord_to_idx[before_from_ord]:ord_to_idx[start]] if start > 0 else ""

    after_to_ord = min(n - 1, end + n_letters)
    after = text[ord_to_idx[end] + 1:ord_to_idx[after_to_ord] + 1] if end < n - 1 else ""
    return before, after


# ------------------------------------------------------------------ model inference

def text_to_ids_bnd(text):
    """lowercase text with single spaces between words -> (ids, boundary) arrays,
    boundary=1 at word end, 2 at the very last letter (sentence end).

    Canon-folds each letter (FOLD: e.g. final-sigma 'ς' -> 'σ') before the A_IDX
    lookup -- the model's 24-letter vocabulary has no separate final-sigma class,
    so without this fold every 'ς' fails `ch in A_IDX` and gets silently DROPPED
    from the array instead of mapped, shrinking full_ids by one for every prior
    final-sigma relative to the ordinal count parse_record/ordinal_context use.
    That silent, cumulative one-off shift was the actual cause of every
    "model prefers a shifted, context-echoing string" artifact in this pipeline."""
    ids, bnd = [], []
    for ch in text:
        if ch == " ":
            if bnd:
                bnd[-1] = 1
        else:
            folded = FOLD.get(ch, ch)
            if folded in A_IDX:
                ids.append(A_IDX[folded]); bnd.append(0)
    if bnd:
        bnd[-1] = 2
    return np.array(ids, np.int64), np.array(bnd, np.int64)


@torch.no_grad()
def score_known(model, ids, bnd_row, gap, target_ids, device):
    """Teacher-forced mean log P(target letter) per gap position, given the true
    text filled in (context is fully known outside the gap either way)."""
    seq = ids.copy()
    seq[gap] = target_ids
    logp = _char_logp(model, [seq], bnd_row, device)[0]   # (T, 24)
    return float(sum(logp[p, c].item() for p, c in zip(gap, target_ids)) / len(gap))


def restore_span(model, full_ids, full_bnd, start, width, device, beam_width=20, ctx=768):
    """Mask `width` letters starting at `start`, beam-restore, return
    [(text, mean_logp), ...] sorted best first. Matches insc/eval/restore.py's own
    validated eval_span exactly: boundary is UNK_BND only INSIDE the gap -- freeing it
    at the fully-visible letter just before the gap (an earlier version of this
    function did that) is input the model never sees in training and produced
    degenerate, context-echoing output, not a genuine alternative reading."""
    n = len(full_ids)
    gap = list(range(start, start + width))
    lo = max(0, gap[0] - ctx // 2); hi = min(n, gap[-1] + ctx // 2)
    ids = full_ids[lo:hi].copy()
    bnd = full_bnd[lo:hi].copy()
    gap_rel = [g - lo for g in gap]
    ids[gap_rel] = MASK
    bnd = np.minimum(bnd, 2)
    bnd[gap_rel] = UNK_BND
    cand = beam_restore(model, ids, gap_rel, bnd, device, beam_width)
    out = [(txt, score / width) for txt, score in cand]
    out.sort(key=lambda x: -x[1])
    return out, (lo, gap_rel)


def score_known_in_window(model, full_ids, full_bnd, start, width, target_ids, device, ctx=768):
    """Must free the SAME (and only the same) boundary positions restore_span does,
    or the two scores aren't comparable."""
    n = len(full_ids)
    gap = list(range(start, start + width))
    lo = max(0, gap[0] - ctx // 2); hi = min(n, gap[-1] + ctx // 2)
    ids = full_ids[lo:hi].copy()
    bnd = np.minimum(full_bnd[lo:hi].copy(), 2)
    gap_rel = [g - lo for g in gap]
    bnd[gap_rel] = UNK_BND
    return score_known(model, ids, bnd, gap_rel, target_ids, device)


# ------------------------------------------------------------------ main

def load_records():
    out = []
    with open(RAW_JSONL, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ed = r.get("edition", "")
            if "[" not in ed:
                continue
            digit = int(str(r["PHI_ID"])[-1])
            if digit not in CKPT_BY_DIGIT:
                continue
            parsed = parse_record(ed)
            if parsed is None:
                continue
            clean, spans = parsed
            ith = normalize_ithaca(r.get("ithaca_text", "")).rstrip(" .")
            if ith != clean:
                continue
            if not (20 <= len(clean) <= 767):
                continue
            out.append(dict(phi_id=r["PHI_ID"], digit=digit, text=clean, spans=spans,
                            main_region=r.get("main_region"), tpq=r.get("tpq"), taq=r.get("taq")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0, help="cap total spans scored (0 = all)")
    ap.add_argument("--beam", type=int, default=20)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("loading records + parsing editions ...", flush=True)
    records = load_records()
    by_digit = {}
    for r in records:
        by_digit.setdefault(r["digit"], []).append(r)
    print(f"records: {[(d, len(v)) for d, v in sorted(by_digit.items())]}", flush=True)

    results = []
    for digit, recs in sorted(by_digit.items()):
        ckpt = os.path.expandvars(CKPT_BY_DIGIT[digit])
        print(f"loading fold checkpoint for digit={digit}: {ckpt}", flush=True)
        model, _ = load_model(ckpt, device)
        model.cfg.attn_impl = "sdpa"
        model.eval()

        for r in recs:
            full_ids, full_bnd = text_to_ids_bnd(r["text"])
            for start, end, restored in r["spans"]:
                width = end - start + 1
                if start >= len(full_ids) or end >= len(full_ids):
                    continue
                # the model's letter vocabulary has no separate final-sigma class (that's
                # a rendering-time/boundary decision, not a distinct identity) -- canon-fold
                # before indexing, same as the agree/disagree comparison already does
                target_ids = np.array([A_IDX[FOLD.get(c, c)] for c in restored], dtype=np.int64)

                exact_cands, _ = restore_span(model, full_ids, full_bnd, start, width, device, a.beam)
                if not exact_cands:
                    continue
                top_text, top_score = exact_cands[0]
                epi_score = score_known_in_window(model, full_ids, full_bnd, start, width,
                                                  target_ids, device)
                epi_canon, top_canon = canon(restored), canon(top_text)
                agree = epi_canon == top_canon

                elastic = None
                if width > 3:
                    best = None
                    for w in (width - 2, width - 1, width + 1, width + 2):
                        if w < 1 or start + w > len(full_ids):
                            continue
                        cands, _ = restore_span(model, full_ids, full_bnd, start, w, device, a.beam)
                        if cands and (best is None or cands[0][1] > best[1]):
                            best = (cands[0][0], cands[0][1], w)
                    if best and best[1] > top_score:
                        elastic = dict(text=best[0], width=best[2], mean_logp=round(best[1], 4))

                ctx_before, ctx_after = ordinal_context(r["text"], start, end, n_letters=15)
                results.append(dict(
                    phi_id=r["phi_id"], digit=digit, region=r["main_region"],
                    tpq=r["tpq"], taq=r["taq"],
                    context_before=ctx_before,
                    context_after=ctx_after,
                    start=start, end=end,
                    epigrapher=restored, width=width,
                    model_top=top_text, model_top_score=round(top_score, 4),
                    epigrapher_score=round(epi_score, 4),
                    confidence_gap=round(top_score - epi_score, 4),
                    agree=agree,
                    elastic=elastic,
                    beam_top20=[t for t, _ in exact_cands[:20]],
                ))
                if a.limit and len(results) >= a.limit:
                    break
            if a.limit and len(results) >= a.limit:
                break
        del model
        torch.cuda.empty_cache()
        if a.limit and len(results) >= a.limit:
            break

    results.sort(key=lambda r: (-r["confidence_gap"] if not r["agree"] else -999, ))
    disagree = [r for r in results if not r["agree"]]
    disagree.sort(key=lambda r: -r["confidence_gap"])

    out_path = a.out or os.path.expandvars("$INS_DATA/phi_disagree.json")
    json.dump(dict(n_total=len(results), n_disagree=len(disagree), results=results),
              open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"scored {len(results)} spans, {len(disagree)} disagreements -> {out_path}", flush=True)
    print("\ntop 10 by confidence gap:")
    for r in disagree[:10]:
        print(f"  PHI_{r['phi_id']}: epigrapher='{r['epigrapher']}' (score={r['epigrapher_score']}) "
              f"vs model='{r['model_top']}' (score={r['model_top_score']}) gap={r['confidence_gap']}")


if __name__ == "__main__":
    main()
