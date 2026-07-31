import sys, os, json
sys.path.insert(0, '$CHARDIFF_ROOT/insc/eval')
import numpy as np
import torch
import phi_disagree as pd
from eval.intrinsic import load_model
from restore import beam_restore, ALIST, NLET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

records = {r["phi_id"]: r for r in pd.load_records()}
r = records[231434]
print("digit:", r["digit"], "text:", r["text"])
full_ids, full_bnd = pd.text_to_ids_bnd(r["text"])
start, end, restored = [s for s in r["spans"] if s[2] == "αναγυ"][0]
width = end - start + 1
print("start", start, "end", end, "width", width, "restored", restored)

ckpt = os.path.expandvars(pd.CKPT_BY_DIGIT[r["digit"]])
model, _ = load_model(ckpt, device)
model.cfg.attn_impl = "sdpa"
model.eval()

# ---- EXACT port of eval_span's own window/gap/beam_restore call ----
ctx = 768
L = width
s = start
chars = full_ids
lo = max(0, s - ctx // 2); hi = min(len(chars), s + L + ctx // 2)
window = chars[lo:hi].copy()
gap = list(range(s - lo, s - lo + L))
gold = "".join(ALIST[c] for c in window[gap])
print("gold letters at gap (should match epigrapher's restoration):", repr(gold))
window[gap] = pd.MASK
bnd = np.minimum(np.asarray(full_bnd[lo:hi]), 2).astype(np.int64)
bnd[gap] = pd.UNK_BND
cand = beam_restore(model, window, gap, bnd, device, beam_width=20)
print("eval_span-style beam_restore top candidates:", cand[:5])

# ---- my own restore_span, same data ----
cands2, _ = pd.restore_span(model, full_ids, full_bnd, start, width, device, beam_width=20)
print("my restore_span top candidates:", cands2[:5])
