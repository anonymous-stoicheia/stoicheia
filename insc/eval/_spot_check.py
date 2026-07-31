import sys, json, os
sys.path.insert(0, '$CHARDIFF_ROOT/insc/eval')
import torch
import phi_disagree as pd
from eval.intrinsic import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CASES = [231434, 288529, 232015, 257833, 230453]

records = {r["phi_id"]: r for r in pd.load_records()}
by_digit = {}
for pid in CASES:
    r = records.get(pid)
    if r is None:
        print(f"PHI_{pid}: not in filtered record set (shouldn't happen)")
        continue
    by_digit.setdefault(r["digit"], []).append(r)

for digit, recs in sorted(by_digit.items()):
    ckpt = os.path.expandvars(pd.CKPT_BY_DIGIT[digit])
    print(f"\n=== loading fold digit={digit}: {ckpt} ===", flush=True)
    model, _ = load_model(ckpt, device)
    model.cfg.attn_impl = "sdpa"
    model.eval()
    for r in recs:
        full_ids, full_bnd = pd.text_to_ids_bnd(r["text"])
        for start, end, restored in r["spans"]:
            width = end - start + 1
            if width < 5:
                continue
            target_ids = [pd.A_IDX[pd.FOLD.get(c, c)] for c in restored]
            import numpy as np
            cands, _ = pd.restore_span(model, full_ids, full_bnd, start, width, device, beam_width=20)
            top_text, top_score = cands[0]
            epi_score = pd.score_known_in_window(model, full_ids, full_bnd, start, width,
                                                  np.array(target_ids), device)
            before, after = pd.ordinal_context(r["text"], start, end, n_letters=12)
            print(f"\nPHI_{r['phi_id']} width={width}")
            print(f"  {before}[{restored}]{after}")
            print(f"  epigrapher={restored!r} score={epi_score:.3f}")
            print(f"  model_top ={top_text!r} score={top_score:.3f}")
            print(f"  gap={top_score - epi_score:.3f}")
    del model
    torch.cuda.empty_cache()
print("\nDONE")
