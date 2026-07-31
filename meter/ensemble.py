"""Probability-averaging macron ensembles over old + new model dumps.

  python -m meter.ensemble [--norma-source hf|git] name=path.json [name=path.json ...]

All files share the old probs format: a JSON list of rows
{"source": ..., "seq": [[gold_label, P(long)], ...]} aligned by k-th dichronon, in
"dev rows then test rows" order (the order meter.predict's --norma now writes, for
either --norma-source). Rows whose gold sequence disagrees with the FIRST (reference)
file are excluded from averaging for that model only. Prob dumps produced before this
ordering convention (interleaved single-file order, git-only) are frozen artifacts --
don't mix them with freshly-generated dumps here, positional alignment won't hold.

Search: every subset of models, uniform prob average, scored on Norma DEV
(balanced acc); the dev-best subset is the headline and its TEST score is
reported alongside every single model and the full average.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict

from meter.norma_data import add_norma_source_arg, load_norma


def load_split(norma_source="hf"):
    """[is_dev, ...] for every macronize row, in "dev rows then test rows" order --
    must match the order meter.predict.run_norma() iterates when writing a probs
    dump, so this stays a valid positional mask for any freshly-generated dump."""
    norma = load_norma(norma_source)
    n_dev = sum(1 for d in norma["dev"] if d["task"] == "macronize")
    n_test = sum(1 for d in norma["test"] if d["task"] == "macronize")
    return [True] * n_dev + [False] * n_test


def bal_acc(pairs):
    recalls = []
    for c in (0, 1):
        tot = sum(1 for g, _ in pairs if g == c)
        if not tot:
            return None
        recalls.append(sum(1 for g, p in pairs if g == c and p == c) / tot)
    return sum(recalls) / 2


def metrics(pairs):
    n = len(pairs)
    acc = sum(g == p for g, p in pairs) / n
    return dict(n=n, acc=round(acc, 4), bal=round(bal_acc(pairs), 4))


def score(models, names, is_dev, want_dev, per_source=False):
    """Average P(long) over the given model names; argmax at 0.5."""
    pairs, by_src = [], defaultdict(list)
    ref = models[names[0]]
    for li, dv in enumerate(is_dev):
        if dv != want_dev:
            continue
        rows = [models[n][li] for n in names]
        gold = [g for g, _ in ref[li]["seq"]]
        ok = [r for r in rows if [g for g, _ in r["seq"]] == gold]
        if not ok:
            continue
        for k, g in enumerate(gold):
            p = sum(r["seq"][k][1] for r in ok) / len(ok)
            pred = 0 if p >= 0.5 else 1
            pairs.append((g, pred))
            by_src[ref[li]["source"]].append((g, pred))
    m = metrics(pairs)
    if per_source:
        bals = [b for s in sorted(by_src) if (b := bal_acc(by_src[s])) is not None]
        m["macro"] = round(sum(bals) / len(bals), 4)
    return m


def main():
    ap = argparse.ArgumentParser()
    add_norma_source_arg(ap)
    ap.add_argument("specs", nargs="+", metavar="name=path.json")
    a = ap.parse_args()
    specs = [s.split("=", 1) for s in a.specs]
    models = {n: json.load(open(p)) for n, p in specs}
    names = [n for n, _ in specs]
    is_dev = load_split(a.norma_source)
    n_rows = len(next(iter(models.values())))
    assert all(len(v) == n_rows == len(is_dev) for v in models.values())

    print(f"{n_rows} lines; models: {names}\n")
    print("singles (dev bal / test bal):")
    for n in names:
        d = score(models, [n], is_dev, True)
        t = score(models, [n], is_dev, False, per_source=True)
        print(f"  {n:<12} dev {d['bal']:.4f}  test {t['bal']:.4f} "
              f"(acc {t['acc']:.4f} macro {t['macro']:.4f})")

    results = []
    for r in range(2, len(names) + 1):
        for combo in itertools.combinations(names, r):
            d = score(models, list(combo), is_dev, True)
            results.append((d["bal"], combo))
    results.sort(reverse=True)
    print("\ntop-5 subsets by DEV bal:")
    for b, combo in results[:5]:
        print(f"  {b:.4f}  {'+'.join(combo)}")

    best = list(results[0][1])
    t = score(models, best, is_dev, False, per_source=True)
    print(f"\nDEV-SELECTED ensemble {'+'.join(best)}:")
    print(f"  TEST bal {t['bal']:.4f} acc {t['acc']:.4f} macro {t['macro']:.4f} "
          f"n {t['n']}")
    allm = score(models, names, is_dev, False, per_source=True)
    print(f"all-model average: TEST bal {allm['bal']:.4f} acc {allm['acc']:.4f} "
          f"macro {allm['macro']:.4f}")


if __name__ == "__main__":
    main()
