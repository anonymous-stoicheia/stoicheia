#!/usr/bin/env python3
"""Fetch the Database of Byzantine Book Epigrams, which the corpus release omits.

DBBE is distributed under CC BY-NC-SA. The pretraining corpus is a CC BY-SA compilation,
and a non-commercial clause cannot be honoured inside one, so the 5,476 DBBE records
(~0.2M words, 0.1% of the corpus) are excluded from the released dataset even though they
were present in the corpus we pretrained on. This script rebuilds that corpus exactly, on
the reader's own terms rather than ours.

  python scripts/fetch_dbbe.py --out dbbe.jsonl

Anything you build from the result inherits DBBE's non-commercial and share-alike terms.
See https://www.dbbe.ugent.be for the licence and citation policy.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

API = "https://www.dbbe.ugent.be/api/occurrences"
UA = {"User-Agent": "stoicheia-corpus-rebuild/1.0 (+https://github.com/anonymous-stoicheia/stoicheia)"}


def fetch_page(page, per_page, retries=3):
    url = f"{API}?page={page}&limit={per_page}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:                                   # noqa: BLE001
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} after {type(e).__name__}", flush=True)
            time.sleep(5 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dbbe.jsonl")
    ap.add_argument("--per-page", type=int, default=100)
    ap.add_argument("--max-pages", type=int, default=0, help="0 = until exhausted")
    a = ap.parse_args()

    out = Path(a.out)
    n = page = 0
    with out.open("w", encoding="utf-8") as fh:
        while True:
            page += 1
            if a.max_pages and page > a.max_pages:
                break
            data = fetch_page(page, a.per_page)
            items = data.get("data") or data.get("items") or []
            if not items:
                break
            for it in items:
                text = (it.get("text") or "").strip()
                if not text:
                    continue
                fh.write(json.dumps({"source": "dbbe",
                                     "id": str(it.get("id", "")),
                                     "license": "CC BY-NC-SA",
                                     "text": text}, ensure_ascii=False) + "\n")
                n += 1
            print(f"  page {page}: {n} records so far", flush=True)
            time.sleep(1)                                        # be polite to the API

    print(f"wrote {n} records to {out}")
    print("These records are CC BY-NC-SA: anything you merge them into inherits those terms.")


if __name__ == "__main__":
    main()
