# Stoicheia

A character-level masked-diffusion Transformer for Ancient Greek, pretrained on an
open, revision-pinned corpus and released as eleven decontaminated checkpoints (ten
rotated literary folds + one documentary-clean model), fine-tuned for restoration of
damaged inscriptions/papyri, morphosyntactic tagging and dependency parsing, and
macronization/metrical scansion.

This repository is the training/evaluation code. The pretrained and fine-tuned model
weights are on the HuggingFace Hub — see [`MODEL_CARDS_INDEX.md`](MODEL_CARDS_INDEX.md)
for the full list, or jump straight to
[`ANON-ORG/Stoicheia-doc_clean`](https://huggingface.co/ANON-ORG/Stoicheia-doc_clean)
(the flagship backbone) or
[`ANON-ORG/Stoicheia-restoration`](https://huggingface.co/ANON-ORG/Stoicheia-restoration) /
[`-tagger-parser`](https://huggingface.co/ANON-ORG/Stoicheia-tagger-parser) for a
ready-to-use downstream model (a `-meter` model is planned but not yet published —
see `MODEL_CARDS_INDEX.md`). All model repos are currently **private**; they will be
made public alongside publication.

## Quickstart (no training required)

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("ANON-ORG/Stoicheia-doc_clean", trust_remote_code=True)

from processing_char_bert import CharBertProcessor  # ships in the model repo
processor = CharBertProcessor()

# a lacuna of UNCERTAIN width, in text that's ALSO fully bare scriptio continua (no
# spaces, no accents) -- the realistic case for damaged, unaccented primary sources.
# Write "[N±M]" for a best-guess width N and a plausible range N-M..N+M; every
# candidate width is scored by the model's own confidence, recovering both the
# width and the text while jointly restoring accents/word-boundaries throughout.
text = "εναρχηηνο[5±3]καιολογοςηνπροστονθεον"
best_text, best_width, candidates = processor.restore_elastic(model, text, mask_dia_boundary=True)
print(best_text)  # -> ἐν ἀρχῇ ἦν ὁ λόγος, καὶ ὁ λόγος ἦν πρὸς τὸν θεόν.
```

## What's here

- `model/`, `data/`, `train/`, `eval/` — the pretraining architecture (`CharBertEncoder`,
  a five-plane character-level masked-diffusion Transformer) and training loop.
- `insc/` — restoration fine-tuning (inscriptions + papyri) and strict-protocol
  evaluation (same-harness comparison against DeepMind's Ithaca).
- `tagger/`, `parser/` — morphosyntactic tagging (factored XPOS, edit-script lemma,
  UPOS) and biaffine dependency parsing, plus a joint multi-task model and a
  pluggable HuggingFace-encoder bridge for cross-encoder ablations.
- `meter/` — macronization (vowel length) and metrical scansion, including the
  *Norma* benchmark protocol and rule-based silver-data mining pipeline.
- `tests/` — CPU-only pytest suite.

See [`REPRODUCING.md`](REPRODUCING.md) for the full environment setup and end-to-end
reproduction walkthrough.

## Citation

```bibtex
  title     = {Stoicheia: A Character-Level Masked-Diffusion Model for Ancient Greek},
  year      = {2026},
  note      = {citation to be finalized on publication}
}
```

## License

Apache 2.0 (see `LICENSE`). Vendored/submoduled third-party code (DeepMind's Ithaca
baseline, `ithaca_upstream/`) retains its own Apache 2.0 license — see `NOTICE`.
