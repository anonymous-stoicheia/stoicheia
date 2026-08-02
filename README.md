# Stoicheia

A character-level masked-diffusion Transformer for Ancient Greek, pretrained on an
open, revision-pinned corpus and released as eleven decontaminated checkpoints (ten
rotated literary folds + one documentary-clean model), fine-tuned for restoration of
damaged inscriptions/papyri, morphosyntactic tagging and dependency parsing, and
macronization/metrical scansion.

This repository is the training/evaluation code. The pretrained and fine-tuned model
weights are on the HuggingFace Hub — see [`MODEL_CARDS_INDEX.md`](MODEL_CARDS_INDEX.md)
for the full list, or jump straight to
[`anonymous-stoicheia/Stoicheia-doc_clean`](https://huggingface.co/anonymous-stoicheia/Stoicheia-doc_clean)
(the flagship backbone) or
[`anonymous-stoicheia/Stoicheia-restoration-test3`](https://huggingface.co/anonymous-stoicheia/Stoicheia-restoration-test3) (or any of the ten digit-rotation checkpoints) /
[`-tagger-parser`](https://huggingface.co/anonymous-stoicheia/Stoicheia-tagger-parser) for a
ready-to-use downstream model (or [`-meter`](https://huggingface.co/anonymous-stoicheia/Stoicheia-meter) for
macronization and scansion). All model repos are public: weights ship as `model.safetensors` with a `config.json`,
loadable directly through `AutoModel.from_pretrained(..., trust_remote_code=True)`.


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anonymous-stoicheia/stoicheia/blob/main/Stoicheia_demo.ipynb)

Run everything in the browser: [`Stoicheia_demo.ipynb`](Stoicheia_demo.ipynb) restores a lacuna of
unknown width, picks the checkpoint that has provably never read your document, tags and parses a
verse of Homer, macronizes and scans a line, and scores the macronizer on the benchmark.

## Quickstart (no training required)

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("anonymous-stoicheia/Stoicheia-doc_clean", trust_remote_code=True)

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
@misc{stoicheia2026,
  title  = {Stoicheia: Character-Level Masked Diffusion for Ancient Greek Textual
            Restoration, Parsing, and Metrical Scansion},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review; citation to be finalized on publication}
}
```

## License

Apache 2.0 (see `LICENSE`). External baselines (DeepMind's Ithaca and predictingthepast releases) are
downloaded separately from their own repositories and retain their own licenses — see
`NOTICE` and `REPRODUCING.md`.
