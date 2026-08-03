# Stoicheia — Released Models and Datasets

All models are character-level masked-diffusion Transformers for Ancient Greek (405M
params, `d_model=1024`, depth 32), loadable via
`AutoModel.from_pretrained(repo_id, trust_remote_code=True)`. See each model card for
usage examples specific to that checkpoint's task.

## Pretrained backbones (11)

Ten rotated, work-level-decontaminated literary folds plus one documentary-clean model.
For any passage in the open training corpus, at least one of these eleven has provably
never seen it during pretraining.

| model | decontamination |
|---|---|
| [`anonymous-stoicheia/Stoicheia-doc_clean`](https://huggingface.co/anonymous-stoicheia/Stoicheia-doc_clean) | zero documentary (inscription/papyrus) exposure of any kind |
| [`anonymous-stoicheia/Stoicheia-fold-0`](https://huggingface.co/anonymous-stoicheia/Stoicheia-fold-0) … [`fold-9`](https://huggingface.co/anonymous-stoicheia/Stoicheia-fold-9) | rotated 80/10/10 literary split, fold *k*'s test set unseen by fold *k*'s model |

## Fine-tuned downstream models (4)

All built on `Stoicheia-doc_clean`.

| model | task |
|---|---|
| [`anonymous-stoicheia/Stoicheia-restoration-test0`](https://huggingface.co/anonymous-stoicheia/Stoicheia-restoration-test0) … [`-test9`](https://huggingface.co/anonymous-stoicheia/Stoicheia-restoration-test9) | documentary restoration, ten checkpoints — checkpoint *k* holds out every PHI/TM identifier ending in digit *k*, so every document in the corpus has a model that provably never saw it |
| [`anonymous-stoicheia/Stoicheia-tagger-parser`](https://huggingface.co/anonymous-stoicheia/Stoicheia-tagger-parser) | morphosyntactic tagging (XPOS/UPOS/lemma) + dependency parsing |
| [`anonymous-stoicheia/Stoicheia-meter`](https://huggingface.co/anonymous-stoicheia/Stoicheia-meter) | macronization (vowel length) + metrical scansion, trained jointly |
| [`anonymous-stoicheia/Stoicheia-macronizer`](https://huggingface.co/anonymous-stoicheia/Stoicheia-macronizer) | macronization only — the arm the paper's macronization ablation is measured on |

## Datasets

| dataset | what it is |
|---|---|
| [`anonymous-stoicheia/AncientGreek`](https://huggingface.co/datasets/anonymous-stoicheia/AncientGreek) | pretraining corpus, ~361M words in `pristine` and `repaired` tiers |
| [`anonymous-stoicheia/SyntheticAncientGreek-CorpusCorporum`](https://huggingface.co/datasets/anonymous-stoicheia/SyntheticAncientGreek-CorpusCorporum) | bronze synthetic augmentation tier, machine-translated from Latin |
| [`anonymous-stoicheia/Inscriptions_2`](https://huggingface.co/datasets/anonymous-stoicheia/Inscriptions_2) | PHI inscriptions used for the 10-fold split and restoration fine-tuning |
| [`anonymous-stoicheia/Stoicheia-meter-silver`](https://huggingface.co/datasets/anonymous-stoicheia/Stoicheia-meter-silver) | silver macronization/scansion training data (Hypotactic-derived + constraint-solver mined) |
| [`anonymous-stoicheia/norma`](https://huggingface.co/datasets/anonymous-stoicheia/norma) | mirror of the *Norma Syllabarum Graecarum* benchmark, in the exact split the paper evaluates on |

`AncientGreek` omits the Database of Byzantine Book Epigrams, which is CC BY-NC-SA and so
cannot travel inside a CC BY-SA compilation; `scripts/fetch_dbbe.py` refetches it under its
own terms. Not re-released at all: the OGA/AGDT treebank splits, which come from Celano's
own repository and are cited rather than duplicated.
