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

Not released as a model: the authorship-attribution / hermeneutic probes, which are
designed to be fine-tuned per-task rather than distributed as a general-purpose model
(see the paper's Discussion section).

## Datasets

| dataset | what it is |
|---|---|
| [`anonymous-stoicheia/AncientGreek`](https://huggingface.co/datasets/anonymous-stoicheia/AncientGreek) | main pretraining corpus (gold + silver tiers) |
| [`anonymous-stoicheia/SyntheticAncientGreek-CorpusCorporum`](https://huggingface.co/datasets/anonymous-stoicheia/SyntheticAncientGreek-CorpusCorporum) | bronze synthetic augmentation tier |
| [`anonymous-stoicheia/Inscriptions_2`](https://huggingface.co/datasets/anonymous-stoicheia/Inscriptions_2) | PHI inscriptions used for the 10-fold split and restoration fine-tuning |
| [`anonymous-stoicheia/Stoicheia-silver-lemma`](https://huggingface.co/datasets/anonymous-stoicheia/Stoicheia-silver-lemma) | silver lemma-warmup data for the tagger (new with this release) |

Not re-released (already public elsewhere, cited not duplicated): the *Norma*
macronization/scansion benchmark (GitHub), the OGA/AGDT treebank
(Celano's own repository).
