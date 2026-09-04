# PromptORE – A Novel Approach Towards Fully Unsupervised Relation Extraction

Code for the CIKM'22 paper [PromptORE – A Novel Approach Towards Fully Unsupervised Relation Extraction](https://doi.org/10.1145/3511808.3557422).

We hope PromptORE will participate in improving Unsupervised Relation Extraction.

## Introduction

Unsupervised Relation Extraction (RE) aims to identify relations between entities in text, without having access to labeled data during training. This setting is particularly relevant for domain specific RE where no annotated dataset is available and for open-domain RE where the types of relations are *a priori* unknown.

Although recent approaches achieve promising results, they heavily depend on hyperparameters whose tuning would most often require labeled data. To mitigate the reliance on hyperparameters, we propose **PromptORE**, a "Prompt-based Open Relation Extraction" model. We adapt the novel prompt-tuning paradigm to work in an unsupervised setting, and use it to embed sentences expressing a relation. We then cluster these embeddings to discover candidate relations, and we experiment different strategies to automatically estimate an adequate number of clusters. To the best of our knowledge, PromptORE is the first unsupervised RE model that does not need hyperparameter tuning.

Results on three general and specific domain datasets show that PromptORE consistently outperforms state-of-the-art models with a relative gain of more than 40% in B3, V-measure and ARI. Qualitative analysis also indicates PromptORE’s ability to identify semantically coherent clusters that are very close to true relations.

## Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/). The project requires Python 3.10+.

```bash
uv sync
```

This installs torch, transformers, accelerate, pandas, scikit-learn, yellowbrick and tqdm into a
local `.venv`, pinned via `uv.lock`. Run any command below with `uv run ...` (or `source .venv/bin/activate`
first).

**GPU note:** `pyproject.toml` pins `torch==2.6.0` from the PyTorch `cu124` wheel index rather than the
default PyPI build. This is deliberate: current default PyPI torch wheels (cu130+) dropped kernels for
pre-Turing GPUs (compute capability < 7.5), which breaks on Pascal-generation cards (e.g. GTX 10xx). The
cu124 build still ships Pascal kernels and remains forward-compatible with newer GPUs (Ampere/Ada/Hopper).
If you only ever run on Turing-or-newer hardware, you can drop the `[tool.uv.sources]`/`[[tool.uv.index]]`
override in `pyproject.toml` and use a plain `torch>=2.4.0` from PyPI instead.

To use 4-bit/8-bit quantization for causal LMs (via `bitsandbytes`, Linux/NVIDIA only):

```bash
uv sync --extra quantization
```

## Running

The source code is specifically designed to work with the FewRel dataset [[1]](#cite-1) [[2]](#cite-2). To have more details on FewRel, please refer to <https://github.com/thunlp/FewRel>. It also supports TACRED and MedDistant19 via `--ds-name`; see `config/*.json` for examples.

### Command Line Interface

PromptORE has the following parameters:
* `--seed=[SEED]`. Random state.
* `--n-rel=[K]`. Number of cluster, if the user knows it in advance.
* `--auto-n-rel`. Activates the estimation of the number of clusters using the elbow rule. *Mutually exclusive with `--n-rel`*.
* `--min-n-rel=[K]`. Only if `--auto-n-rel` is activated. Minimum number of clusters to test.
* `--max-n-rel=[K]`. Only if `--auto-n-rel` is activated. Maximum number of clusters to test.
* `--step-n-rel=[K]`. Only if `--auto-n-rel` is activated. Step to test clusters.
* `--max-len=[LEN]`. Maximum number of tokens in the instances (reasonable values are `fewrel=150, fewrel_nyt=500, fewrel_pubmed=250`).
* `files [FILE1] [FILE2] ...`. FewRel files to load for evaluation. All the files will be concatenated and the metrics aggregated.
* `--model-name=[NAME]`. HF model name or local path. Defaults to `bert-base-uncased`. Any encoder-MLM
  model (BERT, RoBERTa, ModernBERT, ...) or causal LM (OLMo, Qwen, GPT-2, ...) is supported.
* `--model-type=[encoder_mlm|causal_lm]`. Override automatic model-family detection, if needed.
* `--batch-size=[N]`. Embedding computation batch size. Defaults to 256; lower it substantially (e.g. 8-16)
  for multi-billion-parameter causal LMs.
* `--quantization=[4bit|8bit]`. Causal-LM only. Loads the model with `bitsandbytes` 4-bit/8-bit
  quantization to reduce memory. Requires `uv sync --extra quantization`.
* `--prompt-template=[TEMPLATE]`. Authorized parameters are `{sent} {e1} {e2}`, plus `{mask}` for
  encoder-MLM models (required there, forbidden for causal-LM templates). Defaults to a
  family-appropriate template if omitted (see `ore_models.DEFAULT_TEMPLATES`).
* `--device=[DEVICE]`. Force a specific device (e.g. `cpu`, `cuda:0`). By default the model is loaded with
  `accelerate`'s `device_map="auto"`, which automatically shards it across all visible GPUs (falling back
  to CPU) — this is what avoids out-of-memory errors when loading large models on memory-constrained GPUs.

### Supported model families

Encoder-MLM (unchanged default; same [MASK]-position embedding strategy as the original paper):
```bash
uv run python promptore.py --config config/fewrel_config.json --n-inst 100
uv run python promptore.py --config config/fewrel_config.json --model-name roberta-base --n-inst 100
```

Causal LM (last-token hidden state of a completion-style prompt):
```bash
uv run python promptore.py --config config/fewrel_config.json \
    --model-name allenai/OLMo-1B-hf --model-type causal_lm --batch-size 16 --n-inst 100
```

Causal LM, quantized (see `config/qwen3_quantized_fewrel_config.json`):
```bash
uv sync --extra quantization
uv run python promptore.py --config config/qwen3_quantized_fewrel_config.json --n-inst 100
```

### Batch experiments and results table

`experiments/run_experiments.py` runs a batch of PromptORE configurations (different datasets, models,
prompts) defined in a single JSON spec file, and writes a results JSON alongside it. See
`experiments/example_spec.json` for the format: each entry references a `dataset_config` (one of the
`config/*.json` files, used for its defaults) plus any overrides (`model_name`, `model_type`, `batch_size`,
`quantization`, `prompt_template`, `n_inst`, `seed`, ...). Consecutive runs sharing the same model are not
reloaded, and a dataset is only parsed once per `dataset_config` even across many prompt variants, so
grouping a prompt sweep for one model together in the spec is significantly faster.

```bash
uv run python experiments/run_experiments.py --spec experiments/example_spec.json --dry-run  # validate only
uv run python experiments/run_experiments.py --spec experiments/example_spec.json
```

`experiments/render_results_table.py` reads that results JSON and renders a compact metrics table image
(B³ precision/recall/F1, V-measure homogeneity/completeness/F1, ARI — grouped column headers, best value
per column bolded, second-best underlined, one block per dataset), in the style commonly used for OpenRE
papers:

```bash
uv run python experiments/render_results_table.py --results experiments/example_spec_results.json
```

When sweeping several prompts per model, give each run a short `prompt_id` (`"default"`, `"alt_1"`,
`"alt_2"`, ...) in the spec — see `experiments/example_spec.json`. The table's "Run" column then shows
just that short id (dataset and model already have their own columns), and a legend box below the table
maps each id to its full template text. Reuse the same `prompt_id` with the *same* template text across
encoder-MLM models (bert/roberta/ModernBERT templates are already `{mask}`-portable, so this works
directly); causal-LM templates are structurally different, so give them their own id namespace (e.g.
`causal_default`, `causal_alt_1`) to avoid one id mapping to two different templates in the legend. If
`prompt_id` is omitted, it falls back to the run's `id` with the dataset suffix stripped.

### Clustering knowing *k*

For FewRel
```bash
uv run python promptore.py --seed=0 --n-rel=80 --max-len=150 --files "<path-to-fewrel>/train_wiki.json" "<path-to-fewrel>/val_wiki.json"
```

For FewRel NYT
```bash
uv run python promptore.py --seed=0 --n-rel=25 --max-len=500 --files "<path-to-fewrel>/val_nyt.json"
```

For FewRel PubMed
```bash
uv run python promptore.py --seed=0 --n-rel=10 --max-len=250 --files "<path-to-fewrel>/val_pubmed.json"
```

### Estimating the number of clusters with the Elbow Rule

For FewRel
```bash
uv run python promptore.py --seed=0 --auto-n-rel --min-n-rel=10 --max-n-rel=300 --step-n-rel=5 --max-len=150 --files "<path-to-fewrel>/train_wiki.json" "<path-to-fewrel>/val_wiki.json"
```

For FewRel NYT
```bash
uv run python promptore.py --seed=0 --auto-n-rel --min-n-rel=2 --max-n-rel=100 --step-n-rel=2 --max-len=500 --files "<path-to-fewrel>/val_nyt.json"
```

For FewRel PubMed
```bash
uv run python promptore.py --seed=0 --auto-n-rel --min-n-rel=2 --max-n-rel=100 --step-n-rel=2 --max-len=250 --files "<path-to-fewrel>/val_pubmed.json"
```

## License

The source code of PromptORE is licensed under the GPLv3 License. For more details, please refer to the [LICENSE.md file](LICENSE.md).

```
PromptORE
Copyright (C) 2022-2023 Alteca.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```

## Contact

If you have questions using PromptORE, please e-mail us at pygenest@alteca.fr.

## Citation

If you make use of this code in your work, please kindly cite the following paper:

<div class="csl-entry">Genest, Pierre-Yves, Pierre-Edouard Portier, Elöd Egyed-Zsigmond, and Laurent-Walter Goix. “PromptORE - A Novel Approach Towards Fully Unsupervised Relation Extraction.” In <i>Proceedings of the 31st ACM International Conference on Information and Knowledge Management</i>, 11. Atlanta, USA: ACM, 2022. <a href="https://doi.org/10.1145/3511808.3557422">https://doi.org/10.1145/3511808.3557422</a>.</div>

<br/>

```bibtex
@inproceedings{10.1145/3511808.3557422,
    author = {Genest, Pierre-Yves and Portier, Pierre-Edouard and Egyed-Zsigmond, El\"{o}d and Goix, Laurent-Walter},
    title = {PromptORE - A Novel Approach Towards Fully Unsupervised Relation Extraction},
    year = {2022},
    isbn = {9781450392365},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3511808.3557422},
    doi = {10.1145/3511808.3557422},
    booktitle = {Proceedings of the 31st ACM International Conference on Information &amp; Knowledge Management},
    pages = {561–571},
    numpages = {11},
    location = {Atlanta, GA, USA},
    series = {CIKM '22}
}
```
  
## References

<div class="csl-entry"><a name="cite-1"></a><b>[1]</b> Han, Xu, Hao Zhu, Pengfei Yu, Ziyun Wang, Yuan Yao, Zhiyuan Liu, and Maosong Sun. “Fewrel: A Large-Scale Supervised Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation.” In <i>Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing</i>, 4803–9. Brussels, Belgium: Association for Computational Linguistics, 2018. <a href="https://doi.org/10.18653/v1/d18-1514">https://doi.org/10.18653/v1/d18-1514</a>.</div>

<div class="csl-entry"><a name="cite-2"></a><b>[2]</b> Gao, Tianyu, Xu Han, Hao Zhu, Zhiyuan Liu, Peng Li, Maosong Sun, and Jie Zhou. “Fewrel 2.0: Towards More Challenging Few-Shot Relation Classification.” In <i>Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and 9th International Joint Conference on Natural Language Processing</i>, 6250–55. Hong Kong, China: Association for Computational Linguistics, 2019. <a href="https://doi.org/10.18653/v1/d19-1649">https://doi.org/10.18653/v1/d19-1649</a>.</div>
  