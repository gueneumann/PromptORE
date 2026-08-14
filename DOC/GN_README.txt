conda create -n promptore python=3.8.16
conda activate promptore

python -m pip install -r requirements.txt

cd work/OpenIE/PromptORE

TODO also test other prompts from paper and new templates:
- see promptore.py:381 prompt_template
- prompts:
    P_0: "{e1} [MASK] {e2}"
    Prompt_ORE: "{sent} {e1} [MASK] {e2}"
    P_2: "{sent}. In this sentence {e1} is the [MASK] of {e2}.", e.g.,  Lv et al., 2022

TODO Structure of embedding panda data frame:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1165 entries, 0 to 1164
Data columns (total 6 columns):
 #   Column                Non-Null Count  Dtype
---  ------                --------------  -----
 0   input_tokens          1165 non-null   object
 1   input_attention_mask  1165 non-null   object
 2   input_mask            1165 non-null   int64
 3   output_r              1165 non-null   object
 4   output_label          1165 non-null   int64
 5   embedding             1165 non-null   object
dtypes: int64(2), object(4)
memory usage: 54.7+ KB

TODO understanding and tracing promptORE:

text prompt:
understanding how ubiquitin e2 and e3 ligases contribute to controlling mhc ii antigen presentation \
will shed light on the critical regulatory controls of this important pathway of immunity . \
mhc [MASK] antigen presentation

First raw of panda data frame with relation embedding:
input_tokens            [tensor(101), tensor(4824), tensor(2129), tens...
input_attention_mask    [tensor(1), tensor(1), tensor(1), tensor(1), t...
input_mask                                                             39
output_r                         biological_process_involves_gene_product
output_label                                                            0
embedding               [tensor(-0.4921), tensor(0.0098), tensor(0.273...
Name: 0, dtype: object

TODO Adapt promptore to more data sets: DONE

Data sets:
-   works for FewRel
    - results as in paper
-   works for Meddistant19
    - can run with/without NA relations
-   works for TACRED
    - can run with/without no_relations relations

TODO improve software structure in preparation of rest-api and GUI:

- visualization class:
    visualize results and cluster / gold group alignment
    visualize/bcuped_*.py

- define data class: maybe I can use what I have for GUI and RestAPI
    - loading data based on dataset-label
        - data set label
        - data dir root and then select one or more files
        - special labels: no relation
    - loading with/without no relations
    - number of loaded labels from gold
        -> default K value for clustering, if not given

    - prepare test data following others
        - for TACRED:
            - prepare test data as RoCORE: 15 % random instances from classes 32-40
        - for FewRel:
            - prepare test data as RoCORE: from val_wiki.json use random 100 instances per relation
            - here: see also comment by PromptOre about using FewRel for testing
        - for meddistant19:
            SAME

TODO
DONE
- define model class:
    - following LAOIE
    - modernBert
        - https://huggingface.co/docs/transformers/model_doc/modernbert
        - https://www.answer.ai/posts/2025-02-10-modernbert-instruct.html
        - https://jina.ai/de/news/what-should-we-learn-from-modernbert/
    - clausal models
-> with Claude Code DONE:
    - updated conda to uv
    - update ore_models.py to model adapters:
        - encoder-masked-language-models
        - clausal LM
    - using model-type specific prompt-templates
    - multiple GPU
    - Models tested:
        - bert-base/large
        - roberta-base/large
        - answerdotai/ModernBERT-large or -base
        - gpt2 also with quantization 8bit
        - allenai/OLMo-1B-hf --model-type causal_lm
        - NOT YET: Qwen/Qwen3-4B, quantized: 4bit
            - qwen3_quantized_fewrel_config.json


- define FastAPI or Flask GUI using Claude Code
    https://www.contentful.com/blog/fastapi-vs-flask/


- define class get_relation_embeddings
    - from fresh calling model
    - save panda frame with embeddings
    - if frozen and if saved exists load it

TODO and HIERIX:
- how to predict class label from predicted label?
    DOC/GN_PREDICT_LABEL

TODO inference:
    - compute or load saved clusters
    - for new sentence(s):
        do NER and identify similar clusters

TODO How to improve performance?
    - understand results also compared to others - WHAT DID I MEAN HERE?
    - NOTE: others (ASCORE, SelfORE, RoCORE) do clustering with semi-supervised or pre-trained ORE
        they split data in labeled/unlabeled (test) data; use only 2 classes instead of all
TODO:
    - How to handle no relations -> DOC/GN_NOREL.txt
      - define binary classifier trained on dataset
    - I think handling NOREL must be done before clustering OTHER, because
        otherwise noise of NOREL (cause by its sheer size) destroys clustering of OTHER
        and hence must be done first

TODO:
    - how to handle better cluster methods
    - how to handle different (unbalanced) class distribution
    - how to handle data with class distribution shift



