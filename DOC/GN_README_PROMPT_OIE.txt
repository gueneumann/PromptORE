conda create -n promptore python=3.8.16
conda activate promptore

python -m pip install -r requirements.txt

cd work/OpenIE/PromptORE

Testing code in promptoie.py:

    time python promptoie.py --seed=0 --n-rel=10 --max-len=250 --files \
    "/home/neumann/work/OpenIE/FewRel/data/val_pubmed.json"

    time python promptoie.py --seed=0 --n-rel=25 --max-len=500 --files "../FewRel/data/val_nyt.json"

- try printing tokens for [MASK] token:

    - works, NOTE: surface form not necessarily occurs in input text
    - to do so, I need BertForMaskedLM and smaller batch size

- visualize attention matrix like in
    /home/neumann/work/DeepEx/useful_code_to_test/test_nfm_across_attentions_word_level.py
    ok, also works, but need word level not sub-token level

NOTE Ok, so directly extracting the textual token-level relation phrase does not make sense
    - relation embedding is only a single MASK token
        - but it is a semantic related token not a surface related token
    - unclear how to identify relevant relation token sequence
        so that it could go beyond simple baseline

Eventually better:
    - do clustering using relation embeddings
    - for clusters try to identify common substrings
        - BUT how is this going beyond simple baseline???
    - also means that method is closer to promptORE and can make use of BertModel




