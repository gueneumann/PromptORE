Used FewRel with promptore

1. load and save locally data set
From XMAN:  /local/data/OpenIE/FewRel/data
From MAC:   /Users/gune00/dataRelationExtraction/FewRel/data

Size of different data sets: with non-relations:
train_wiki.json:
    Max len: 36, number relations: 64, number of instances: 44800
    700 instances per class; no no-relations
val_wiki.json:
    Max len: 36, number relations: 16, number of instances: 11200
    700 instances per class; no no-relations
val_nyt.json:
    Max len: 271, number relations: 25, number of instances: 2500
    100 instances per class; no no-relations
val_semeval.json:
    Max len: 97, number relations: 17, number of instances: 8851
    NNN instances per class; no no-relations
val_pubmed.json:
    Max len: 145, number relations: 10, number of instances: 1000
    100 instances per class; no no-relations

TODO Testing full FewRel (56,0000 sentences/extractions):
Max len: 36, number relations: 80, number of instances: 56000

Given K = 80
time python promptore.py --seed=0 --n-rel=80 --max-len=150 --files \
"/local/data/OpenIE/FewRel/data/train_wiki.json" "/local/data/OpenIE/FewRel/data/val_wiki.json" --ds-name FewRel

Paper:
B3 = 48.8,  V-measure = 78.071,8,   ARI = 43.4

B3: prec=0.5017297509238209 rec=0.5083061734693878 f1=0.5049965523969856
V-measure: hom=0.7177842642445366 comp=0.7325327832557007 f1=0.7250835336597113
ARI=0.4584557718213026

real	8m57,388s

time python promptore.py --seed=0 --n-rel=16 --max-len=150 --files \
"/local/data/OpenIE/FewRel/data/val_wiki.json" --ds-name FewRel

TODO Testing val-nyt.json (2500 sentences/extractions):
Max len: 271, number relations: 25, number of instances: 2500

time python promptore.py --seed=0 --n-rel=25 --max-len=500 \
--files "/local/data/OpenIE/FewRel/data/val_nyt.json" --ds-name FewRel

Paper:
B3 = 65.2,  V-measure = 78.0,   ARI = 56.9

B3: prec=0.6362412671476179 rec=0.6932240000000001 f1=0.6635114539297959
V-measure: hom=0.7737070467067817 comp=0.806616201200377 f1=0.7898189685978103
ARI=0.5866578812915124

real	1m12,969s

TODO Testing val-pubmed.json (1000 sentences/extractions):
Max len: 145, number relations: 10, number of instances: 1000

time python promptore.py --seed=0 --n-rel=10 --max-len=250 \
--files "/local/data/OpenIE/FewRel/data/val_pubmed.json" --ds-name FewRel

Paper:
B3 = 77.4,  V-measure = 81.1,   ARI = 73.8

B3: prec=0.7616512739097857 rec=0.76042 f1=0.7610351389376624
V-measure: hom=0.8073184613939894 comp=0.8099439673164786 f1=0.8086290831979072
ARI=0.7252604318279915

real	0m22,137s

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

TODO Computing K via Elbow rule:
    NOTE: This takes time ! Because k-means has to be called several times

TODO Testing full FewRel (56,0000 sentences/extractions):

time python promptore.py --seed=0 --auto-n-rel --min-n-rel=10 --max-n-rel=300 --step-n-rel=5 --max-len=150 --files \
"/local/data/OpenIE/FewRel/data/train_wiki.json" \
"/local/data/OpenIE/OpenIE/FewRel/data/val_wiki.json" --ds-name FewRel

Paper:
Estimated n_rel=65
B3 = 49.5,  V-measure = 71.2,   ARI = 42.2
(When running with given k=65, I get similar results)

(Note, below the correct k for FewRel is computed!)
Estimated n_rel=80
Do k-mean clustering
Evaluate
B3: prec=0.5017297509238209 rec=0.5083061734693878 f1=0.5049965523969856
V-measure: hom=0.7177842642445366 comp=0.7325327832557007 f1=0.7250835336597113
ARI=0.4584557718213026

real	253m0,410s

TODO Testing val-nyt.json (2500 sentences/extractions):

time python promptore.py --seed=0 --auto-n-rel --min-n-rel=2 --max-n-rel=100 --step-n-rel=2 --max-len=500 --files \
"/local/data/OpenIE/FewRel/data/val_nyt.json" --ds-name FewRel

Paper:
Estimated n_rel=26
B3 = 64.1,  V-measure = 77.4,   ARI = 56.2

Estimated n_rel=28
Do k-mean clustering
Evaluate
B3: prec=0.6609290708073414 rec=0.642984 f1=0.6518330511110864
V-measure: hom=0.7873540935520276 comp=0.7883598235028937 f1=0.7878566375641427
ARI=0.5814207658672949

real	3m26,696s

TODO Testing val-pubmed.json (1000 sentences/extractions):

time python promptore.py --seed=0 --auto-n-rel --min-n-rel=2 --max-n-rel=100 --step-n-rel=2 --max-len=250 --files \
"/local/data/OpenIE/FewRel/data/val_pubmed.json" --ds-name FewRel

Paper:
Estimated n_rel=10
B3 = 77.4,  V-measure = 81.1,   ARI = 73.8

Estimated n_rel=10
Do k-mean clustering
Evaluate
B3: prec=0.7616512739097857 rec=0.76042 f1=0.7610351389376624
V-measure: hom=0.8073184613939894 comp=0.8099439673164786 f1=0.8086290831979072
ARI=0.7252604318279915

real	1m19,967s