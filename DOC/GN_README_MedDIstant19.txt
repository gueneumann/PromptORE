GN: 29.04.2026:

Used MedDistant19 with promptore

1. load and save locally data set
From XMAN:  /local/data/OpenIE/med_distant19/MedDistant19_COLING/opennre_format
From MAC:   /Users/gune00/data/RelationExtraction/med_distant19/MedDistant19_COLING/opennre_format

Data format:

    {
    "text": "The paper 's aim is to review dentin hypersensitivity ( DHS ) , discussing pain mechanisms and aetiology .",
    "h": {"id": "C0030193",
            "pos": [75, 79],
            "name": "pain"},
    "t": {"id": "C0011432",
            "pos": [30, 53],
            "name": "dentin hypersensitivity"},
    "relation": "NA"
    }

Size of different data sets: with non-relations:

med_distant19_train.txt:
    Max len: 249, number relations: 22, number of instances: 450071
    Relation: 'NA', Instances: 405252
med_distant19_val.txt:
    Max len: 128, number relations: 22, number of instances: 39434
    Relation: 'NA', Instances: 35954
med_distant19_test.txt:
    Max len: 150, number relations: 22, number of instances: 91568
    Relation: 'NA', Instances: 83439

Class distribution for med_distant19_val.txt

Number of different relations: 22
    Relation: 'NA', Instances: 35954
    Relation: 'active_ingredient_of', Instances: 8
    Relation: 'associated_finding_of', Instances: 9
    Relation: 'associated_morphology_of', Instances: 413
    Relation: 'causative_agent_of', Instances: 78
    Relation: 'cause_of', Instances: 28
    Relation: 'component_of', Instances: 145
    Relation: 'direct_device_of', Instances: 5
    Relation: 'direct_morphology_of', Instances: 13
    Relation: 'direct_procedure_site_of', Instances: 307
    Relation: 'direct_substance_of', Instances: 151
    Relation: 'finding_site_of', Instances: 1479
    Relation: 'focus_of', Instances: 40
    Relation: 'indirect_procedure_site_of', Instances: 15
    Relation: 'interpretation_of', Instances: 97
    Relation: 'interprets', Instances: 377
    Relation: 'is_modification_of', Instances: 55
    Relation: 'method_of', Instances: 143
    Relation: 'occurs_after', Instances: 41
    Relation: 'procedure_site_of', Instances: 45
    Relation: 'uses_device', Instances: 26
    Relation: 'uses_substance', Instances: 5

DONE TODO NOTE: I am using character span for entities!
    I think, I need token index here -> see promptore.py line 197
    So, how can  map character position to token id
    Method:
        assume tokens are separated by blank
        get start-end pos substring and split string -> give number t_n of tokens for entity name
        get prefix upto end-pos -> split string -> get token id of last token of name
        t_last - t_n + 1 -> token id of first token of name

3. do testing

    time python promptore.py --seed=0 --n-rel=22 --max-len=500 \
    --files "/local/data/OpenIE/med_distant19/MedDistant19_COLING/opennre_format/med_distant19_val.txt" \
    --ds-name meddistant19

    B3: prec=0.8496110721022327 rec=0.08600072612189964 f1=0.15619120934704991
    V-measure: hom=0.29185042882134715 comp=0.04669775173386966 f1=0.0805129647790255
    ARI=-0.0012832535337836937

    real	18m21,738s
    user	20m0,483s
    sys	0m8,892s

    # TODO testing without NA; note: reduces k from 22 to 21
    time python promptore.py --seed=0 --n-rel=21 --max-len=500 \
    --files "/local/data/OpenIE/med_distant19/MedDistant19_COLING/opennre_format/med_distant19_val.txt" \
    --ds-name meddistant19 --ignore-na

    B3: prec=0.6693121479657362 rec=0.33673550522687934 f1=0.44805266149073564
    V-measure: hom=0.6639628109652105 comp=0.4729673118496614 f1=0.5524221754154277
    ARI=0.23069664398448278

    real	1m44,421s
    user	1m53,530s
    sys	0m6,023s

    # TODO calling at MacBookPro: takes 26 minutes
    time python promptore.py --seed=0 --n-rel=21 --max-len=500 \
    --files "/Users/gune00/data/RelationExtraction/med_distant19/MedDistant19_COLING/opennre_format/med_distant19_val.txt" \
    --ds-name meddistant19 --ignore-na



TODO how to improve:

Unbalanced group size!

how to use NA relations? remove as done for TACRED




