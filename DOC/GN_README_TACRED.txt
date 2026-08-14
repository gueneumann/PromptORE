GN: 10.07.2026:

Used TACRED with promptore

1. load and save locally data set
From XMAN:  /local/data/OpenIE/TACRED/data/json/testix.json
From MAC:   /Users/gune00/dataRelationExtraction/TACRED/data/json/testix.json

Size of different data sets: with non-relations
train.json:
    Max len: 96, number relations: 42, number of instances: 68124
    Relation: 'no_relation', Instances: 55112
dev.json:
    Max len: 95, number relations: 42, number of instances: 22631
    Relation: 'no_relation', Instances: 17195
test.json:
    Max len: 96, number relations: 42, number of instances: 15509
    Relation: 'no_relation', Instances: 12184

Class distribution for test.json

Number of different relations: 42
Relation: 'no_relation', Instances: 12184
Relation: 'org:alternate_names', Instances: 213
Relation: 'org:city_of_headquarters', Instances: 82
Relation: 'org:country_of_headquarters', Instances: 108
Relation: 'org:dissolved', Instances: 2
Relation: 'org:founded', Instances: 37
Relation: 'org:founded_by', Instances: 68
Relation: 'org:member_of', Instances: 18
Relation: 'org:members', Instances: 31
Relation: 'org:number_of_employees/members', Instances: 19
Relation: 'org:parents', Instances: 62
Relation: 'org:political/religious_affiliation', Instances: 10
Relation: 'org:shareholders', Instances: 13
Relation: 'org:stateorprovince_of_headquarters', Instances: 51
Relation: 'org:subsidiaries', Instances: 44
Relation: 'org:top_members/employees', Instances: 346
Relation: 'org:website', Instances: 26
Relation: 'per:age', Instances: 200
Relation: 'per:alternate_names', Instances: 11
Relation: 'per:cause_of_death', Instances: 52
Relation: 'per:charges', Instances: 103
Relation: 'per:children', Instances: 37
Relation: 'per:cities_of_residence', Instances: 189
Relation: 'per:city_of_birth', Instances: 5
Relation: 'per:city_of_death', Instances: 28
Relation: 'per:countries_of_residence', Instances: 148
Relation: 'per:country_of_birth', Instances: 5
Relation: 'per:country_of_death', Instances: 9
Relation: 'per:date_of_birth', Instances: 9
Relation: 'per:date_of_death', Instances: 54
Relation: 'per:employee_of', Instances: 264
Relation: 'per:origin', Instances: 132
Relation: 'per:other_family', Instances: 60
Relation: 'per:parents', Instances: 88
Relation: 'per:religion', Instances: 47
Relation: 'per:schools_attended', Instances: 30
Relation: 'per:siblings', Instances: 55
Relation: 'per:spouse', Instances: 66
Relation: 'per:stateorprovince_of_birth', Instances: 8
Relation: 'per:stateorprovince_of_death', Instances: 14
Relation: 'per:stateorprovinces_of_residence', Instances: 81
Relation: 'per:title', Instances: 500

Test:

time python promptore.py --seed=0 --n-rel=42 --max-len=500 \
    --files "/local/data/OpenIE/TACRED/data/json/test.json" \
    --ds-name tacred

B3: prec=0.7059681932007787 rec=0.1254134172197281 f1=0.21298975692513827
V-measure: hom=0.4665257535066234 comp=0.1535434150431394 f1=0.23104505443002715
ARI=0.008301868206657393

# TODO testing without no_relations; note: reduces k from 42 to 41
time python promptore.py --seed=0 --n-rel=41 --max-len=500 \
    --files "/local/data/OpenIE/TACRED/data/json/test.json" \
    --ds-name tacred --ignore-na

B3: prec=0.5380988399786907 rec=0.31657878085219643 f1=0.3986314151359812
V-measure: hom=0.6754247461103918 comp=0.5875117790814606 f1=0.6284084533269465
ARI=0.31858735580706143

Max len: 92, number relations: 41, number of instances: 5436
time python promptore.py --seed=0 --n-rel=41 --max-len=500 \
    --files "/local/data/OpenIE/TACRED/data/json/dev.json" \
    --ds-name tacred --ignore-na

B3: prec=0.544753657448458 rec=0.3316677735554309 f1=0.41230674264811823
V-measure: hom=0.6686051092226706 comp=0.5906663478896775 f1=0.627223837742715
ARI=0.30350767352659985

TODO
Problem: non-relations

How handled by others: basically remove it; IMHO methods are not completely zero-shot
-> TODO test similar setting also with promptORE

Wang et al., 2022: MatchPrompt
- remove no-relation types
- first 0-30 relation instances are used as source domain
- last 31-40 relation instances are used as open domain
- if I read it correctly, they select 15 % random instances from open domain 31-40 relations
- Results:
    B3: 83.00, V: 84.5, ARI: 75.3

Zhao et al., 2023: ASCORE, Zhao et AL., 2021 RoCORE
- same as above
- Results:
    ASCORE: B3: 78.00, V: 83.1, ARI: 78.1
    RoCORE: B3: 86.00, V: 88.8, ARI: 81,2

Jamal et al., 2025: UOREX
- same as above; compare: SelfCORE, RSN, RoCORE, KNORD
-   UOREX: B3: 83.90, V: 86,3, ARI: 80,6

Hogan et al., 2023: KNORD, semi-supervised
- have results for non-relations
- separate classes into known (high frequent) and novel (low frequent) classes
    top 50% most frequent classes as known labels
- they fine-tune BERT with novel classes only using similar prompt as promptORE
- they mix 15% known instances to the novel instance which defines test set
- they only use F!-micro measure (B3?)



