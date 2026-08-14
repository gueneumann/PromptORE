import json
import pandas as pd
################################################################################
#   PromptORE internal data frame:
# {
#    'tokens': instance['tokens'],
#    'r': relation,
#    'h': instance['h'][0],
#    'h_id': instance['h'][1],
#    'h_count': i_h,
#    'h_start': h_pos[0],
#    'h_end': h_pos[len(h_pos) - 1],
#    't': instance['t'][0],
#    't_id': instance['t'][1],
#    't_count': i_t,
#    't_start': t_pos[0],
#    't_end': t_pos[len(t_pos) - 1],
#      }

# -> NOTE: at least for FewRel entity span is on token level

################################################################################
# Dataset FewRel:
# fewrel dataset. Dataset can be downloaded at:
#   https://github.com/thunlp/FewRel/tree/master/data
# Fewrel does not have labelled no-relations
# Source data path: /local/data/OpenIE/FewRel/data
# For each relation, list all instances
# Each instance has:
#   tokens
#   h and t entity: json object with three list element
#       name: list of tokens
#       type string
#       span: [start, end] on token-level
#  Example:
#  "biological_process_involves_gene_product":
#    [{'tokens': ['understanding', 'how', 'ubiquitin', 'e2', 'and', 'e3', 'ligases', 'contribute',
#                   'to', 'controlling', 'mhc', 'ii', 'antigen', 'presentation', 'will', 'shed', 'light', 'on',
#                   'the', 'critical', 'regulatory', 'controls', 'of', 'this', 'important', 'pathway', 'of',
#                   'immunity', '.'],
#       'h': ['mhc', 'C4049595', [[10]]],
#       't': ['antigen presentation', 'C0206431', [[12, 13]]]}
#     ]
################################################################################

def transform_fewrel_object (instance, relation,
                             i_h, h_pos,
                             i_t, t_pos ):
    return {'tokens': instance['tokens'],
            'r': relation,
            'h': instance['h'][0],
            'h_id': instance['h'][1],
            'h_count': i_h,
            'h_start': h_pos[0],
            # This is the token id of the last token, because token id starts from 0
            'h_end': h_pos[len(h_pos) - 1],
            't': instance['t'][0],
            't_id': instance['t'][1],
            't_count': i_t,
            't_start': t_pos[0],
            't_end': t_pos[len(t_pos) - 1]
            }

def parse_fewrel(path: str, expand: bool = False) -> pd.DataFrame:
    with open(path, 'r', encoding='utf-8') as file:
        fewrel_json = json.load(file)

    data_tuples = []
    # Fewrel: for each relaion, lists all instaances
    for relation, instances in fewrel_json.items():
        for instance in instances:
            # each instance can have multiple (head, tail) pairs
            # if not expand then only use first pair
            for i_h, h_pos in enumerate(instance['h'][2]):
                for i_t, t_pos in enumerate(instance['t'][2]):
                    data_tuples.append(
                        transform_fewrel_object(instance, relation,
                                                i_h, h_pos, i_t, t_pos))
                    if not expand:
                        break
                if not expand:
                    break
    print(data_tuples[0])
    return pd.DataFrame(data_tuples)

################################################################################
# Dataset MedDistant19
# Source data path: /local/data/OpenIE/med_distant19/MedDistant19_COLING/opennre_format
#  - Enumeration of all instances
#  - relation name is part of instance
#       NO RELATION as NA
#  - each instance has:
#       token string, relation string, h and t entities
#  - h and t entity: json object of three elements
#    name: string of tokens
#    id: type
#    span: [start,end] on character level
# Example:
# {
#     "text": "The paper 's aim is to review dentin hypersensitivity ( DHS ) , discussing pain mechanisms and aetiology .",
#     "h": {"id": "C0030193",
#             "pos": [75, 79],
#             "name": "pain"},
#     "t": {"id": "C0011432",
#             "pos": [30, 53],
#             "name": "dentin hypersensitivity"},
#     "relation": "NA"
#     }
#
#   TODO NOTE: meddistant19 uses character span for entities!
#       But I need token index here -> see promptore.py line 197
#       So, how can  map character position to token id -> GN_README_MedDIstant19.txt
################################################################################

# Maps char span position of entity mention to token span position
# Assumes tokens are separated by a single blank
def get_entity_token_start_end_from_char_span (text, char_start, char_end):
    prefix_char_end = text[:char_end]
    mention = prefix_char_end[char_start:]
    end_token_id = len(prefix_char_end.split(' ')) - 1
    start_token_id = end_token_id - len(mention.split(' ')) + 1
    return start_token_id, end_token_id

def transform_meddistant19_object (instance):
    h = instance['h']
    t = instance['t']
    text_string = instance['text']
    h_start, h_end = get_entity_token_start_end_from_char_span(text_string,h['pos'][0],h['pos'][1])
    t_start, t_end = get_entity_token_start_end_from_char_span(text_string,t['pos'][0],t['pos'][1])
    return {'text': text_string,
            'tokens': text_string.split(' '),
            'r': instance['relation'],
            'h': h['name'],
            'h_id': h['id'],
            'h_count': len(h['name'].split(' ')),
            # This should be id of start token
            'h_start': h_start,
            'h_end': h_end,
            't': t['name'],
            't_id': t['id'],
            't_count': len(t['name'].split(' ')),
            't_start': t_start,
            't_end': t_end
            }

def parse_meddistant19(path: str, expand: bool = False, ignore_na: bool = False,
                       NOREL_LABEL = 'NA') -> pd.DataFrame:
    data_tuples = []

    with open(path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, start=1):
            line = line.strip()
            if not line:  # skip blank lines
                continue
            try:
                instance = json.loads(line)
                if ignore_na:
                    if instance['relation'] != NOREL_LABEL:
                        data_tuples.append(transform_meddistant19_object(instance))
                else:
                    data_tuples.append(transform_meddistant19_object(instance))
            except json.JSONDecodeError as e:
                print(f"Skipping line {line_num}: {e}")
    print(data_tuples[0])
    return pd.DataFrame(data_tuples)

################################################################################
# Dataset TACRED
# Source data path: /local/data/OpenIE/TACRED/data/json
#  - List of all instances
#  - relation name is part of instance
#       NO RELATION as "no_relation"
# Each instance has:
#   tokens: list of tokens
#   subj_start, subj_end: token ids at token level
#   obj_start, object_end: token ids at token level
#   subj_type, obj_type: NE type
# Example:
# {"id": "098f665fb966708cfcd2", "docid": "eng-NG-31-101172-8859554",
# "relation": "no_relation",
# "token": ["He", "has", "served", "as", "a", "pol
# icy", "aide", "to", "the", "late", "U.S.", "Senator", "Alan", "Cranston", ",", "as", ...,  "."],
# "subj_start": 33, "subj_end": 36,
# "obj_start": 43, "obj_end": 45,
# "subj_type": "ORGANIZATION", "obj_type": "ORGANIZATION", ...}
################################################################################

def transform_tacred_object (instance):
    return {'tokens': instance['token'],
            'r': instance['relation'],
            'h': ' '.join(instance['token'][instance['subj_start']:instance['subj_end']+1]),
            'h_id': instance['subj_type'],
            'h_count': 0,
            'h_start': instance['subj_start'],
            'h_end': instance['subj_end'],
            't': ' '.join(instance['token'][instance['obj_start']:instance['obj_end']+1]),
            't_id': instance['obj_type'],
            't_count': 0,
            't_start': instance['obj_start'],
            't_end': instance['obj_end'],
            }

def parse_tacred(path: str, expand: bool = False, ignore_na: bool = False,
                 NOREL_LABEL = 'no_relation') -> pd.DataFrame:
    with open(path, 'r', encoding='utf-8') as file:
        data_json = json.load(file)

    data_tuples = []
    for instance in data_json:
        if ignore_na:
            if instance['relation'] != NOREL_LABEL:
                data_tuples.append(transform_tacred_object(instance))
        else:
            data_tuples.append(transform_tacred_object(instance))
    print(data_tuples[0])
    return pd.DataFrame(data_tuples)


################################################################################
# End specific data set processing
################################################################################

################################################################################
# Dataset Wrapper for uniform call
################################################################################

def parse_dataset(path: str, expand: bool = False,
                  ds_name: str ="fewrel", ignore_na: bool = False) -> pd.DataFrame:
    normalize_ds_name = ds_name.lower()
    if normalize_ds_name == "fewrel":
        return parse_fewrel(path, expand)
    elif normalize_ds_name == "tacred":
        return parse_tacred(path, expand, ignore_na)
    elif normalize_ds_name == "meddistant19":
        return parse_meddistant19(path, expand, ignore_na)
    else:
        return exit(f"Data set not known: {ds_name}")

################################################################################
# Computing max_length and (relation, instances_idx) of given panda data frame
################################################################################

def get_data_frame_statistcs (data_frame, gn_debug):
    # 1. Length of the maximum token list
    max_token_length = data_frame['tokens'].apply(len).max()
    if gn_debug:
        print(f"Max token list length: {max_token_length}")
        print(f"Total number of instances: {len(data_frame)}")

    # 2. List of all different relations with their instance indices
    relation_groups = (
        data_frame.groupby('r')
        .apply(lambda g: g.index.tolist())
        .reset_index()
        .rename(columns={0: 'instance_indices'})
    )

    # Result: list of (relation, [indices]) tuples
    relation_instance_list = list(zip(relation_groups['r'], relation_groups['instance_indices']))

    if gn_debug:
        print(f"Number of different relations: {len(relation_instance_list)}")
        for relation, indices in relation_instance_list:
            print(f"Relation: {relation!r}, Instances: {len(indices)}")

    return max_token_length, len(relation_instance_list)

################################################################################
# Get subset of instances, e.g., N instances per group
################################################################################

def get_dataset_subset(data_frame: pd.DataFrame, N=10, random=False, seed=42):
    if random:
        return (data_frame.groupby('r', group_keys=False)
                .apply(lambda g: g.sample(min(N, len(g)), random_state=seed)))
    else:
        return (data_frame.groupby('r', group_keys=False)
                .apply(lambda g: g.head(N)))

