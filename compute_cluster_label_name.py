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

from collections import defaultdict

# Assume:
# df        = your pandas DataFrame with columns: tokens, r, h, h_id, h_count, h_start, h_end, t, t_id, t_count, t_start, t_end
# labels    = list of cluster indices, len(labels) == len(df)

# TODO
#   Simple get tokens between head and tail NE
#   How can I use token of MASKED embedding?

def create_relphrase_tokens(instance):
    tokens = instance['tokens']
    rel_start = instance['h_end']
    rel_end = instance['t_start']
    rel_phrase = tokens[rel_start+1:rel_end]
    if rel_phrase == []:
        rel_phrase = tokens[instance['t_end']+1:instance['h_start']]
    return rel_phrase

def create_cluster_label(relphrases):
    return relphrases

def predict_cluster_labels(data_frame: pd.DataFrame, predicted_labels):
    # --- 1. Attach cluster labels to the DataFrame ---
    data_frame['cluster'] = predicted_labels

    # --- 2. Group instances by cluster index ---
    clusters = defaultdict(list)
    for _, row in data_frame.iterrows():
        clusters[row['cluster']].append(row)
    # clusters is now a dict: { cluster_idx: [row, row, ...], ... }

    # --- 3. Iterate over clusters and access tokens ---
    for cluster_idx, group_df in data_frame.groupby('cluster'):
        # --- Per-instance/sentence processing ---
        collected_tokens = []  # will gather all "created" tokens across instances in this cluster

        i = 0
        for _, inst in group_df.iterrows():

            print(f"Relation {i}: {inst['r']}({inst['h']} , {inst['t']} )")
            #print(f"\t\tTokens: {' '.join(inst['tokens'])}")
            i += 1
            # Apply your per-sentence method (e.g. get embeddings, mask, return selected tokens)
            created_tokens = create_relphrase_tokens(inst)
                # returns a list of tokens (the masked/selected ones)
            collected_tokens.append(created_tokens)  # or .append() if you want per-instance lists

        # --- After processing all instances in the cluster ---
        cluster_label_name = create_cluster_label(collected_tokens)
        print(f"Cluster {cluster_idx}:")
        for i in range(len(cluster_label_name)):
            print(f"\t{i}: {' '.join(cluster_label_name[i])}")


