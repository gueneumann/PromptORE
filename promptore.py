"""PromptORE

---
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
"""
import argparse, os, json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.kernel_ridge import KernelRidge
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix, adjusted_rand_score, \
    homogeneity_score, completeness_score
from yellowbrick.cluster import KElbowVisualizer
from tqdm.auto import tqdm

import parse_data
import compute_cluster_label_name as labeller
from visualize import bcubed_sankey as bcubed_sankey
from visualize import bcubed_heatmap as bcubed_heatmap_large

import ore_models

################################################################################
# Metrics
################################################################################

## Simple version just for clarity and to double check
def bcubed_loop(labels_true, labels_pred):
    C = contingency_matrix(labels_true, labels_pred)

    row_sums = C.sum(axis=1)  # true class sizes  |T_i|
    col_sums = C.sum(axis=0)  # predicted cluster sizes  |P_i|
    n = C.sum()               # total number of samples

    precision_sum = 0.0
    recall_sum = 0.0

    for i in range(C.shape[0]):       # loop over true classes (rows)
        for j in range(C.shape[1]):   # loop over predicted clusters (columns)
            if C[i, j] == 0:
                continue              # skip empty cells

            # C[i,j] samples all share the same T_i and P_i
            # so they all get the same P and R value
            # but we must count each sample, so we multiply by C[i,j]

            p_ij = C[i, j] / col_sums[j]   # precision for samples in cell (i,j)
            r_ij = C[i, j] / row_sums[i]   # recall for samples in cell (i,j)

            precision_sum += C[i, j] * p_ij   # sum over all samples in cell
            recall_sum    += C[i, j] * r_ij   # sum over all samples in cell

    P_bcubed = precision_sum / n
    R_bcubed = recall_sum / n

    print(f"BCubed Precision: {P_bcubed:.4f}")
    print(f"BCubed Recall:    {R_bcubed:.4f}")
    print(f"BCubed F1:        {2*P_bcubed*R_bcubed/(P_bcubed+R_bcubed):.4f}")


def bcubed(targets, predictions, beta: float = 1):
    """B3 metric (see Baldwin1998)
    Args:
        targets (torch.Tensor): true labels
        predictions (torch.Tensor): predicted labels
        beta (float, optional): beta for f_score. Defaults to 1.
    Returns:
        Tuple[float, float, float]: b3 f1, precision and recall
    """

    # compute how the target instances are distributed across the predicted classes
    cont_mat = contingency_matrix(targets, predictions)
    # Normlaized by number of all instances
    cont_mat_norm = cont_mat / cont_mat.sum()

    if args.gn_debug:
        print(f"Cont matrix: {cont_mat.sum()}, \n{cont_mat}")

    precision = np.sum(cont_mat_norm * (cont_mat /
                       cont_mat.sum(axis=0))).item()
    recall = np.sum(cont_mat_norm * (cont_mat /
                    np.expand_dims(cont_mat.sum(axis=1), 1))).item()
    f1_score = (1 + beta) * precision * recall / (beta * (precision + recall))


    if args.gn_debug:
        bcubed_sankey.bcubed_sankey(targets, predictions, args.ds_name)
        bcubed_heatmap_large.bcubed_head_map(targets, predictions, args.ds_name)

    return f1_score, precision, recall


def v_measure(targets, predictions):
    """V-measure
    Args:
        targets (torch.Tensor): true labels
        predictions (torch.Tensor): predictions
    Returns:
        Tuple[float, float, float]: V-measure f1, homogeneity (~prec), completeness (~rec)
    """
    homogeneity = homogeneity_score(targets, predictions)
    completeness = completeness_score(targets, predictions)
    v = 2 * homogeneity * completeness / (homogeneity + completeness)

    return v, homogeneity, completeness


def evaluate_promptore(fewrel: pd.DataFrame, predicted_labels: torch.Tensor) -> tuple:
    """Evaluate PromptORE
    Args:
        fewrel (pd.DataFrame): fewrel
        predicted_labels (torch.Tensor): predicted labels

    Returns:
        tuple: scores
    """
    labels = torch.Tensor(fewrel['output_label'].tolist()).long()

    if args.gn_debug:
        print(f"Labels: {len(labels)}, {labels}")
        print(f"Predic: {len(predicted_labels)}, {predicted_labels}")

    ari = adjusted_rand_score(labels, predicted_labels)
    v, v_hom, v_comp = v_measure(labels, predicted_labels)
    b3, b3_prec, b3_rec = bcubed(labels, predicted_labels)

    return b3, b3_prec, b3_rec, v, v_hom, v_comp, ari

################################################################################
# PromptORE
################################################################################


def compute_promptore_relation_embedding(ore_model: 'ore_models.BaseOreModel',
                                         fewrel: pd.DataFrame,
                                         template: str,
                                         max_len: int = 128,
                                         batch_size: int = 256) -> pd.DataFrame:
    """Compute PromptORE relation embedding for the dataframe

    Args:
        ore_model (ore_models.BaseOreModel): loaded model adapter (encoder-MLM or causal-LM)
        fewrel (pd.DataFrame): fewrel dataset
        template (str): prompt template. Authorized parameters are {e1} {e2} {sent},
            plus {mask} for encoder-MLM models.
        max_len (int, optional): max nb of tokens. Defaults to 128.
        batch_size (int, optional): embedding computation batch size. Defaults to 256.

    Returns:
        pd.DataFrame: fewrel dataset with relation embeddings
    """
    fewrel = fewrel.copy()

    # Tokenize fewrel
    rows = []
    for _, instance in tqdm(fewrel.iterrows(), total=len(fewrel)):
        tokens = instance['tokens'].copy()
        head = ' '.join(tokens[instance['h_start']:instance['h_end']+1])
        tail = ' '.join(tokens[instance['t_start']:instance['t_end']+1])

        sent = ' '.join(tokens)
        text = ore_model.format_prompt(template, sent=sent, e1=head, e2=tail)

        input_ids, attention_mask = ore_model.tokenize(text, max_len)
        rows.append({
            'input_tokens': input_ids,
            'input_attention_mask': attention_mask,
            'input_mask': ore_model.compute_target_index(input_ids, attention_mask),
            'output_r': instance['r'],
        })
    complete_fewrel = pd.DataFrame(rows)
    complete_fewrel['output_label'] = pd.factorize(complete_fewrel['output_r'])[0]

    # Predict embeddings

    tokens = torch.stack(complete_fewrel['input_tokens'].tolist(), dim=0)
    attention_mask = torch.stack(
        complete_fewrel['input_attention_mask'].tolist(), dim=0)
    masks = torch.Tensor(complete_fewrel['input_mask'].tolist()).long()
    dataset = TensorDataset(tokens, attention_mask, masks)
    dataloader = DataLoader(dataset, num_workers=1,
                            batch_size=batch_size, shuffle=False)

    embeddings = []
    for batch in tqdm(dataloader):
        tokens, attention_mask, mask = batch
        embeddings.append(ore_model.compute_batch_embeddings(tokens, attention_mask, mask))
    embeddings = torch.cat(embeddings, dim=0)

    complete_fewrel['embedding'] = list(embeddings)
    print(complete_fewrel.head(3))
    return complete_fewrel


def compute_kmeans_clustering(fewrel_relation_embeddings: pd.DataFrame, n_rel: int, \
    random_state: int):
    """Compute kmeans clustering with fixed nb of clusters
    Args:
        fewrel_relation_embeddings (pd.DataFrame): relation embeddings
        n_rel (int): number of relations (nb of clusters)
    Returns:
        torch.Tensor: predicted labels
    """
    embeddings = torch.stack(fewrel_relation_embeddings['embedding'].tolist())

    model = KMeans(init='k-means++', n_init=10, n_clusters=n_rel, random_state=random_state)
    model.fit(embeddings)
    predicted_labels = model.predict(embeddings)

    return predicted_labels


def estimate_n_rel(fewrel_relation_embeddings: pd.DataFrame, random_state: int, \
    k_range: tuple = [10, 300], k_step: int = 5) -> int:
    """Estimate number of clusters using the elbow rule

    Args:
        fewrel_relation_embeddings (pd.DataFrame): relation embeddings
        k_range (tuple, optional): range of clusters to test. Defaults to [10, 300].
        k_step (int, optional): step. Defaults to 5.

    Returns:
        int: estimated number of clusters
    """
    embeddings = torch.stack(fewrel_relation_embeddings['embedding'].tolist())

    ks = np.arange(k_range[0], k_range[1], k_step)
    model = KMeans(init='k-means++', n_init=10, random_state=random_state)
    visualizer = KElbowVisualizer(
        model, k=ks, metric='silhouette', timings=False, locate_elbow=False)
    visualizer.fit(embeddings)
    silhouette = pd.DataFrame()
    silhouette['ks'] = ks
    silhouette['scores'] = visualizer.k_scores_

    # Kernel ridge
    model = KernelRidge(kernel='rbf', degree=3, gamma=1e-3)
    X = silhouette['ks'].values.reshape(-1, 1)
    model.fit(X=X, y=silhouette['scores'])
    p = model.predict(X=X)

    k_elbow = silhouette['ks'][p.argmax()]
    return k_elbow


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='PromptORE')
    parser.add_argument('--config', help='Path to JSON config file', default='config.json')
    parser.add_argument('--seed', help='Random state', type=int, required=False)
    parser.add_argument('--n-rel', help='Number of clusters', type=int, default=0)
    parser.add_argument('--auto-n-rel', help='estimate the number of clusters', action='store_true')
    parser.add_argument('--min-n-rel', help='In case of cluster estimation, min nb of cluster',
        type=int, default=10)
    parser.add_argument('--max-n-rel', help='In case of cluster estimation, max nb of cluster',
        type=int, default=300)
    parser.add_argument('--step-n-rel', help='In case of cluster estimation, step', type=int, default=5)
    parser.add_argument('--max-len',
        help='Maximum number of tokens (fewrel=150, fewrel_nyt=500, fewrel_pubmed=250)',
        type=int, required=False)
    parser.add_argument('--gn-debug', help='Print helpful statements', action='store_true')
    parser.add_argument('--files', help='File(s) to load from Fewrel', default=[], nargs='+')
    parser.add_argument('--ds-name', help='Name of dataset: FewRel, Tacred, Meddistant19', default="na")
    parser.add_argument('--ignore-na', help='Ignore no relations', action='store_true')
    parser.add_argument('--n-inst', help='Max number of instances to process; for testing. Use 0 for all',
        type=int, default=0)
    parser.add_argument('--model-name', help='HF model name or local path', default='bert-base-uncased')
    parser.add_argument('--model-type', choices=['encoder_mlm', 'causal_lm'], default=None,
        help='Override model family auto-detection')
    parser.add_argument('--batch-size', help='Embedding computation batch size', type=int, default=256)
    parser.add_argument('--quantization', choices=['4bit', '8bit'], default=None,
        help='Causal-LM only: load in 4-bit or 8-bit via bitsandbytes (requires the '
             '"quantization" extra: uv sync --extra quantization)')
    parser.add_argument('--prompt-template', default=None,
        help='Prompt template. Encoder-MLM templates must contain {mask}; causal-LM '
             'templates must not. Defaults to a family-appropriate template if omitted.')
    parser.add_argument('--device', default=None,
        help='Force a specific device (e.g. cpu, cuda:0). Defaults to accelerate '
             'auto-sharding the model across all visible GPUs (device_map="auto").')
    #parser.set_defaults(auto_n_rel=False)

    # Step 1: do a first parse just to get the --config path
    args, remaining = parser.parse_known_args()

    # Step 2: load the JSON and inject as defaults
    if os.path.exists(args.config):
        print(f"Load config: {args.config}")
        with open(args.config) as f:
            config = json.load(f)
            print(config)
        parser.set_defaults(**config)

    # Step 3: parse again — CLI args override the JSON defaults
    args = parser.parse_args()

    # Fail fast on an invalid model/quantization combination before loading
    # any dataset or downloading any model weights.
    resolved_model_type = ore_models.detect_model_type(args.model_name, args.model_type)
    if args.quantization and resolved_model_type != 'causal_lm':
        parser.error(f"--quantization is only supported for causal-LM models, but "
                     f"'{args.model_name}' resolved to model_type='{resolved_model_type}'")

    # Read docred files
    print(f"Read dataset rel files: {args.ds_name}\nfiles: {args.files}")
    files = args.files
    fewrel_files = [parse_data.parse_dataset(file, ds_name=args.ds_name, ignore_na=args.ignore_na) for file in files]
    fewrel = pd.concat(fewrel_files).reset_index(drop=True)

    n_max_token_list, n_groups = parse_data.get_data_frame_statistcs(fewrel, args.gn_debug)
    print(f"Max len: {n_max_token_list}, number relations: {n_groups}, number of instances: {len(fewrel)}")

    if args.n_inst > 0:
        size_of_inst = args.n_inst
        print(f"Using subset of instances: {size_of_inst}")
        fewrel = parse_data.get_dataset_subset(fewrel, N=size_of_inst, random=True)

    do_ore = True

    if do_ore:
        ore_model = ore_models.create_ore_model(
            model_name=args.model_name, model_type=args.model_type,
            device=args.device, quantization=args.quantization)
        print(f"Resolved model_type={ore_model.model_type}, "
              f"input_device={ore_model.input_device}, "
              f"hf_device_map={getattr(ore_model.model, 'hf_device_map', None)}")

        # Compute relation embeddings
        prompt_template = args.prompt_template or ore_models.DEFAULT_TEMPLATES[ore_model.model_type]
        print(f"Compute relation embeddings using prompt template: {prompt_template}")
        relation_embeddings = compute_promptore_relation_embedding(ore_model,
            fewrel, template=prompt_template, max_len=args.max_len, batch_size=args.batch_size)

        # Compute clustering
        # cluster-size:  known n-groups from gold data if not given manual
        print(f"How to get K for k-means clustering?")
        print(f"\testimate: {args.auto_n_rel} or from data: {n_groups} or manual: {args.n_rel}")

        if args.auto_n_rel:
            print(f"\tEstimate K via Elbow Rule")
            n_rel = estimate_n_rel(
                relation_embeddings, args.seed, (args.min_n_rel, args.max_n_rel), args.step_n_rel)
            print(f'\tEstimated n_rel={n_rel}')
        else:
            n_rel = args.n_rel if args.n_rel > 0 else n_groups
            print(f"\tUse K -> {n_rel}")

        print(f"Do k-mean clustering with K={n_rel}")
        predicted_labels = compute_kmeans_clustering(relation_embeddings, n_rel, args.seed)

        if args.gn_debug:
            print(f"Do labelling")
            labeller.predict_cluster_labels(fewrel, predicted_labels)

        # Evaluation
        print(f"Evaluate")
        b3, b3_prec, b3_rec, v, v_hom, v_comp, ari = evaluate_promptore(relation_embeddings,
                                                                        predicted_labels)
        print(f'       B3: prec={b3_prec:.4f} rec={b3_rec:.4f} f1={b3:.4f}')
        print(f'V-measure: hom={v_hom:.4f} comp={v_comp:.4f} f1={v:.4f}')
        print(f'      ARI: {ari:.4f}')
