"""Batch-run PromptORE experiments defined in a JSON spec file.

A spec lists "runs": each references a dataset config (config/*.json, same
format promptore.py's --config uses) plus overrides (model_name, model_type,
batch_size, quantization, prompt_template, ...). See example_spec.json for
the format.

Consecutive runs that share the same model/quantization/device reuse the
already-loaded model instead of reloading it, and runs against the same
dataset_config reuse the already-parsed dataframe -- so grouping a prompt
sweep for one model together in the spec (as example_spec.json does) is
significantly faster than reloading per run.

Usage:
    uv run python experiments/run_experiments.py --spec experiments/example_spec.json
    uv run python experiments/run_experiments.py --spec my_spec.json --output my_results.json
    uv run python experiments/run_experiments.py --spec my_spec.json --dry-run
"""
import argparse
import datetime
import gc
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

import ore_models
import parse_data
import promptore

# Fields promptore.run_pipeline()/resolve_n_rel() expect on `args`, with the
# same defaults promptore.py's own argparse setup uses. A run's dataset_config
# JSON and then the run entry itself are layered on top of these.
DEFAULT_RUN_FIELDS = {
    'seed': None,
    'n_rel': 0,
    'auto_n_rel': False,
    'min_n_rel': 10,
    'max_n_rel': 300,
    'step_n_rel': 5,
    'max_len': None,
    'gn_debug': False,
    'files': [],
    'ds_name': 'na',
    'ignore_na': False,
    'n_inst': 0,
    'model_name': None,
    'model_type': None,
    'batch_size': 256,
    'quantization': None,
    'prompt_template': None,
    'device': None,
}

# Keys that are run-harness bookkeeping, not pipeline parameters -- never
# copied onto the args namespace.
NON_ARG_RUN_KEYS = {'id', 'dataset_config'}


def load_run_args(run: dict) -> SimpleNamespace:
    """Merge a run's dataset_config JSON (defaults) with the run's own
    overrides (only keys explicitly present and non-null) into a namespace
    shaped like promptore.py's parsed `args`.
    """
    with open(run['dataset_config']) as f:
        config = json.load(f)

    merged = {**DEFAULT_RUN_FIELDS, **config}
    for key, value in run.items():
        if key in NON_ARG_RUN_KEYS or value is None:
            continue
        merged[key] = value

    if not merged.get('model_name'):
        raise ValueError(f"Run {run.get('id')!r} has no model_name "
                         f"(set it on the run or in its dataset_config)")
    return SimpleNamespace(**merged)


class ModelCache:
    """Single-slot cache: reuses the loaded model across consecutive runs
    that share the same (model_name, model_type, quantization, device)."""

    def __init__(self):
        self.signature = None
        self.ore_model = None

    def get(self, args) -> 'ore_models.BaseOreModel':
        signature = (args.model_name, args.model_type, args.quantization, args.device)
        if signature == self.signature and self.ore_model is not None:
            return self.ore_model
        self.clear()
        self.ore_model = ore_models.create_ore_model(
            model_name=args.model_name, model_type=args.model_type,
            device=args.device, quantization=args.quantization)
        self.signature = signature
        return self.ore_model

    def clear(self):
        if self.ore_model is not None:
            del self.ore_model
            self.ore_model = None
            self.signature = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class DatasetCache:
    """Caches the parsed (pre-subsampling) dataframe per dataset_config path,
    since parsing large TACRED/MedDistant19 files repeatedly across a prompt
    sweep is wasted work."""

    def __init__(self):
        self._cache = {}

    def get(self, dataset_config_path: str, args) -> tuple:
        key = (dataset_config_path, args.ds_name, args.ignore_na, tuple(args.files))
        if key not in self._cache:
            fewrel_files = [parse_data.parse_dataset(f, ds_name=args.ds_name, ignore_na=args.ignore_na)
                            for f in args.files]
            fewrel = pd.concat(fewrel_files).reset_index(drop=True)
            _, n_groups = parse_data.get_data_frame_statistcs(fewrel, False)
            self._cache[key] = (fewrel, n_groups)
        fewrel, n_groups = self._cache[key]
        if args.n_inst and args.n_inst > 0:
            fewrel = parse_data.get_dataset_subset(fewrel, N=args.n_inst, random=True)
        return fewrel, n_groups


def run_experiments(spec_path: str, output_path: str, dry_run: bool = False) -> list:
    with open(spec_path) as f:
        spec = json.load(f)

    model_cache = ModelCache()
    dataset_cache = DatasetCache()
    results = []

    for i, run in enumerate(spec['runs']):
        run_id = run.get('id', f'run_{i}')
        print(f"\n=== [{i + 1}/{len(spec['runs'])}] {run_id} ===")
        result = {'status': 'error', 'error': None}
        try:
            args = load_run_args(run)
            resolved_type = ore_models.detect_model_type(args.model_name, args.model_type)
            if args.quantization and resolved_type != 'causal_lm':
                raise ValueError(f"--quantization is only supported for causal-LM models, but "
                                 f"'{args.model_name}' resolved to model_type='{resolved_type}'")

            if dry_run:
                result = {'status': 'dry_run_ok', 'resolved_model_type': resolved_type}
                print(f"  dry-run OK: model_type={resolved_type}, dataset_config={run['dataset_config']}")
            else:
                fewrel, n_groups = dataset_cache.get(run['dataset_config'], args)
                ore_model = model_cache.get(args)

                start = time.time()
                result = promptore.run_pipeline(ore_model, fewrel, n_groups, args)
                elapsed = time.time() - start

                result.update({
                    'status': 'ok',
                    'resolved_model_type': ore_model.model_type,
                    'elapsed_seconds': round(elapsed, 1),
                    'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
                    'ds_name': args.ds_name,
                    'model_name': args.model_name,
                    'batch_size': args.batch_size,
                    'quantization': args.quantization,
                    'seed': args.seed,
                    'n_inst': args.n_inst,
                })
                print(f"  -> B3 f1={result['b3_f1']:.4f}  V f1={result['v_f1']:.4f}  "
                      f"ARI={result['ari']:.4f}  ({elapsed:.1f}s)")
        except Exception as e:
            traceback.print_exc()
            result = {'status': 'error', 'error': str(e)}
            model_cache.clear()  # a failed load may have left the GPU in a bad state

        results.append({**run, 'result': result})

        # Write after every run so a long sweep isn't lost to a later crash/interrupt.
        with open(output_path, 'w') as f:
            json.dump({'runs': results}, f, indent=2)

    n_ok = sum(1 for r in results if r['result'].get('status') in ('ok', 'dry_run_ok'))
    n_err = sum(1 for r in results if r['result'].get('status') == 'error')
    print(f"\nDone: {n_ok} ok, {n_err} failed, out of {len(results)} runs. Wrote {output_path}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--spec', required=True, help='Path to an experiment spec JSON file')
    parser.add_argument('--output', default=None,
        help='Where to write results JSON. Defaults to <spec>_results.json next to the spec.')
    parser.add_argument('--dry-run', action='store_true',
        help='Validate the spec (config loading, model-family detection) without loading any model or dataset')
    cli_args = parser.parse_args()

    default_output = str(Path(cli_args.spec).with_name(Path(cli_args.spec).stem + '_results.json'))
    run_experiments(cli_args.spec, cli_args.output or default_output, dry_run=cli_args.dry_run)
