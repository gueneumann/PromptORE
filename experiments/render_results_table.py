"""Render a results JSON (from run_experiments.py) as a compact table image,
styled after the metrics table (B3 / V-measure / ARI, grouped column headers,
bold-best / underlined-second-best per column) commonly used in OpenRE papers.

Each run's "Run" column shows a short prompt id (e.g. "default", "alt_1",
"alt_2", ...) rather than the full template text; a legend box below the
table maps each id to its actual prompt template. Set an explicit
"prompt_id" field on a run in the spec to control this label (see
experiments/example_spec.json); if omitted, it falls back to the run's "id"
with a redundant trailing "_<dataset>" stripped.

Usage:
    uv run python experiments/render_results_table.py --results experiments/example_spec_results.json
    uv run python experiments/render_results_table.py --results my_results.json --output my_table.png
"""
import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

METRIC_COLUMNS = [
    ('b3_prec', 'Prec.'), ('b3_rec', 'Rec.'), ('b3_f1', 'F1'),
    ('v_hom', 'Hom.'), ('v_comp', 'Comp.'), ('v_f1', 'F1'),
    ('ari', 'ARI'),
]
GROUP_SPANS = [('B³', 3), ('V-measure', 3), ('ARI', 1)]

METRIC_COL_WIDTH_IN = 0.62
MIN_COL_WIDTHS_IN = {'dataset': 0.7, 'model': 1.0, 'run': 0.55}
COL_PAD_IN = 0.20
ROW_HEIGHT_IN = 0.27
HEADER_HEIGHT_IN = 0.62
MARGIN_IN = 0.15
LEGEND_LINE_HEIGHT_IN = 0.185
LEGEND_FONTSIZE = 7.3
DPI = 220

FS_HEADER = 9.5
FS_BODY = 8.5


def short_run_label(run_id: str, ds_name: str) -> str:
    """Strip a redundant trailing "_<dataset>" from a run id (the dataset
    already has its own column). Fallback when a run has no prompt_id."""
    suffix = '_' + re.sub(r'\W+', '', ds_name).lower()
    if run_id.lower().endswith(suffix):
        return run_id[:-len(suffix)]
    return run_id


def load_rows(results_path: str) -> list:
    with open(results_path) as f:
        spec = json.load(f)

    rows = []
    n_skipped = 0
    for run in spec['runs']:
        result = run.get('result', {})
        if result.get('status') != 'ok':
            n_skipped += 1
            continue
        ds_name = result.get('ds_name', run.get('dataset_config', '?'))
        prompt_id = run.get('prompt_id') or short_run_label(run.get('id', '?'), ds_name)
        rows.append({
            'ds_name': ds_name,
            'model_name': result.get('model_name', run.get('model_name', '?')),
            'prompt_id': prompt_id,
            'prompt_template': result.get('prompt_template', ''),
            **{key: result[key] for key, _ in METRIC_COLUMNS},
        })
    if n_skipped:
        print(f"Skipping {n_skipped} run(s) without status=='ok' (failed or dry-run).")
    return rows


def group_by_dataset(rows: list) -> dict:
    groups = {}
    for row in rows:
        groups.setdefault(row['ds_name'], []).append(row)
    for ds_rows in groups.values():
        ds_rows.sort(key=lambda r: r['b3_f1'], reverse=True)
    return groups


def best_and_second(ds_rows: list, key: str):
    values = sorted({r[key] for r in ds_rows}, reverse=True)
    best = values[0] if len(values) > 0 else None
    second = values[1] if len(values) > 1 else None
    return best, second


def prompt_id_sort_key(prompt_id: str):
    if prompt_id == 'default':
        return (0, 0, '')
    m = re.match(r'alt_(\d+)$', prompt_id)
    if m:
        return (1, int(m.group(1)), '')
    return (2, 0, prompt_id)


def build_legend_entries(rows: list) -> list:
    """Unique (prompt_id, template) pairs, in a sensible reading order
    (default, alt_1, alt_2, ..., then anything else alphabetically)."""
    seen = {}
    for row in rows:
        seen.setdefault(row['prompt_id'], row['prompt_template'])
    return sorted(seen.items(), key=lambda kv: prompt_id_sort_key(kv[0]))


def measure_text_widths_in(entries: list, dpi: int = DPI) -> list:
    """entries: list of (text, fontsize, fontweight, family). Returns
    rendered widths in inches, measured via a throwaway scratch figure."""
    scratch_fig = plt.figure(figsize=(30, 4), dpi=dpi)
    scratch_ax = scratch_fig.add_axes([0, 0, 1, 1])
    scratch_ax.axis('off')
    texts = [scratch_ax.text(0, 0, text, fontsize=fontsize, fontweight=fontweight, family=family)
             for text, fontsize, fontweight, family in entries]
    scratch_fig.canvas.draw()
    renderer = scratch_fig.canvas.get_renderer()
    widths = [t.get_window_extent(renderer=renderer).width / dpi for t in texts]
    plt.close(scratch_fig)
    return widths


def compute_column_width(header_text: str, body_texts: list, min_width: float,
                          body_fontweight: str = 'normal', body_family: str = 'serif') -> float:
    entries = [(header_text, FS_HEADER, 'bold', 'serif')]
    entries += [(t, FS_BODY, body_fontweight, body_family) for t in body_texts]
    widths = measure_text_widths_in(entries)
    return max([min_width] + [w + COL_PAD_IN for w in widths])


def wrap_legend_line(prompt_id: str, template: str, max_width_in: float) -> list:
    prefix = f"{prompt_id}: "
    # Rough monospace char width estimate at LEGEND_FONTSIZE, for wrapping only.
    char_width_in = LEGEND_FONTSIZE * 0.6 / 72
    chars_per_line = max(20, int(max_width_in / char_width_in))
    wrapped = textwrap.wrap(template, width=max(1, chars_per_line - len(prefix)),
                            subsequent_indent=' ' * len(prefix)) or ['']
    wrapped[0] = prefix + wrapped[0]
    return wrapped


def render_table(results_path: str, output_path: str):
    rows = load_rows(results_path)
    if not rows:
        raise ValueError(f"No successful runs found in {results_path}")
    groups = group_by_dataset(rows)
    legend_entries = build_legend_entries(rows)

    # --- Dynamic column widths ---
    col_width = {
        'dataset': compute_column_width('Dataset', list(groups.keys()),
            MIN_COL_WIDTHS_IN['dataset'], body_fontweight='bold'),
        'model': compute_column_width('Model', [r['model_name'] for r in rows],
            MIN_COL_WIDTHS_IN['model']),
        'run': compute_column_width('Run', [r['prompt_id'] for r in rows],
            MIN_COL_WIDTHS_IN['run'], body_family='monospace'),
    }

    n_data_rows = len(rows)
    n_groups = len(groups)
    total_width = col_width['dataset'] + col_width['model'] + col_width['run'] \
        + len(METRIC_COLUMNS) * METRIC_COL_WIDTH_IN
    table_height = HEADER_HEIGHT_IN + n_data_rows * ROW_HEIGHT_IN + (n_groups - 1) * 0.03

    # --- Legend layout (wrapped to the table width) ---
    legend_lines = []
    for prompt_id, template in legend_entries:
        legend_lines.extend(wrap_legend_line(prompt_id, template, total_width - 0.2))
    legend_title_h = 0.22
    legend_box_h = legend_title_h + len(legend_lines) * LEGEND_LINE_HEIGHT_IN + 0.14
    legend_gap = 0.16

    fig_w = total_width + 2 * MARGIN_IN
    fig_h = table_height + legend_gap + legend_box_h + 2 * MARGIN_IN
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis('off')
    plt.rcParams['font.family'] = 'serif'

    x = MARGIN_IN
    col_x = {}
    for name in ('dataset', 'model', 'run'):
        col_x[name] = x
        x += col_width[name]
    metric_x = []
    for _ in METRIC_COLUMNS:
        metric_x.append(x)
        x += METRIC_COL_WIDTH_IN
    right_edge = x

    top_y = fig_h - MARGIN_IN

    def y_of(row_idx_from_top):
        return top_y - row_idx_from_top

    # --- Header ---
    header_row1_y = y_of(0.28)
    header_row2_y = y_of(0.55)
    ax.text(col_x['dataset'], y_of(0.42), 'Dataset', fontsize=FS_HEADER, fontweight='bold',
            va='center', ha='left')
    ax.text(col_x['model'], y_of(0.42), 'Model', fontsize=FS_HEADER, fontweight='bold',
            va='center', ha='left')
    ax.text(col_x['run'], y_of(0.42), 'Run', fontsize=FS_HEADER, fontweight='bold',
            va='center', ha='left')

    mi = 0
    for group_label, span in GROUP_SPANS:
        group_left = metric_x[mi]
        group_right = metric_x[mi + span - 1] + METRIC_COL_WIDTH_IN
        group_center = (group_left + group_right) / 2
        if span > 1:
            ax.text(group_center, header_row1_y, group_label, fontsize=FS_HEADER,
                    fontweight='bold', va='center', ha='center')
            ax.plot([group_left + 0.05, group_right - 0.05],
                    [header_row1_y - 0.14, header_row1_y - 0.14], color='black', linewidth=0.7)
            for j in range(span):
                sub_label = METRIC_COLUMNS[mi + j][1]
                ax.text(metric_x[mi + j] + METRIC_COL_WIDTH_IN / 2, header_row2_y, sub_label,
                        fontsize=FS_HEADER - 0.5, va='center', ha='center', style='italic')
        else:
            ax.text(group_center, y_of(0.42), group_label, fontsize=FS_HEADER,
                    fontweight='bold', va='center', ha='center')
        mi += span

    ax.plot([MARGIN_IN, right_edge], [top_y, top_y], color='black', linewidth=1.4)
    ax.plot([MARGIN_IN, right_edge], [top_y - HEADER_HEIGHT_IN, top_y - HEADER_HEIGHT_IN],
            color='black', linewidth=1.0)

    # --- Body ---
    row_i = 0
    group_items = list(groups.items())
    for gi, (ds_name, ds_rows) in enumerate(group_items):
        best = {key: best_and_second(ds_rows, key)[0] for key, _ in METRIC_COLUMNS}
        second = {key: best_and_second(ds_rows, key)[1] for key, _ in METRIC_COLUMNS}

        for r_idx, row in enumerate(ds_rows):
            y = top_y - HEADER_HEIGHT_IN - (row_i + 0.5) * ROW_HEIGHT_IN
            if r_idx == 0:
                ax.text(col_x['dataset'], y, ds_name, fontsize=FS_BODY, fontweight='bold',
                        va='center', ha='left')
            ax.text(col_x['model'], y, row['model_name'], fontsize=FS_BODY, va='center', ha='left')
            ax.text(col_x['run'], y, row['prompt_id'], fontsize=FS_BODY, va='center', ha='left',
                    family='monospace')

            for ci, (key, _) in enumerate(METRIC_COLUMNS):
                value = row[key]
                text = f"{value:.4f}"
                is_best = best[key] is not None and abs(value - best[key]) < 1e-12
                is_second = (not is_best) and second[key] is not None and abs(value - second[key]) < 1e-12
                cx = metric_x[ci] + METRIC_COL_WIDTH_IN / 2
                ax.text(cx, y, text, fontsize=FS_BODY, va='center', ha='center',
                        fontweight='bold' if is_best else 'normal')
                if is_second:
                    ax.add_line(Line2D([cx - 0.19, cx + 0.19], [y - 0.085, y - 0.085],
                                       color='black', linewidth=0.8))
            row_i += 1

        block_bottom_y = top_y - HEADER_HEIGHT_IN - row_i * ROW_HEIGHT_IN
        if gi < len(group_items) - 1:
            ax.plot([MARGIN_IN, right_edge], [block_bottom_y, block_bottom_y],
                    color='0.6', linewidth=0.6)

    bottom_y = top_y - HEADER_HEIGHT_IN - n_data_rows * ROW_HEIGHT_IN
    ax.plot([MARGIN_IN, right_edge], [bottom_y, bottom_y], color='black', linewidth=1.4)

    # --- Legend box: prompt_id -> template text ---
    legend_top_y = bottom_y - legend_gap
    legend_bottom_y = legend_top_y - legend_box_h
    ax.add_patch(Rectangle((MARGIN_IN, legend_bottom_y), total_width, legend_box_h,
                           facecolor='0.97', edgecolor='0.4', linewidth=0.7))
    ax.text(MARGIN_IN + 0.12, legend_top_y - legend_title_h * 0.7, 'Prompt templates',
            fontsize=FS_BODY, fontweight='bold', va='center', ha='left')
    for li, line in enumerate(legend_lines):
        ly = legend_top_y - legend_title_h - (li + 0.6) * LEGEND_LINE_HEIGHT_IN
        ax.text(MARGIN_IN + 0.12, ly, line, fontsize=LEGEND_FONTSIZE, family='monospace',
                va='center', ha='left')

    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"Wrote {output_path} ({n_data_rows} rows, {len(groups)} dataset group(s), "
          f"{len(legend_entries)} prompt variant(s))")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', required=True, help='Path to a results JSON from run_experiments.py')
    parser.add_argument('--output', default=None,
        help='Output image path. Defaults to <results>_table.png next to the results file.')
    cli_args = parser.parse_args()

    default_output = str(Path(cli_args.results).with_name(Path(cli_args.results).stem + '_table.png'))
    render_table(cli_args.results, cli_args.output or default_output)
