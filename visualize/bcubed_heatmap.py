import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics.cluster import contingency_matrix



# ── Contingency matrix + BCubed ───────────────────────────────────────────
def bcubed_head_map(labels_true, labels_pred, ds_name):
    C = contingency_matrix(labels_true, labels_pred)
    row_sums = C.sum(axis=1)
    col_sums = C.sum(axis=0)
    n        = C.sum()

    P_cell = np.where(col_sums[None, :] > 0, C / col_sums[None, :], 0.0)
    R_cell = np.where(row_sums[:, None] > 0, C / row_sums[:, None], 0.0)
    P_bcubed = np.sum(C * P_cell) / n
    R_bcubed = np.sum(C * R_cell) / n
    F_bcubed = 2 * P_bcubed * R_bcubed / (P_bcubed + R_bcubed)

    # ── Adaptive figure size ──────────────────────────────────────────────────
    n_rows, n_cols = C.shape
    cell_size  = max(0.55, min(1.1, 8.0 / max(n_rows, n_cols)))
    fig_w      = max(10, n_cols * cell_size + 4.5)
    fig_h      = max( 8, n_rows * cell_size + 3.0)
    show_text  = cell_size >= 0.75          # show count text only if cells are big enough
    font_cell  = max(5, int(cell_size * 9))

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='#F7F9FC')
    fig.suptitle('BCubed Precision & Recall — Contingency Matrix\n'
                 f'({n_rows} true classes × {n_cols} predicted clusters,  n={n} samples)',
                 fontsize=13, fontweight='bold', y=0.99, color='#1a1a2e')

    # Grid: top margin bar | main heatmap | right margin bar | colorbar
    gs = fig.add_gridspec(2, 3,
                          width_ratios=[6, 0.6, 0.4],
                          height_ratios=[0.6, 6],
                          hspace=0.03, wspace=0.03,
                          left=0.09, right=0.97,
                          top=0.91, bottom=0.09)

    ax_main  = fig.add_subplot(gs[1, 0])
    ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_cbar  = fig.add_subplot(gs[1, 2])

    cmap   = plt.cm.Blues
    norm   = mcolors.Normalize(vmin=0, vmax=C.max())

    # ── Main heatmap ──────────────────────────────────────────────────────────
    img = ax_main.imshow(C, aspect='auto', cmap=cmap,
                         norm=norm, origin='lower',
                         extent=[0, n_cols, 0, n_rows],
                         interpolation='nearest', zorder=1)

    if show_text:
        for i in range(n_rows):
            for j in range(n_cols):
                val = C[i, j]
                if val == 0:
                    continue
                intensity = val / C.max()
                tc = 'white' if intensity > 0.55 else '#1a1a2e'
                ax_main.text(j + 0.5, i + 0.5, str(val),
                             ha='center', va='center',
                             fontsize=font_cell, color=tc, zorder=2)

    ax_main.set_xlim(0, n_cols)
    ax_main.set_ylim(0, n_rows)

    # Tick labels: show all if ≤16, else every other
    step = 1 if n_cols <= 16 else 2
    ax_main.set_xticks(np.arange(0, n_cols, step) + 0.5)
    ax_main.set_xticklabels([f'C{j}' for j in range(0, n_cols, step)],
                             fontsize=max(6, font_cell - 1), rotation=45, ha='right')
    step_r = 1 if n_rows <= 16 else 2
    ax_main.set_yticks(np.arange(0, n_rows, step_r) + 0.5)
    ax_main.set_yticklabels([f'{i}' for i in range(0, n_rows, step_r)],
                             fontsize=max(6, font_cell - 1))

    ax_main.set_xlabel('Predicted Cluster  →  column sum = |Pᵢ|', fontsize=9, labelpad=6)
    ax_main.set_ylabel('True Class  →  row sum = |Tᵢ|',           fontsize=9, labelpad=6)
    ax_main.tick_params(length=0)
    for sp in ax_main.spines.values():
        sp.set_visible(False)

    # Light grid
    for x in range(n_cols + 1):
        ax_main.axvline(x, color='white', linewidth=0.4, zorder=3)
    for y in range(n_rows + 1):
        ax_main.axhline(y, color='white', linewidth=0.4, zorder=3)

    # ── Top bar: column sums ──────────────────────────────────────────────────
    ax_top.bar(np.arange(n_cols) + 0.5, col_sums,
               width=0.8, color='#4C72B0', alpha=0.7)
    ax_top.set_xlim(0, n_cols)
    ax_top.set_ylim(0, col_sums.max() * 1.4)
    ax_top.axis('off')
    ax_top.set_title('Column sums  (predicted cluster sizes)', fontsize=8,
                     color='#555', pad=3)

    # ── Right bar: row sums ───────────────────────────────────────────────────
    ax_right.barh(np.arange(n_rows) + 0.5, row_sums,
                  height=0.8, color='#DD8452', alpha=0.7)
    ax_right.set_ylim(0, n_rows)
    ax_right.set_xlim(0, row_sums.max() * 1.4)
    ax_right.axis('off')

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cb = fig.colorbar(img, cax=ax_cbar)
    cb.set_label('Sample count C[i,j]', fontsize=8, labelpad=6)
    cb.ax.tick_params(labelsize=7)

    # ── BCubed score annotation ───────────────────────────────────────────────
    score_txt = (f'BCubed\n'
                 f'P = {P_bcubed:.3f}\n'
                 f'R = {R_bcubed:.3f}\n'
                 f'F1= {F_bcubed:.3f}')
    ax_main.text(0.01, 0.99, score_txt,
                 ha='left', va='top', fontsize=8, color='white',
                 fontweight='bold', linespacing=1.7,
                 bbox=dict(boxstyle='round,pad=0.5', fc='#1a1a2e',
                           ec='none', alpha=0.88),
                 transform=ax_main.transAxes, zorder=10)

    fig.text(0.5, 0.01,
             'Color intensity = sample count  |  P = precision,  R = recall  (BCubed, averaged per sample)',
             ha='center', fontsize=8, color='#777', style='italic')


    plt.savefig(f'VIS/{ds_name}_bcubed_heatmap_large.png', dpi=150, bbox_inches='tight',
                facecolor='#F7F9FC')
    print("Heatmap done.")
