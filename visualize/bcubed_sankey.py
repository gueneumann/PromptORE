import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from sklearn.metrics.cluster import contingency_matrix
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# ── Demo data: 16 classes, 700 instances each ─────────────────────────────
#N_CLASSES   = 16
#N_PER_CLASS = 700
#np.random.seed(42)
#X, labels_true = make_blobs(n_samples=N_CLASSES * N_PER_CLASS,
 #                            centers=N_CLASSES, cluster_std=2.5)
#labels_pred = KMeans(n_clusters=N_CLASSES, random_state=42, n_init=10).fit_predict(X)

# ── Contingency matrix + BCubed ───────────────────────────────────────────

def bcubed_sankey(labels_true, labels_pred, ds_name):
    C = contingency_matrix(labels_true, labels_pred)
    row_sums = C.sum(axis=1)
    col_sums = C.sum(axis=0)
    n        = C.sum()

    P_cell = np.where(col_sums[None, :] > 0, C / col_sums[None, :], 0.0)
    R_cell = np.where(row_sums[:, None] > 0, C / row_sums[:, None], 0.0)
    P_bcubed = np.sum(C * P_cell) / n
    R_bcubed = np.sum(C * R_cell) / n
    F_bcubed = 2 * P_bcubed * R_bcubed / (P_bcubed + R_bcubed)

    n_true, n_pred = C.shape

    # ── Colour palettes ───────────────────────────────────────────────────────
    true_cmap = plt.cm.tab20
    pred_cmap = plt.cm.tab20b
    true_colors = [true_cmap(i / n_true) for i in range(n_true)]
    pred_colors = [pred_cmap(j / n_pred) for j in range(n_pred)]

    # ── Layout constants ──────────────────────────────────────────────────────
    LEFT_X,  RIGHT_X  = 0.22, 0.78   # band centre x positions
    BAND_W            = 0.045         # width of each node band
    GAP               = 0.004         # gap between stacked flows on a node
    FIG_H             = 11

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(13, FIG_H), facecolor='#F7F9FC')
    ax.set_facecolor('#F7F9FC')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    fig.suptitle('BCubed — Sankey Diagram of Contingency Matrix\n'
                 f'({n_true} true classes × {n_pred} predicted clusters,  n={n} samples)',
                 fontsize=13, fontweight='bold', y=0.99, color='#1a1a2e')

    # ── Compute node y-positions (bottom of each band) ────────────────────────
    MARGIN = 0.04
    USABLE = 1.0 - 2 * MARGIN

    def band_positions(sums, usable=USABLE, margin=MARGIN, gap=GAP):
        """Return list of (y_bottom, height) for each node, top-to-bottom."""
        total = sums.sum()
        heights = sums / total * (usable - gap * (len(sums) - 1))
        positions = []
        y = margin
        for h in heights:
            positions.append((y, h))
            y += h + gap
        return positions[::-1]   # reverse so class 0 is at top

    true_bands = band_positions(row_sums)
    pred_bands = band_positions(col_sums)

    # For each predicted cluster node we need to know the current fill level
    # when drawing flows — track separately for left and right side of each flow.
    true_fill = [b[0] for b in true_bands]   # current bottom fill on true side
    pred_fill = [b[0] for b in pred_bands]   # current bottom fill on pred side

    # Sort flow drawing order: largest flows first for visual clarity
    flows = []
    for i in range(n_true):
        for j in range(n_pred):
            if C[i, j] > 0:
                flows.append((C[i, j], i, j))
    flows.sort(key=lambda x: -x[0])

    # ── Draw flows (cubic Bezier ribbons) ─────────────────────────────────────
    total = n
    for (val, i, j) in flows:
        h = val / total * (USABLE - GAP * (max(n_true, n_pred) - 1))

        y0_bot = true_fill[i]
        y1_bot = pred_fill[j]
        true_fill[i] += h
        pred_fill[j] += h

        # Base colour from true class, with alpha scaled by flow size
        base = true_colors[i]
        alpha = 0.25 + 0.55 * (val / C.max())

        # Four corners of the ribbon
        x_l0, x_l1 = LEFT_X  + BAND_W / 2, LEFT_X  + BAND_W / 2
        x_r0, x_r1 = RIGHT_X - BAND_W / 2, RIGHT_X - BAND_W / 2

        # Control points for cubic bezier (horizontal tangents)
        cx = (x_l0 + x_r0) / 2

        # Build path as a filled polygon using sampled bezier curves
        t_vals = np.linspace(0, 1, 60)

        def bezier_y(t, y_start, y_end):
            # cubic bezier with horizontal tangents
            return (1-t)**3 * y_start + 3*(1-t)**2*t * y_start + 3*(1-t)*t**2 * y_end + t**3 * y_end

        def bezier_x(t, x_start, x_end):
            cx_mid = (x_start + x_end) / 2
            return (1-t)**3*x_start + 3*(1-t)**2*t*cx_mid + 3*(1-t)*t**2*cx_mid + t**3*x_end

        # Top edge: left top → right top
        top_x = [bezier_x(t, x_l1, x_r1) for t in t_vals]
        top_y = [bezier_y(t, y0_bot + h, y1_bot + h) for t in t_vals]

        # Bottom edge: right bottom → left bottom (reversed)
        bot_x = [bezier_x(t, x_r0, x_l0) for t in t_vals]
        bot_y = [bezier_y(t, y1_bot, y0_bot) for t in t_vals]

        xs = top_x + bot_x
        ys = top_y + bot_y

        ax.fill(xs, ys, color=base, alpha=alpha, linewidth=0, zorder=1)

    # ── Draw node bands ───────────────────────────────────────────────────────
    for i, (y_bot, h) in enumerate(true_bands):
        rect = mpatches.FancyBboxPatch(
            (LEFT_X - BAND_W / 2, y_bot), BAND_W, h,
            boxstyle='square,pad=0', linewidth=0,
            facecolor=true_colors[i], zorder=3)
        ax.add_patch(rect)
        # Label to the left
        ax.text(LEFT_X - BAND_W / 2 - 0.01, y_bot + h / 2,
                f'Class {i}  ({row_sums[i]})',
                ha='right', va='center', fontsize=7.5,
                color='#1a1a2e', zorder=4)

    for j, (y_bot, h) in enumerate(pred_bands):
        rect = mpatches.FancyBboxPatch(
            (RIGHT_X - BAND_W / 2, y_bot), BAND_W, h,
            boxstyle='square,pad=0', linewidth=0,
            facecolor=pred_colors[j], zorder=3)
        ax.add_patch(rect)
        # Label to the right
        ax.text(RIGHT_X + BAND_W / 2 + 0.01, y_bot + h / 2,
                f'Cluster {j}  ({col_sums[j]})',
                ha='left', va='center', fontsize=7.5,
                color='#1a1a2e', zorder=4)

    # ── Column headers ────────────────────────────────────────────────────────
    ax.text(LEFT_X,  0.975, 'True Classes',
            ha='center', va='center', fontsize=11, fontweight='bold', color='#1a1a2e')
    ax.text(RIGHT_X, 0.975, 'Predicted Clusters',
            ha='center', va='center', fontsize=11, fontweight='bold', color='#1a1a2e')

    # ── BCubed score box ──────────────────────────────────────────────────────
    score_lines = [
        ('BCubed Scores', '#ffffff', 9, True),
        (f'Precision   {P_bcubed:.4f}', '#6EC6F5', 9, False),
        (f'Recall        {R_bcubed:.4f}', '#F5A56E', 9, False),
        (f'F1               {F_bcubed:.4f}', '#82D98C', 9, False),
    ]
    box_x, box_y = 0.50, 0.50
    ax.text(box_x, box_y,
            '\n'.join(t for t, *_ in score_lines),
            ha='center', va='center', fontsize=9,
            color='white', linespacing=2.2, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.7', fc='#1a1a2e', ec='none', alpha=0.0),
            transform=ax.transAxes, zorder=0)   # invisible placeholder for sizing

    # Draw each line individually with its colour
    line_ys = [box_y + 0.065, box_y + 0.022, box_y - 0.022, box_y - 0.065]
    for (txt, col, fs, bold), ly in zip(score_lines, line_ys):
        ax.text(box_x, ly, txt,
                ha='center', va='center', fontsize=fs,
                color=col, fontweight='bold' if bold else 'normal',
                transform=ax.transAxes, zorder=10)

    # Background box behind score
    bg = mpatches.FancyBboxPatch((0.415, box_y - 0.09), 0.170, 0.18,
                                   boxstyle='round,pad=0.01',
                                   facecolor='#1a1a2e', edgecolor='none',
                                   transform=ax.transAxes, zorder=9, clip_on=False)
    ax.add_patch(bg)

    fig.text(0.5, 0.005,
             'Flow width ~ number of shared samples  |  Colour follows true class',
             ha='center', fontsize=8.5, color='#777', style='italic')

    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.savefig(f'VIS/{ds_name}_bcubed_sankey.png', dpi=150, bbox_inches='tight',
                facecolor='#F7F9FC')
    print("Sankey done.")
