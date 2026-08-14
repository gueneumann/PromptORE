import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics.cluster import contingency_matrix

# GN Does not work for examples larger than 5 classes
def bcubed_bipartite(labels_true, labels_pred):
    C = contingency_matrix(labels_true, labels_pred)
    row_sums = C.sum(axis=1)
    col_sums = C.sum(axis=0)
    n = C.sum()

    P_cell = np.zeros_like(C, dtype=float)
    R_cell = np.zeros_like(C, dtype=float)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if C[i, j] > 0:
                P_cell[i, j] = C[i, j] / col_sums[j]
                R_cell[i, j] = C[i, j] / row_sums[i]

    P_bcubed = np.sum(C * P_cell) / n
    R_bcubed = np.sum(C * R_cell) / n
    F_bcubed = 2 * P_bcubed * R_bcubed / (P_bcubed + R_bcubed)

    n_true = C.shape[0]
    n_pred = C.shape[1]

    TRUE_COLORS = ['#4C72B0', '#DD8452', '#55A868']
    PRED_COLORS = ['#9B59B6', '#E74C3C', '#1ABC9C']
    NODE_R = 0.055

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})

    fig = plt.figure(figsize=(12, 8.5), facecolor='#F7F9FC')
    gs  = gridspec.GridSpec(2, 1, height_ratios=[5.5, 1],
                            hspace=0.08,
                            left=0.02, right=0.98,
                            top=0.93, bottom=0.06)

    ax     = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    for a in (ax, ax_bar):
        a.set_facecolor('#F7F9FC')
        a.axis('off')

    fig.suptitle('BCubed — Bipartite Graph of Contingency Matrix',
                 fontsize=15, fontweight='bold', y=0.98, color='#1a1a2e')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    left_x  = 0.15
    right_x = 0.85
    true_ys = np.linspace(0.82, 0.18, n_true)
    pred_ys = np.linspace(0.82, 0.18, n_pred)
    max_count = C.max()

    # Column headers
    ax.text(left_x,  0.95, 'True Classes',       ha='center', va='center',
            fontsize=12, fontweight='bold', color='#1a1a2e', transform=ax.transAxes)
    ax.text(right_x, 0.95, 'Predicted Clusters', ha='center', va='center',
            fontsize=12, fontweight='bold', color='#1a1a2e', transform=ax.transAxes)

    # ── Edges + labels ────────────────────────────────────────────────────────
    # Place label at 25% from source for crossing edges, 50% for straight ones
    # so labels spread along the edge rather than pile up in the centre.
    edge_list = [(i, j) for i in range(n_true) for j in range(n_pred) if C[i,j] > 0]

    # Fraction along edge where label sits (vary per edge to avoid overlap)
    label_fracs = {}
    crossing_edges = [(i,j) for (i,j) in edge_list if i != j]
    straight_edges = [(i,j) for (i,j) in edge_list if i == j]

    # Straight edges → label at centre
    for (i,j) in straight_edges:
        label_fracs[(i,j)] = 0.50

    # Crossing edges → spread labels: near source side or near target side
    # Sort crossing edges by how much they cross (|i-j|)
    for k, (i,j) in enumerate(crossing_edges):
        # alternate near-source vs near-target
        label_fracs[(i,j)] = 0.25 if k % 2 == 0 else 0.75

    for (i, j) in edge_list:
        val  = C[i, j]
        x0, y0 = left_x,  true_ys[i]
        x1, y1 = right_x, pred_ys[j]

        dx, dy = x1 - x0, y1 - y0
        dist   = np.hypot(dx, dy)
        ux, uy = dx / dist, dy / dist

        sx = x0 + ux * NODE_R
        sy = y0 + uy * NODE_R
        ex = x1 - ux * NODE_R
        ey = y1 - uy * NODE_R

        lw    = 1.5 + 5.0 * val / max_count
        alpha = 0.35 + 0.55 * val / max_count

        ax.annotate('', xy=(ex, ey), xytext=(sx, sy),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='#888899',
                                    lw=lw, alpha=alpha,
                                    connectionstyle='arc3,rad=0.0'),
                    zorder=2)

        # Label position along edge
        t  = label_fracs[(i, j)]
        mx = sx + t * (ex - sx)
        my = sy + t * (ey - sy)

        # Small perpendicular nudge to lift label off the line
        perp_x = -uy * 0.025
        perp_y =  ux * 0.025

        label = f'C={val}   P={P_cell[i,j]:.2f}   R={R_cell[i,j]:.2f}'
        ax.text(mx + perp_x, my + perp_y, label,
                ha='center', va='center', fontsize=7.5, color='#333344',
                bbox=dict(boxstyle='round,pad=0.28', fc='white',
                          ec='#ccccdd', lw=0.6, alpha=0.95),
                transform=ax.transAxes, zorder=5)

    # True class nodes
    for i, (y, col) in enumerate(zip(true_ys, TRUE_COLORS)):
        circle = plt.Circle((left_x, y), NODE_R, color=col,
                             transform=ax.transAxes, zorder=6, clip_on=False)
        ax.add_patch(circle)
        ax.text(left_x, y, f'Class {i}\n|T|={row_sums[i]}',
                ha='center', va='center', fontsize=9, fontweight='bold',
                color='white', transform=ax.transAxes, zorder=7)

    # Predicted cluster nodes
    for j, (y, col) in enumerate(zip(pred_ys, PRED_COLORS)):
        circle = plt.Circle((right_x, y), NODE_R, color=col,
                             transform=ax.transAxes, zorder=6, clip_on=False)
        ax.add_patch(circle)
        ax.text(right_x, y, f'Cluster {j}\n|P|={col_sums[j]}',
                ha='center', va='center', fontsize=9, fontweight='bold',
                color='white', transform=ax.transAxes, zorder=7)

    # ── Score strip ───────────────────────────────────────────────────────────
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(0, 1)

    strip = plt.Rectangle((0.02, 0.05), 0.96, 0.90,
                            transform=ax_bar.transAxes,
                            facecolor='#1a1a2e', zorder=1, clip_on=False)
    ax_bar.add_patch(strip)

    metrics = [
        (0.22, 'Precision', f'{P_bcubed:.4f}', '#6EC6F5'),
        (0.50, 'Recall',    f'{R_bcubed:.4f}', '#F5A56E'),
        (0.78, 'F1',        f'{F_bcubed:.4f}', '#82D98C'),
    ]
    for xc, label, val, col in metrics:
        ax_bar.text(xc, 0.72, label,
                    ha='center', va='center', fontsize=9, color='#aaaacc',
                    transform=ax_bar.transAxes, zorder=2)
        ax_bar.text(xc, 0.28, val,
                    ha='center', va='center', fontsize=15, fontweight='bold',
                    color=col, transform=ax_bar.transAxes, zorder=2)

    # "BCubed Scores" title centred between the three metrics, tucked at top
    ax_bar.text(0.50, 0.93, 'BCubed Scores',
                ha='center', va='top', fontsize=9, fontweight='bold',
                color='#ddddee', transform=ax_bar.transAxes, zorder=2)

    fig.text(0.50, 0.012,
             'Edge thickness ~ overlap count  |  Label: C = count,  P = precision,  R = recall',
             ha='center', fontsize=8.5, color='#777', style='italic')

    plt.savefig('DOC/bcubed_bipartite.png', dpi=150, bbox_inches='tight',
                facecolor='#F7F9FC')
    print("Done.")
