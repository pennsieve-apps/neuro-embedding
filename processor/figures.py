"""Figures of neuro-embedding (PNG, 150 dpi, session id in the title) and the interactive 3-D HTML."""
from __future__ import annotations

from pathlib import Path

import numpy as np

DPI = 150
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": "white"})
    return plt


def subsample(n: int, max_points: int, seed: int = 0) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, max_points, replace=False))


def _colour_values(label: np.ndarray, direction: np.ndarray | None):
    """(values in 0..1, cmap name per point group) for colouring by a behaviour variable; with a
    direction (+1 / -1 / 0) the two running directions get two single-hue ramps, as in CEBRA."""
    lo, hi = np.nanpercentile(label, [1, 99]) if np.isfinite(label).any() else (0, 1)
    v = np.clip((label - lo) / max(hi - lo, 1e-9), 0, 1)
    return v


def embedding_panels(sid: str, panels: list[tuple[str, np.ndarray, str]], label: np.ndarray, label_name: str,
                     direction: np.ndarray | None, out: Path, max_points: int, seed: int) -> Path:
    """One 3-D scatter per method (PCA, CEBRA-Time, CEBRA-Behavior, shuffled), all coloured by the
    same behaviour variable."""
    plt = _plt()
    n = len(panels)
    fig = plt.figure(figsize=(3.6 * n, 4.0))
    idx = subsample(label.size, max_points, seed)
    v = _colour_values(label[idx], None)
    for k, (title, emb, note) in enumerate(panels):
        ax = fig.add_subplot(1, n, k + 1, projection="3d")
        e = emb[idx]
        if direction is not None:
            d = direction[idx]
            for sign, cmap in ((1, "Blues"), (-1, "Oranges"), (0, "Greys")):
                m = d == sign
                if m.any():
                    ax.scatter(e[m, 0], e[m, 1], e[m, 2], c=0.25 + 0.75 * v[m], cmap=cmap, vmin=0, vmax=1, s=1.2,
                               alpha=0.8, linewidths=0, rasterized=True)
        else:
            ax.scatter(e[:, 0], e[:, 1], e[:, 2], c=v, cmap="viridis", vmin=0, vmax=1, s=1.2, alpha=0.8,
                       linewidths=0, rasterized=True)
        ax.set_title(f"{title}\n{note}", fontsize=9, color=INK)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.grid(False)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.pane.set_edgecolor(GRID)
            a.pane.fill = False
    if direction is not None:
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize

        lo, hi = np.nanpercentile(label, [1, 99])
        for i, (cmap, txt) in enumerate((("Blues", "increasing"), ("Oranges", "decreasing"))):
            cax = fig.add_axes([0.25 + 0.3 * i, 0.06, 0.2, 0.02])
            cb = fig.colorbar(cm.ScalarMappable(Normalize(lo, hi), cmap=cmap), cax=cax, orientation="horizontal")
            cb.set_label(f"{label_name} ({txt})", fontsize=8)
    else:
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize

        lo, hi = np.nanpercentile(label, [1, 99])
        cax = fig.add_axes([0.35, 0.06, 0.3, 0.02])
        cb = fig.colorbar(cm.ScalarMappable(Normalize(lo, hi), cmap="viridis"), cax=cax, orientation="horizontal")
        cb.set_label(label_name, fontsize=8)
    fig.suptitle(f"{sid}: population embeddings coloured by {label_name}", fontsize=11, color=INK)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.14, wspace=0.02)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def embedding_html(sid: str, panels: list[tuple[str, np.ndarray]], label: np.ndarray, label_name: str,
                   out: Path, max_points: int, seed: int) -> Path:
    """Interactive 3-D scatter (plotly, the library embedded in the file: works offline)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    idx = subsample(label.size, max_points, seed)
    fig = make_subplots(rows=1, cols=len(panels), specs=[[{"type": "scene"}] * len(panels)],
                        subplot_titles=[p[0] for p in panels], horizontal_spacing=0.02)
    lo, hi = np.nanpercentile(label, [1, 99])
    for k, (title, emb) in enumerate(panels):
        e = emb[idx]
        fig.add_trace(go.Scatter3d(
            x=e[:, 0], y=e[:, 1], z=e[:, 2], mode="markers", name=title,
            marker=dict(size=1.6, color=label[idx], colorscale="Viridis", cmin=lo, cmax=hi, opacity=0.8,
                        colorbar=dict(title=label_name, len=0.6) if k == len(panels) - 1 else None,
                        showscale=k == len(panels) - 1),
            hovertemplate=f"{label_name}: %{{marker.color:.3g}}<extra>{title}</extra>"), row=1, col=k + 1)
    for k in range(len(panels)):
        key = "scene" if k == 0 else f"scene{k + 1}"
        fig.layout[key].update(xaxis=dict(showticklabels=False, title=""), yaxis=dict(showticklabels=False, title=""),
                               zaxis=dict(showticklabels=False, title=""), aspectmode="cube")
    fig.update_layout(title=f"{sid}: embeddings coloured by {label_name} (drag to rotate)", height=560,
                      margin=dict(l=0, r=0, t=70, b=0), showlegend=False, template="plotly_white")
    fig.write_html(out, include_plotlyjs=True, full_html=True, config={"displaylogo": False})
    return out


def explained_variance(sid: str, evr: np.ndarray, fa_k: list[int], fa_ll: list[float], fa_best: int,
                       out: Path) -> Path:
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    k = np.arange(1, evr.size + 1)
    cum = np.cumsum(evr)
    a1.bar(k, evr * 100, color=SERIES[0], width=0.8, label="per component")
    a1.plot(k, cum * 100, color=INK, lw=2, label="cumulative")
    for f in (0.5, 0.8):
        if cum[-1] >= f:
            n = int(np.searchsorted(cum, f) + 1)
            a1.axhline(f * 100, color=MUTED, lw=0.8, ls=":")
            a1.annotate(f"{int(f * 100)}% at {n} PCs", (n, f * 100), xytext=(4, 4), textcoords="offset points",
                        fontsize=8, color=MUTED)
    a1.set_xlabel("principal component")
    a1.set_ylabel("variance explained (%)")
    a1.set_ylim(0, 100)
    a1.set_title("PCA of smoothed, z-scored activity")
    a1.legend(frameon=False, fontsize=8, loc="center right")
    a2.plot(fa_k, fa_ll, color=SERIES[1], lw=2, marker="o", ms=5)
    a2.axvline(fa_best, color=MUTED, lw=0.8, ls=":")
    a2.annotate(f"best: {fa_best} factors", (fa_best, max(fa_ll)), xytext=(4, -12), textcoords="offset points",
                fontsize=8, color=MUTED)
    a2.set_xlabel("number of factors")
    a2.set_ylabel("held-out log-likelihood / bin")
    a2.set_title("Factor analysis, blocked cross-validation")
    fig.suptitle(f"{sid}: how many dimensions does the population use?", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def cebra_loss(sid: str, curves: list[tuple[str, np.ndarray, float | None]], out: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for k, (name, loss, gof) in enumerate(curves):
        if loss is None or not len(loss):
            continue
        w = max(1, len(loss) // 100)
        sm = np.convolve(loss, np.ones(w) / w, mode="valid")
        lab = name + (f" (GoF {gof:.2f} bits)" if gof is not None and np.isfinite(gof) else "")
        ax.plot(np.arange(sm.size) + w, sm, lw=2, color=SERIES[k], label=lab)
    ax.set_xlabel("training iteration")
    ax.set_ylabel("InfoNCE loss (lower = better)")
    ax.set_title(f"{sid}: CEBRA training loss vs the shuffled-label control", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def gpfa_trajectories(sid: str, traj: np.ndarray, groups: np.ndarray | None, group_name: str, bin_s: float,
                      offset_s: float, out: Path, max_traj: int = 60) -> Path:
    """GPFA latent trajectories (first two orthonormalised latents vs time, and latent 1 vs 2),
    trial-averaged per condition when a condition is given."""
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    tt = offset_s + (np.arange(traj.shape[2]) + 0.5) * bin_s
    if groups is not None:
        levels = [g for g in dict.fromkeys(groups.tolist()) if g is not None][: len(SERIES)]
    else:
        levels = [None]
    for li, g in enumerate(levels):
        m = np.ones(traj.shape[0], bool) if g is None else (groups == g)
        col = SERIES[li]
        sel = np.flatnonzero(m)[: max_traj // max(1, len(levels))]
        for i in sel:
            axes[2].plot(traj[i, 0], traj[i, 1], color=col, lw=0.5, alpha=0.25)
        mean = traj[m].mean(axis=0)
        lab = "all segments" if g is None else f"{group_name} = {g} (n={int(m.sum())})"
        for d in range(min(2, traj.shape[1])):
            axes[d].plot(tt, mean[d], color=col, lw=2, label=lab)
        axes[2].plot(mean[0], mean[1], color=col, lw=2.5)
    for d in range(2):
        axes[d].set_xlabel("time in segment (s)" if offset_s == 0 else "time from trial alignment (s)")
        axes[d].set_ylabel(f"latent {d + 1}")
        if offset_s != 0:
            axes[d].axvline(0, color=MUTED, lw=0.8, ls=":")
    axes[0].legend(frameon=False, fontsize=7)
    axes[2].set_xlabel("latent 1")
    axes[2].set_ylabel("latent 2")
    axes[2].set_title("single trials (thin) and means", fontsize=9)
    fig.suptitle(f"{sid}: GPFA latent trajectories", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def consistency_matrix(M: np.ndarray, names: list[str], title: str, out: Path) -> Path:
    plt = _plt()
    n = len(names)
    fig, ax = plt.subplots(figsize=(1.2 * n + 3, 1.0 * n + 2))
    im = ax.imshow(M * 100, vmin=0, vmax=100, cmap="Blues")
    ax.grid(False)
    ax.set_xticks(range(n), names, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(n), names, fontsize=8)
    for i in range(n):
        for j in range(n):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j] * 100:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > 0.6 else INK)
    fig.colorbar(im, ax=ax, label="consistency (R², %)")
    ax.set_xlabel("target session")
    ax.set_ylabel("source session")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def trial_trajectories(sid: str, tj: dict, out: Path) -> Path:
    """Condition-averaged PCA trajectories: PC1 and PC2 vs time, and PC1 vs PC2."""
    plt = _plt()
    traj, tt = tj["traj"], tj["time"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for k, c in enumerate(tj["conditions"][: len(SERIES)]):
        col = SERIES[k]
        lab = f"{c} (n={tj['counts'][k]})"
        for d in range(min(2, traj.shape[2])):
            axes[d].plot(tt, traj[k, :, d], color=col, lw=2, label=lab)
        axes[2].plot(traj[k, :, 0], traj[k, :, 1], color=col, lw=2)
        axes[2].scatter([traj[k, 0, 0]], [traj[k, 0, 1]], color=col, s=18, marker="o", zorder=3)
        z = int(np.argmin(np.abs(tt)))
        axes[2].scatter([traj[k, z, 0]], [traj[k, z, 1]], color=col, s=30, marker="^", zorder=3, edgecolors="white")
    for d in range(2):
        axes[d].axvline(0, color=MUTED, lw=0.8, ls=":")
        axes[d].set_xlabel(f"time from {tj['align']} (s)")
        axes[d].set_ylabel(f"PC {d + 1}")
    axes[0].legend(frameon=False, fontsize=7)
    axes[2].set_xlabel("PC 1")
    axes[2].set_ylabel("PC 2")
    axes[2].set_title("dot: window start, triangle: alignment event", fontsize=8)
    fig.suptitle(f"{sid}: trial-averaged population trajectories", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out
