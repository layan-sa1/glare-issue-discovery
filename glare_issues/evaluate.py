"""Section 6: evaluation measures (filter, clusters vs human grouping, human ratings) and figures."""
import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, cohen_kappa_score, confusion_matrix,
                             normalized_mutual_info_score, precision_recall_fscore_support)


def filter_scores(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {"precision": p, "recall": r, "f1": f, "weighted_f1": fw, "kappa_vs_human": cohen_kappa_score(y_true, y_pred),
            "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(), "n": int(len(y_true))}


def external_scores(pred, truth):
    """ARI / NMI against the human grouping. Two readings of HDBSCAN noise:
    'all'      : noise points form one extra group (penalises leaving reviews out)
    'clustered': only points that received a cluster are scored, coverage reported separately."""
    pred, truth = np.asarray(pred), np.asarray(truth)
    m = pred >= 0
    out = {"coverage": float(m.mean()), "ARI_all": adjusted_rand_score(truth, pred), "NMI_all": normalized_mutual_info_score(truth, pred)}
    if m.sum() > 1 and len(set(pred[m])) > 1:
        out["ARI_clustered"] = adjusted_rand_score(truth[m], pred[m])
        out["NMI_clustered"] = normalized_mutual_info_score(truth[m], pred[m])
    else:
        out["ARI_clustered"] = out["NMI_clustered"] = float("nan")
    return out


def rating_summary(ratings: pd.DataFrame):
    """ratings: cluster_id, rater, relevance, consistency, fluency, coherence (1-5). Mean, SD and quadratic-weighted kappa."""
    dims = ["relevance", "consistency", "fluency", "coherence"]
    res = {d: {"mean": float(ratings[d].mean()), "sd": float(ratings[d].std(ddof=1))} for d in dims}
    raters = sorted(ratings["rater"].unique())
    if len(raters) >= 2:
        a = ratings[ratings["rater"] == raters[0]].set_index("cluster_id")
        b = ratings[ratings["rater"] == raters[1]].set_index("cluster_id")
        common = a.index.intersection(b.index)
        for d in dims:
            res[d]["kappa_qw"] = float(cohen_kappa_score(a.loc[common, d], b.loc[common, d], weights="quadratic"))
    return res


# ----------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_config_comparison(table: pd.DataFrame, path):
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a, col, title in zip(ax, ["silhouette", "noise_ratio", "n_clusters"], ["Silhouette (non-noise)", "Noise share", "Clusters found"]):
        piv = table.pivot(index="embedding", columns="clusterer", values=col)
        piv.plot.bar(ax=a, rot=0)
        a.set_title(title); a.set_xlabel("")
    plt.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_rank_size(ranked: pd.DataFrame, path):
    plt = _plt()
    fig, a = plt.subplots(figsize=(5, 3.6))
    a.bar(range(1, len(ranked) + 1), ranked["occurrences"])
    a.set_yscale("log"); a.set_xlabel("Cluster rank"); a.set_ylabel("Review occurrences (log)")
    a.set_title("Issue clusters ranked by size")
    plt.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_umap2d(X, labels, path, seed):
    from .cluster import umap_reduce
    plt = _plt()
    Z = umap_reduce(X, 30, 2, 0.1, seed)
    fig, a = plt.subplots(figsize=(5.2, 4.2))
    m = labels < 0
    a.scatter(Z[m, 0], Z[m, 1], s=2, c="lightgrey", label="noise")
    a.scatter(Z[~m, 0], Z[~m, 1], s=2, c=labels[~m], cmap="tab20")
    a.set_title("UMAP projection of clustered reviews"); a.legend(markerscale=4)
    plt.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_confusion(cm, path):
    plt = _plt()
    fig, a = plt.subplots(figsize=(3.6, 3.2))
    a.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            a.text(j, i, cm[i][j], ha="center", va="center")
    a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_xlabel("LLM"); a.set_ylabel("Human"); a.set_title("Problem filter")
    plt.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_benchmark(ours: dict, path):
    """Our ARI/NMI next to the published Mini-BAR values (Wei et al., 2023; best embedding per language)."""
    plt = _plt()
    ref = {"Mini-BAR EN": (0.597, 0.713), "Mini-BAR FR": (0.357, 0.587), "Mini-BAR EN+FR": (0.308, 0.603)}
    for k, v in ours.items():
        ref[k] = v
    names = list(ref)
    fig, a = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(names))
    a.bar(x - 0.2, [ref[n][0] for n in names], 0.4, label="ARI")
    a.bar(x + 0.2, [ref[n][1] for n in names], 0.4, label="NMI")
    a.set_xticks(x); a.set_xticklabels(names, rotation=20, ha="right"); a.set_ylim(0, 1); a.legend()
    a.set_title("Indirect benchmark against Mini-BAR [16]")
    plt.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
