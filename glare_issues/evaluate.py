"""Section 6: evaluation measures (problem filter, clusters versus human grouping, human ratings) and figures."""
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, cohen_kappa_score, confusion_matrix,
                             normalized_mutual_info_score, precision_recall_fscore_support)

# Published Mini-BAR results used for the indirect benchmark (Section 6.4).
# Source: Wei et al., "Zero-shot bilingual app reviews mining with large language models", ICTAI 2023,
# arXiv:2311.03058. Checked against the paper on Nov 8, 2026.
#   Table III: ChatGPT, "Problem Report" class (precision, recall, F1).
#   Table V:   Instructor embedding, the best one in the paper (ARI, NMI).
#   Table VI:  ChatGPT summaries, mean human rating 1 to 5.
MINIBAR_FILTER = {"Mini-BAR EN": (0.762, 0.972, 0.854), "Mini-BAR FR": (0.782, 0.973, 0.868)}
MINIBAR_CLUSTERS = {"Mini-BAR EN": (0.597, 0.713), "Mini-BAR FR": (0.357, 0.587), "Mini-BAR EN+FR": (0.308, 0.603)}
MINIBAR_RATINGS = {"relevance": 4.81, "consistency": 4.84, "fluency": 4.95, "coherence": 4.94}

RATING_DIMS = ["relevance", "consistency", "fluency", "coherence"]


# ----------------------------------------------------------------- Stage 1: problem filter
def filter_scores(y_true, y_pred):
    """Compare the LLM filter (y_pred) with the human reference labels (y_true), both 1 = problem report.

    Returns precision, recall and F1 for the problem class, weighted F1, Cohen's kappa between
    the filter and the humans, the 2x2 confusion matrix [[TN, FP], [FN, TP]] and the sample size.
    """
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    _, _, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {"precision": p, "recall": r, "f1": f, "weighted_f1": fw, "kappa_vs_human": cohen_kappa_score(y_true, y_pred),
            "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(), "n": int(len(y_true))}


def label_agreement(a, b):
    """Agreement between two annotators who gave the same reviews a 0/1 label.

    Returns Cohen's kappa, the raw share of identical labels and the number of reviews.
    Use it on the two annotator columns of the filter sheet, before the disagreements are settled.
    """
    a, b = np.asarray(a).astype(int), np.asarray(b).astype(int)
    return {"kappa": float(cohen_kappa_score(a, b)), "percent_agreement": float((a == b).mean()), "n": int(len(a))}


# ----------------------------------------------------------------- Stage 2: clusters
def external_scores(pred, truth):
    """ARI and NMI of the predicted clusters against the human grouping.

    HDBSCAN noise (label -1) is read in two ways:
    'all'      : noise points form one extra group (penalizes leaving reviews out);
    'clustered': only points that received a cluster are scored, and coverage is reported separately.
    """
    pred, truth = np.asarray(pred), np.asarray(truth)
    m = pred >= 0
    out = {"coverage": float(m.mean()), "ARI_all": adjusted_rand_score(truth, pred), "NMI_all": normalized_mutual_info_score(truth, pred)}
    if m.sum() > 1 and len(set(pred[m])) > 1:
        out["ARI_clustered"] = adjusted_rand_score(truth[m], pred[m])
        out["NMI_clustered"] = normalized_mutual_info_score(truth[m], pred[m])
    else:
        out["ARI_clustered"] = out["NMI_clustered"] = float("nan")
    return out


def grouping_agreement(a, b):
    """Agreement between two annotators who grouped the same reviews by issue.

    Group names are free text, so the same issue can have different names for each annotator.
    Kappa does not fit this case; ARI and NMI compare the groupings without matching the names.
    """
    a = pd.Series(a).astype(str).str.strip().to_numpy()   # hidden spaces or line breaks would split one issue into two
    b = pd.Series(b).astype(str).str.strip().to_numpy()
    return {"ARI": float(adjusted_rand_score(a, b)), "NMI": float(normalized_mutual_info_score(a, b)),
            "same_name_share": float((a == b).mean()), "n": int(len(a))}


# ----------------------------------------------------------------- Stage 3: issue descriptions
def rating_summary(ratings: pd.DataFrame):
    """Mean and SD of the 1 to 5 ratings per dimension, and agreement between all raters.

    ratings: one row per cluster per rater, with columns cluster (or cluster_id), rater,
    relevance, consistency, fluency, coherence. Agreement is the quadratic-weighted Cohen's kappa
    averaged over every pair of raters (not only the first two), computed on the clusters both rated.
    """
    ratings = ratings.rename(columns={"cluster": "cluster_id"})
    for d in RATING_DIMS:
        ratings[d] = pd.to_numeric(ratings[d], errors="coerce")
    raters = sorted(ratings["rater"].dropna().unique())
    res = {"n_raters": len(raters), "n_clusters": int(ratings["cluster_id"].nunique())}
    for d in RATING_DIMS:
        res[d] = {"mean": float(ratings[d].mean()), "sd": float(ratings[d].std(ddof=1))}
        kappas = []
        for r1, r2 in combinations(raters, 2):
            a = ratings[ratings["rater"] == r1].dropna(subset=[d]).set_index("cluster_id")[d]
            b = ratings[ratings["rater"] == r2].dropna(subset=[d]).set_index("cluster_id")[d]
            common = a.index.intersection(b.index)
            if len(common) > 1 and (a[common].nunique() > 1 or b[common].nunique() > 1):
                kappas.append(cohen_kappa_score(a[common].astype(int), b[common].astype(int),
                                                labels=[1, 2, 3, 4, 5], weights="quadratic"))
        res[d]["kappa_qw_mean_pairs"] = float(np.mean(kappas)) if kappas else float("nan")
        res[d]["n_pairs"] = len(kappas)
    return res


# ----------------------------------------------------------------- figures
def _plt():
    """Import matplotlib with a backend that works without a screen (Colab, servers)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_config_comparison(table: pd.DataFrame, path):
    """Bar charts of silhouette, noise share and number of clusters for every embedding and clusterer."""
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a, col, title in zip(ax, ["silhouette", "noise_ratio", "n_clusters"], ["Silhouette (non-noise)", "Noise share", "Clusters found"]):
        piv = table.pivot(index="embedding", columns="clusterer", values=col)
        piv.plot.bar(ax=a, rot=0)
        a.set_title(title); a.set_xlabel("")
    plt.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_rank_size(ranked: pd.DataFrame, path):
    """Size of each issue cluster in rank order, on a log scale."""
    plt = _plt()
    fig, a = plt.subplots(figsize=(5, 3.6))
    a.bar(range(1, len(ranked) + 1), ranked["occurrences"])
    a.set_yscale("log"); a.set_xlabel("Cluster rank"); a.set_ylabel("Review occurrences (log)")
    a.set_title("Issue clusters ranked by size")
    plt.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_umap2d(X, labels, path, seed):
    """2D UMAP map of the reviews, colored by cluster, with noise in grey."""
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
    """Confusion matrix of the problem filter (rows: human label, columns: LLM decision)."""
    plt = _plt()
    fig, a = plt.subplots(figsize=(3.6, 3.2))
    a.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            a.text(j, i, cm[i][j], ha="center", va="center")
    a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_xlabel("LLM"); a.set_ylabel("Human"); a.set_title("Problem filter")
    plt.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_benchmark(ours: dict, path, ref=""):
    """Our ARI and NMI next to the published Mini-BAR values (MINIBAR_CLUSTERS above).

    ours: {configuration name: (ARI, NMI)}. ref: citation shown in the title, for example "[12]",
    so it matches the reference number used in the report.
    """
    plt = _plt()
    vals = dict(MINIBAR_CLUSTERS)
    vals.update(ours)
    names = list(vals)
    fig, a = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(names))
    a.bar(x - 0.2, [vals[n][0] for n in names], 0.4, label="ARI")
    a.bar(x + 0.2, [vals[n][1] for n in names], 0.4, label="NMI")
    a.set_xticks(x); a.set_xticklabels(names, rotation=20, ha="right"); a.set_ylim(0, 1); a.legend()
    a.set_title(f"Indirect benchmark against Mini-BAR {ref}".strip())
    plt.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
