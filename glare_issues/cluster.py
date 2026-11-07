"""Stage 2b: UMAP -> HDBSCAN (and K-Means baseline), random-search tuning, representative reviews, ranking."""
import itertools
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def umap_reduce(X, n_neighbors, n_components, min_dist, seed, metric="cosine"):
    import umap
    n_neighbors = min(n_neighbors, max(2, len(X) - 1))
    return umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist,
                     metric=metric, random_state=seed).fit_transform(X)


def hdbscan_fit(Z, min_cluster_size, min_samples, cluster_selection_method):
    import hdbscan
    c = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                        cluster_selection_method=cluster_selection_method, metric="euclidean",
                        gen_min_span_tree=True, core_dist_n_jobs=1).fit(Z)
    try:
        rv = float(c.relative_validity_)
    except Exception:
        rv = float("nan")
    return c.labels_, rv


def internal_scores(Z, labels, seed=0, max_n=10000):
    """Silhouette on non-noise points (Euclidean in the reduced space), cluster count, noise share."""
    labels = np.asarray(labels)
    mask = labels >= 0
    k = len(set(labels[mask]))
    out = {"n_clusters": k, "noise_ratio": float(1 - mask.mean()), "silhouette": float("nan")}
    if k >= 2 and mask.sum() > k:
        n = min(max_n, int(mask.sum()))
        out["silhouette"] = float(silhouette_score(Z[mask], labels[mask], sample_size=n if n < mask.sum() else None,
                                                   random_state=seed))
    return out


def tune_hdbscan(X, cfg, log=print):
    """Random search over UMAP x HDBSCAN. Objective: relative validity (DBCV approximation), unsupervised.
    Trials with too much noise or too few clusters are rejected. Human labels are NOT used here."""
    rng = np.random.RandomState(cfg.seed)
    grid = list(itertools.product(
        cfg.umap_space["n_neighbors"], cfg.umap_space["n_components"], cfg.umap_space["min_dist"],
        cfg.hdbscan_space["min_cluster_size"], cfg.hdbscan_space["min_samples"], cfg.hdbscan_space["cluster_selection_method"]))
    picks = rng.choice(len(grid), size=min(cfg.n_trials, len(grid)), replace=False)
    umap_cache, trials, best, fallback = {}, [], None, None
    for idx in picks:
        nn, nc, md, mcs, ms, sel = grid[idx]
        key = (nn, nc, md)
        if key not in umap_cache:
            umap_cache[key] = umap_reduce(X, nn, nc, md, cfg.seed)
        Z = umap_cache[key]
        labels, rv = hdbscan_fit(Z, mcs, ms, sel)
        s = internal_scores(Z, labels, cfg.seed)
        ok = (s["noise_ratio"] <= cfg.max_noise_ratio and s["n_clusters"] >= cfg.min_clusters and np.isfinite(rv))
        rec = dict(n_neighbors=nn, n_components=nc, min_dist=md, min_cluster_size=mcs, min_samples=ms,
                   cluster_selection_method=sel, relative_validity=rv, accepted=bool(ok), **s)
        trials.append(rec)
        if s["n_clusters"] >= 2 and (fallback is None or s["noise_ratio"] < fallback[0]["noise_ratio"]):
            fallback = (rec, Z, labels)
        if ok and (best is None or rv > best["relative_validity"]):
            best, best_Z, best_labels = rec, Z, labels
    if best is None:   # nothing satisfied the limits: use the least noisy usable trial
        if fallback is None:
            raise RuntimeError("No trial found at least 2 clusters; widen the search space.")
        best, best_Z, best_labels = fallback
        log("   WARNING: no trial passed the limits, using the trial with the least noise.")
    log(f"   best HDBSCAN trial: {best}")
    return best, best_Z, best_labels, trials


def tune_kmeans(Z, cfg, log=print):
    """K-Means on the SAME reduced space as the tuned HDBSCAN run, k chosen by silhouette."""
    rows, best = [], None
    for k in cfg.kmeans_k_range:
        if k >= len(Z):
            continue
        labels = KMeans(n_clusters=k, n_init=10, random_state=cfg.seed).fit_predict(Z)
        s = internal_scores(Z, labels, cfg.seed)
        rows.append({"k": k, **s})
        if best is None or s["silhouette"] > best[1]["silhouette"]:
            best = ({"k": k}, s, labels)
    log(f"   best K-Means k={best[0]['k']} silhouette={best[1]['silhouette']:.3f}")
    return best[0], best[2], rows


def representatives(X, labels, m=10):
    """Indices of the m reviews closest (cosine) to each cluster centroid in the ORIGINAL embedding space."""
    reps = {}
    for c in sorted(set(labels) - {-1}):
        idx = np.where(labels == c)[0]
        cen = X[idx].mean(0)
        cen /= max(np.linalg.norm(cen), 1e-12)
        sim = X[idx] @ cen
        reps[int(c)] = idx[np.argsort(-sim)[:m]].tolist()
    return reps


def rank_clusters(labels, counts):
    """Cluster importance = number of review occurrences (unique texts weighted by their duplicate counts)."""
    import pandas as pd
    d = pd.DataFrame({"cluster": labels, "count": counts})
    d = d[d["cluster"] >= 0].groupby("cluster").agg(unique_reviews=("count", "size"), occurrences=("count", "sum"))
    return d.sort_values("occurrences", ascending=False).reset_index()


def cluster_top_terms(texts, labels, n_terms=8):
    """Explainability (Section 7): class-based TF-IDF as in BERTopic. Each cluster is one 'document' and
    W[t,c] = ( f[t,c] / sum_t' f[t',c] ) * log(1 + A / F[t]),  F[t] = sum_c f[t,c],  A = mean number of words per cluster."""
    from sklearn.feature_extraction.text import CountVectorizer
    labels = np.asarray(labels)
    cl = sorted(set(labels) - {-1})
    docs = [" ".join(np.asarray(texts, dtype=object)[labels == c]) for c in cl]
    cv = CountVectorizer(token_pattern=r"(?u)\b\w\w+\b", min_df=1)
    f = cv.fit_transform(docs).toarray().astype(float)
    A = f.sum(1).mean()
    F = np.clip(f.sum(0), 1e-12, None)
    w = (f / np.clip(f.sum(1, keepdims=True), 1, None)) * np.log(1 + A / F)
    vocab = np.array(cv.get_feature_names_out())
    return {int(c): vocab[np.argsort(-w[i])[:n_terms]].tolist() for i, c in enumerate(cl)}
