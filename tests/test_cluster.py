from types import SimpleNamespace
import numpy as np
from sklearn.datasets import make_blobs
from glare_issues.cluster import internal_scores, rank_clusters, tune_hdbscan, tune_kmeans


def test_noise_share():
    Z = np.array([[0, 0], [0, 1], [9, 9], [9, 8], [50, 50]], dtype=float)
    s = internal_scores(Z, [0, 0, 1, 1, -1])
    assert s["n_clusters"] == 2 and abs(s["noise_ratio"] - 0.2) < 1e-9

def test_ranking():
    r = rank_clusters(np.array([0, 0, 1, -1]), np.array([1, 2, 10, 99]))
    assert r["cluster"].tolist() == [1, 0]


def test_kmeans_finds_three_groups():
    Z, _ = make_blobs(n_samples=300, centers=3, random_state=42)
    cfg = SimpleNamespace(seed=42, kmeans_k_range=[2, 3, 4])
    best, _, _ = tune_kmeans(Z, cfg)
    assert best["k"] == 3


def test_fallback_when_no_trial_passes():
    X, _ = make_blobs(n_samples=300, centers=3, n_features=10, random_state=42)
    cfg = SimpleNamespace(seed=42, n_trials=2, max_noise_ratio=1.0, min_clusters=1000,
                          umap_space={"n_neighbors": [15], "n_components": [5], "min_dist": [0.0]},
                          hdbscan_space={"min_cluster_size": [10, 20], "min_samples": [5],
                                         "cluster_selection_method": ["eom"]})
    best, _, labels, _ = tune_hdbscan(X, cfg)
    assert best["accepted"] is False and len(labels) == 300
