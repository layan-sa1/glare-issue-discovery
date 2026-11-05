"""Run the full methodology. Examples:
    python run_pipeline.py --demo                      # synthetic data + mock LLM + offline embedder (smoke test, NOT results)
    python run_pipeline.py --input data/glare_sample.csv --llm anthropic --stage all
Stages: prepare, filter, annotate, cluster, summarize, evaluate, all
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from glare_issues import cluster as C, evaluate as E
from glare_issues.config import Config
from glare_issues.embed import get_embeddings
from glare_issues.llm import make_llm
from glare_issues.preprocess import preprocess


def P(cfg, name):
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    return os.path.join(cfg.out_dir, name)


def stage_prepare(cfg, df):
    d = preprocess(df, cfg)
    d.to_csv(P(cfg, "01_preprocessed.csv"), index=False)
    print(f"[prepare] {len(df):,} raw -> {len(d):,} unique texts ({int(d['count'].sum()):,} occurrences)")
    return d


def stage_filter(cfg, d, llm):
    d = d.copy()
    d["is_problem"] = llm.classify_problem(d["text"].tolist(), cfg.filter_batch_size)
    d.to_csv(P(cfg, "02_filtered_all.csv"), index=False)
    kept = d[d["is_problem"] == 1].reset_index(drop=True)
    kept.to_csv(P(cfg, "02_problem_reviews.csv"), index=False)
    print(f"[filter] {len(kept):,} of {len(d):,} unique texts kept as problem reports (LLM calls: {llm.n_calls})")
    return d, kept


def stage_annotate(cfg, d_all, kept, n_filter=400, n_group=300):
    """Blank sheets for the human parts of the evaluation (filter labels, issue grouping)."""
    Path(cfg.out_dir, "annotations").mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(cfg.seed)
    f = d_all.iloc[rng.choice(len(d_all), min(n_filter, len(d_all)), replace=False)][["uid", "text"]]
    f.assign(human_problem="").to_csv(P(cfg, "annotations/filter_sheet.csv"), index=False)
    g = kept.iloc[rng.choice(len(kept), min(n_group, len(kept)), replace=False)][["uid", "text"]]
    g.assign(issue_group="").to_csv(P(cfg, "annotations/grouping_sheet.csv"), index=False)
    print("[annotate] wrote annotations/filter_sheet.csv and grouping_sheet.csv (fill the last column, keep the file name + _done)")


def stage_cluster(cfg, kept, embedders):
    texts = kept["text"].tolist()
    rows, store = [], {}
    for name, spec in embedders.items():
        print(f"[cluster] embedding with {name}")
        X = get_embeddings(texts, name, spec, cfg)
        best, Z, lab_h, trials = C.tune_hdbscan(X, cfg)
        pd.DataFrame(trials).to_csv(P(cfg, f"03_trials_hdbscan_{name}.csv"), index=False)
        kbest, lab_k, krows = C.tune_kmeans(Z, cfg)
        pd.DataFrame(krows).to_csv(P(cfg, f"03_trials_kmeans_{name}.csv"), index=False)
        sk = C.internal_scores(Z, lab_k, cfg.seed)
        rows.append({"embedding": name, "clusterer": "HDBSCAN", **{k: best[k] for k in ("silhouette", "noise_ratio", "n_clusters", "relative_validity")},
                     "params": json.dumps({k: best[k] for k in ("n_neighbors", "n_components", "min_dist", "min_cluster_size", "min_samples", "cluster_selection_method")})})
        rows.append({"embedding": name, "clusterer": "K-Means", "silhouette": sk["silhouette"], "noise_ratio": 0.0,
                     "n_clusters": sk["n_clusters"], "relative_validity": float("nan"), "params": json.dumps(kbest)})
        store[(name, "HDBSCAN")] = (X, lab_h)
        store[(name, "K-Means")] = (X, lab_k)
        kept[f"cl_{name}_HDBSCAN"] = lab_h
        kept[f"cl_{name}_KMeans"] = lab_k
    table = pd.DataFrame(rows)
    table.to_csv(P(cfg, "03_config_comparison.csv"), index=False)
    kept.to_csv(P(cfg, "03_clustered.csv"), index=False)
    E.fig_config_comparison(table, P(cfg, "fig_config_comparison.png"))
    print(table[["embedding", "clusterer", "silhouette", "noise_ratio", "n_clusters"]].to_string(index=False))
    return table, store


def stage_summarize(cfg, kept, store, table, llm, choice=None):
    if choice is None:   # default: highest relative validity among HDBSCAN configurations
        h = table[table.clusterer == "HDBSCAN"].sort_values("relative_validity", ascending=False)
        choice = (h.iloc[0]["embedding"], "HDBSCAN")
    X, labels = store[choice]
    reps = C.representatives(X, labels, cfg.summary_samples)
    ranked = C.rank_clusters(labels, kept["count"].values)
    terms = C.cluster_top_terms(kept["text"].tolist(), labels)
    out = []
    for _, r in ranked.head(cfg.summary_top_clusters).iterrows():
        c = int(r["cluster"])
        sample = kept.iloc[reps[c]]["text"].tolist()
        s = llm.summarise_cluster(sample, int(r["unique_reviews"]))
        out.append({"rank": len(out) + 1, "cluster": c, "unique_reviews": int(r["unique_reviews"]),
                    "occurrences": int(r["occurrences"]), "label": s.get("label"), "summary_ar": s.get("summary_ar"),
                    "top_terms": " ".join(terms[c]),
                    "sample_reviews": " || ".join(sample[:3])})
    res = pd.DataFrame(out)
    res.to_csv(P(cfg, "04_ranked_issues.csv"), index=False, encoding="utf-8-sig")
    E.fig_rank_size(ranked, P(cfg, "fig_cluster_sizes.png"))
    E.fig_umap2d(X, labels, P(cfg, "fig_umap2d.png"), cfg.seed)
    pd.DataFrame({"cluster": res["cluster"], "label": res["label"], "summary_ar": res["summary_ar"]}).assign(
        rater="", relevance="", consistency="", fluency="", coherence="").to_csv(
        P(cfg, "annotations/summary_rating_sheet.csv"), index=False, encoding="utf-8-sig")
    print(f"[summarize] {choice}: {len(res)} issue descriptions written (LLM calls: {llm.n_calls})")
    return res


def stage_evaluate(cfg, kept, d_all):
    report, ann = {}, Path(cfg.out_dir, "annotations")
    f = ann / "filter_sheet_done.csv"
    if f.exists():
        h = pd.read_csv(f).dropna(subset=["human_problem"]).merge(d_all[["uid", "is_problem"]], on="uid")
        report["filter"] = E.filter_scores(h["human_problem"].astype(int), h["is_problem"].astype(int))
        E.fig_confusion(report["filter"]["confusion"], P(cfg, "fig_filter_confusion.png"))
    g = ann / "grouping_sheet_done.csv"
    if g.exists():
        h = pd.read_csv(g).dropna(subset=["issue_group"]).merge(kept, on="uid", suffixes=("", "_k"))
        ext, ours = [], {}
        for col in [c for c in kept.columns if c.startswith("cl_")]:
            s = E.external_scores(h[col].values, h["issue_group"].values)
            ext.append({"config": col[3:], **s})
            ours[col[3:].replace("_", " ")] = (s["ARI_all"], s["NMI_all"])
        pd.DataFrame(ext).to_csv(P(cfg, "05_external_scores.csv"), index=False)
        report["external"] = ext
        E.fig_benchmark(ours, P(cfg, "fig_benchmark.png"))
    r = ann / "summary_rating_sheet_done.csv"
    if r.exists():
        report["ratings"] = E.rating_summary(pd.read_csv(r).dropna(subset=["relevance"]))
    json.dump(report, open(P(cfg, "05_evaluation_report.json"), "w"), indent=2, ensure_ascii=False, default=float)
    print("[evaluate]", "wrote 05_evaluation_report.json" if report else "no completed annotation sheets found yet")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input"); ap.add_argument("--out", default="results"); ap.add_argument("--llm", default="mock")
    ap.add_argument("--model", default=""); ap.add_argument("--stage", default="all")
    ap.add_argument("--target", type=int, default=100_000); ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    cfg = Config(out_dir=a.out, llm_backend=a.llm, llm_model=a.model, target_sample=a.target, n_trials=a.trials)
    embedders = cfg.embedders
    if a.demo:
        from glare_issues.demo_data import make_demo_reviews
        cfg.out_dir, cfg.cache_dir, cfg.llm_backend = "demo_results", "demo_cache", "mock"
        cfg.n_trials, cfg.kmeans_k_range = 8, [5, 7, 10, 15]
        cfg.umap_space = {"n_neighbors": [10, 15], "n_components": [5, 10], "min_dist": [0.0, 0.1]}
        cfg.hdbscan_space = {"min_cluster_size": [10, 20, 40], "min_samples": [1, 5], "cluster_selection_method": ["eom", "leaf"]}
        df = make_demo_reviews()
        embedders = {"DemoTFIDF-A": {"kind": "tfidf", "name": ""}}
        print("DEMO MODE: synthetic data, mock LLM, offline embedder. These numbers are NOT research results.")
    else:
        df = pd.read_csv(a.input)
    llm = make_llm(cfg)
    d = stage_prepare(cfg, df)
    d_all, kept = stage_filter(cfg, d, llm)
    stage_annotate(cfg, d_all, kept)
    table, store = stage_cluster(cfg, kept, embedders)
    stage_summarize(cfg, kept, store, table, llm)
    if a.demo:   # the synthetic data carries the true issue, so the evaluation code can be exercised
        truth = df.drop_duplicates("content").set_index("content")["true_issue"]
        raw = d.set_index("uid")["raw_text"]
        kept["issue_group"] = kept["uid"].map(raw).map(truth)
        Path(cfg.out_dir, "annotations").mkdir(exist_ok=True, parents=True)
        kept.loc[kept.issue_group != "none", ["uid", "issue_group"]].to_csv(P(cfg, "annotations/grouping_sheet_done.csv"), index=False)
    stage_evaluate(cfg, kept, d_all)


if __name__ == "__main__":
    main()
