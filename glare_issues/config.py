"""Central configuration. Every value used in the report lives here so the experiments are reproducible."""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Config:
    # ---------- paths ----------
    input_csv: str = "data/glare_sample.csv"        # engineered GLARE CSV (or a chunk/sample of it)
    out_dir: str = "results"
    cache_dir: str = "cache"
    seed: int = 42

    # ---------- column names in the engineered GLARE file ----------
    col_id: str = "review_id"
    col_app: str = "app_id"
    col_text: str = "content"
    col_rating: str = "rating"
    col_date: str = "at"

    # ---------- preprocessing ----------
    min_arabic_ratio: float = 0.5        # keep reviews with >= 50% Arabic letters (same rule as the EDA)
    min_words: int = 2                   # drop one-word reviews (same rule as the pilot)
    target_sample: int = 30_000         # N: reviews kept after per-app capping (unique texts)
    sampling: str = "uniform"            # per-app cap: "uniform" (default, used by the team) or "sqrt" (comparison only)

    # ---------- LLM (problem filter + summaries) ----------
    llm_backend: str = "gemini"          # "mock" | "anthropic" | "openai" | "gemini"
    llm_model: str = "gemini-3.1-flash-lite"    # e.g. set through the LLM_MODEL environment variable
    filter_batch_size: int = 20
    summary_samples: int = 10            # m representative reviews per cluster
    summary_top_clusters: int = 30       # summarise the largest clusters only

    # ---------- embeddings (model name -> loader kind) ----------
    # IDs below are the public Hugging Face checkpoints named in the cited papers; check availability before running.
    embedders: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "AraBERT":  {"kind": "hf_mean", "name": "aubmindlab/bert-base-arabertv02"},
        "MARBERT":  {"kind": "hf_mean", "name": "UBC-NLP/MARBERT"},
        # Arabic sentence-transformer: set to the Matryoshka/NLI-tuned checkpoint of Nacar & Koubaa [9]
        # or another Arabic sentence-similarity model, e.g. a fine-tuned paraphrase-multilingual-mpnet-base-v2.
        "ArabicSBERT": {"kind": "st", "name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"},
    })
    max_seq_len: int = 128
    embed_batch_size: int = 64

    # ---------- optimisation (random search spaces) ----------
    n_trials: int = 30
    umap_space: Dict[str, List] = field(default_factory=lambda: {
        "n_neighbors": [10, 15, 30, 50],
        "n_components": [5, 10, 15, 25],
        "min_dist": [0.0, 0.05, 0.1],
    })
    hdbscan_space: Dict[str, List] = field(default_factory=lambda: {
        "min_cluster_size": [10, 20, 30, 50, 100],
        "min_samples": [1, 5, 10, 20],
        "cluster_selection_method": ["eom", "leaf"],
    })
    kmeans_k_range: List[int] = field(default_factory=lambda: [10, 15, 20, 30, 40, 50, 75, 100])
    max_noise_ratio: float = 0.60        # trials with more noise than this are rejected
    min_clusters: int = 5                # trials with fewer clusters are rejected
