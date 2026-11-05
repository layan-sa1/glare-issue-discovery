"""Stage 2a: review -> vector. Three model families are compared on the same reviews."""
import hashlib
from pathlib import Path
import numpy as np


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def embed_hf_mean(texts, model_name, max_len=128, batch_size=64):
    """BERT-style encoder (AraBERT, MARBERT): mean of last-layer token vectors, padding masked out."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = AutoTokenizer.from_pretrained(model_name), AutoModel.from_pretrained(model_name).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
    return np.vstack(out)


def embed_st(texts, model_name, batch_size=64):
    """Sentence-transformer trained for semantic similarity (one vector per sentence)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name).encode(texts, batch_size=batch_size, show_progress_bar=False,
                                                 convert_to_numpy=True)


def embed_tfidf_svd(texts, dim=64, seed=0):
    """Offline stand-in used ONLY by the demo/tests (no model download needed)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2).fit_transform(texts)
    return TruncatedSVD(min(dim, X.shape[1] - 1), random_state=seed).fit_transform(X)


def get_embeddings(texts, key, spec, cfg):
    """Cached, L2-normalised embeddings. spec = {'kind': ..., 'name': ...}."""
    Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
    h = hashlib.md5(("\n".join(texts)).encode("utf-8")).hexdigest()[:10]
    path = Path(cfg.cache_dir) / f"emb_{key}_{h}.npy"
    if path.exists():
        return np.load(path)
    kind = spec["kind"]
    if kind == "hf_mean":
        X = embed_hf_mean(texts, spec["name"], cfg.max_seq_len, cfg.embed_batch_size)
    elif kind == "st":
        X = embed_st(texts, spec["name"], cfg.embed_batch_size)
    elif kind == "tfidf":
        X = embed_tfidf_svd(texts, seed=cfg.seed)
    else:
        raise ValueError(kind)
    X = _l2(X.astype(np.float32))
    np.save(path, X)
    return X
