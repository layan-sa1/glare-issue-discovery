"""Section 2.2: conservative Arabic normalization, language filter, duplicate merging with counts, per-app capping."""
import re
import numpy as np
import pandas as pd

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_DIACRITICS = re.compile(r"[ؗ-ًؚ-ْٰ]")
_TATWEEL = "ـ"
_ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_REPEAT = re.compile(r"(.)\1{2,}")                       # 3+ repeated characters -> 2
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿️‍]")
_SPACES = re.compile(r"\s+")
_AR_LETTER = re.compile(r"[ء-ي]")
_ANY_LETTER = re.compile(r"[A-Za-zء-ي]")


def normalize_arabic(text: str) -> str:
    """Conservative normalization: dialect wording and word forms are kept (no stemming, no ta-marbuta/ya mapping)."""
    if not isinstance(text, str):
        return ""
    t = _URL.sub(" ", text)
    t = _EMAIL.sub(" ", t)
    t = _EMOJI.sub(" ", t)
    t = _DIACRITICS.sub("", t).replace(_TATWEEL, "")
    t = t.translate(_ALEF).translate(_DIGITS)
    t = _REPEAT.sub(r"\1\1", t)
    return _SPACES.sub(" ", t).strip()


def arabic_ratio(text: str) -> float:
    """Share of Arabic letters among all Arabic and Latin letters in the text (0 if there are none)."""
    letters = _ANY_LETTER.findall(text)
    if not letters:
        return 0.0
    return len(_AR_LETTER.findall(text)) / len(letters)


def app_quotas(counts: pd.Series, target: int) -> pd.Series:
    """Water-filling cap: q_a = min(N_a, c) with the largest c such that sum_a q_a <= target.
    The remainder is given, one review each, to the largest apps (ties broken by app_id) so the result is deterministic."""
    counts = counts.astype(int)
    if counts.sum() <= target:
        return counts
    lo, hi = 0, int(counts.max())
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if np.minimum(counts.values, mid).sum() <= target:
            lo = mid
        else:
            hi = mid - 1
    q = np.minimum(counts, lo)
    rem = target - int(q.sum())
    if rem > 0:
        order = counts[counts > lo].reset_index()
        order.columns = ["app", "n"]
        order = order.sort_values(["n", "app"], ascending=[False, True]).head(rem)
        for a in order["app"]:
            q[a] += 1
    return q


def preprocess(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Normalize, filter by language and length, merge duplicates inside each app, and cap each app's share of the sample."""
    d = df[[cfg.col_id, cfg.col_app, cfg.col_text, cfg.col_rating]].copy()
    d.columns = ["review_id", "app_id", "raw_text", "rating"]
    d["text"] = d["raw_text"].map(normalize_arabic)
    d = d[d["text"].str.len() > 0]
    d = d[d["text"].map(arabic_ratio) >= cfg.min_arabic_ratio]
    d["n_words"] = d["text"].str.split().str.len()
    d = d[d["n_words"] >= cfg.min_words]
    # merge duplicates inside each app and keep how often each text occurs
    g = d.groupby(["app_id", "text"], sort=True)
    out = g.agg(review_id=("review_id", "first"), raw_text=("raw_text", "first"),
                rating_mean=("rating", "mean"), count=("review_id", "size"),
                n_words=("n_words", "first")).reset_index()
    # per-app cap on unique texts (deterministic sample)
    quotas = app_quotas(out.groupby("app_id").size(), cfg.target_sample)
    rng = np.random.RandomState(cfg.seed)
    parts = []
    for app, sub in out.groupby("app_id", sort=True):
        q = int(quotas.get(app, 0))
        if q >= len(sub):
            parts.append(sub)
        elif q > 0:
            parts.append(sub.iloc[np.sort(rng.choice(len(sub), q, replace=False))])
    res = pd.concat(parts, ignore_index=True)
    res.insert(0, "uid", np.arange(len(res)))
    return res
