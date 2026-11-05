import numpy as np, pandas as pd
from glare_issues.preprocess import normalize_arabic, arabic_ratio, app_quotas
from glare_issues.evaluate import external_scores, filter_scores
from glare_issues.llm import _extract_json


def test_normalize():
    assert normalize_arabic("أهلاً  بكـــم https://x.co ٣") == "اهلا بكم 3"
    assert normalize_arabic("ممتاااااز") == "ممتاز"[:0] + "ممتااز"   # 3+ repeats -> 2, no stemming
    assert normalize_arabic(None) == ""


def test_ratio():
    assert arabic_ratio("تطبيق") == 1.0 and arabic_ratio("app") == 0.0 and arabic_ratio("12 !!") == 0.0


def test_quotas_cap_and_determinism():
    c = pd.Series({"a": 1000, "b": 50, "c": 20, "d": 5})
    q = app_quotas(c, 200)
    assert q.sum() == 200 and q["d"] == 5 and q["c"] == 20 and q["b"] == 50 and q["a"] == 125
    assert app_quotas(c.iloc[::-1], 200).sort_index().equals(q.sort_index())
    assert app_quotas(c, 5000).equals(c)


def test_external_scores_perfect_and_noise():
    truth = np.array([0, 0, 1, 1, 2, 2])
    s = external_scores(np.array([5, 5, 7, 7, 9, 9]), truth)
    assert s["ARI_all"] == 1 and s["NMI_all"] == 1 and s["coverage"] == 1
    truth2 = np.array([0, 0, 1, 1, 2, 3])          # the two noise points belong to different human groups
    s2 = external_scores(np.array([5, 5, 7, 7, -1, -1]), truth2)
    assert abs(s2["coverage"] - 4 / 6) < 1e-12 and s2["ARI_clustered"] == 1 and s2["ARI_all"] < 1


def test_filter_scores():
    s = filter_scores([1, 1, 0, 0], [1, 0, 0, 0])
    assert s["precision"] == 1 and s["recall"] == 0.5 and abs(s["f1"] - 2 / 3) < 1e-9


def test_json_extract():
    assert _extract_json('noise [{"id":0,"problem":1}] end') == [{"id": 0, "problem": 1}]
