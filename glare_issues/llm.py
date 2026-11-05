"""Stage 1 (problem filter) and Stage 3 (issue summaries) with an LLM. Zero-shot, temperature 0, cached on disk."""
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List

FILTER_SYSTEM = (
    "You label Arabic mobile-app reviews. A review is a PROBLEM REPORT (1) if it describes something that does not "
    "work or went wrong for the user: a bug, crash, failure, slowness, login/account/payment/billing trouble, bad "
    "delivery or support experience, or a missing feature the user complains about. It is NOT a problem report (0) if "
    "it is only praise, thanks, a neutral remark, or a vague complaint with no stated issue (for example 'سيئ جدا'). "
    "A positive review that also mentions a problem counts as 1. Use only the review text. "
    'Answer with JSON only: a list like [{"id": 0, "problem": 1}, ...] containing every id.'
)

SUMMARY_SYSTEM = (
    "You write short issue descriptions for app developers from Arabic user reviews that were grouped by meaning. "
    "Use ONLY what the reviews say. Do not guess causes, versions or features that are not mentioned. If the reviews "
    "do not share one clear issue, set label to 'Mixed' and say so. "
    'Answer with JSON only: {"label": "<=6 English words", "summary_ar": "one or two Arabic sentences", '
    '"evidence_ids": [ids of the 3 reviews that best support the summary]}.'
)


def _extract_json(text: str):
    text = text.strip()
    m = re.search(r"(\[.*\]|\{.*\})", text, re.S)
    if not m:
        raise ValueError(f"no JSON in LLM output: {text[:200]}")
    return json.loads(m.group(1))


class LLMClient:
    def __init__(self, cache_path: str):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, str] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                self.cache[rec["k"]] = rec["v"]
        self.n_calls = 0

    name = "base"

    def _raw(self, system: str, user: str) -> str:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def complete(self, system: str, user: str) -> str:
        key = hashlib.sha256(f"{self.name}||{system}||{user}".encode("utf-8")).hexdigest()
        if key in self.cache:
            return self.cache[key]
        out = self._raw(system, user)
        self.n_calls += 1
        self.cache[key] = out
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"k": key, "v": out}, ensure_ascii=False) + "\n")
        return out

    # ---------------- task wrappers ----------------
    def classify_problem(self, texts: List[str], batch_size: int = 20) -> List[int]:
        labels: List[int] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            user = "\n".join(f"{j}\t{t}" for j, t in enumerate(chunk))
            for attempt in range(3):
                try:
                    res = _extract_json(self.complete(FILTER_SYSTEM + ("" if attempt == 0 else " "), user))
                    got = {int(r["id"]): int(r["problem"]) for r in res}
                    labels.extend(got[j] for j in range(len(chunk)))
                    break
                except Exception:
                    if attempt == 2:
                        raise
        return labels

    def summarise_cluster(self, reviews: List[str], size: int) -> Dict:
        user = f"Cluster size: {size} reviews. Sample reviews:\n" + "\n".join(f"{j}\t{t}" for j, t in enumerate(reviews))
        return _extract_json(self.complete(SUMMARY_SYSTEM, user))


class AnthropicLLM(LLMClient):
    name = "anthropic"

    def __init__(self, cache_path, model):
        super().__init__(cache_path)
        import anthropic  # pip install anthropic ; needs ANTHROPIC_API_KEY
        self.client, self.model = anthropic.Anthropic(), model
        self.name = f"anthropic:{model}"

    def _raw(self, system, user):
        r = self.client.messages.create(model=self.model, max_tokens=1500, temperature=0,
                                        system=system, messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


class OpenAILLM(LLMClient):
    name = "openai"

    def __init__(self, cache_path, model):
        super().__init__(cache_path)
        from openai import OpenAI  # pip install openai ; needs OPENAI_API_KEY (or a compatible base URL)
        self.client, self.model = OpenAI(), model
        self.name = f"openai:{model}"

    def _raw(self, system, user):
        r = self.client.chat.completions.create(model=self.model, temperature=0, messages=[
            {"role": "system", "content": system}, {"role": "user", "content": user}])
        return r.choices[0].message.content


class MockLLM(LLMClient):
    """Keyword rules. ONLY for testing that the pipeline runs end to end. Never use it to report results."""
    name = "mock"
    PROBLEM_WORDS = ("مشكل", "ما يشتغل", "لا يعمل", "ما اقدر", "تعليق", "يعلق", "بطيء", "بطي", "تعطل", "يقفل", "خطا", "خطأ",
                     "فشل", "ما يفتح", "مو شغال", "ما وصل", "تحديث", "الدفع", "تسجيل الدخول")

    def _raw(self, system, user):
        if system.startswith("You label"):
            rows = [ln.split("\t", 1) for ln in user.splitlines()]
            return json.dumps([{"id": int(i), "problem": int(any(w in t for w in self.PROBLEM_WORDS))} for i, t in rows])
        lines = [ln.split("\t", 1) for ln in user.splitlines()[1:]]
        return json.dumps({"label": "Mock label", "summary_ar": "ملخص تجريبي: " + lines[0][1][:60],
                           "evidence_ids": [int(l[0]) for l in lines[:3]]}, ensure_ascii=False)


def make_llm(cfg) -> LLMClient:
    cache = os.path.join(cfg.cache_dir, f"llm_cache_{cfg.llm_backend}.jsonl")
    model = cfg.llm_model or os.environ.get("LLM_MODEL", "")
    if cfg.llm_backend == "mock":
        return MockLLM(cache)
    if not model:
        raise ValueError("Set cfg.llm_model or the LLM_MODEL environment variable.")
    return {"anthropic": AnthropicLLM, "openai": OpenAILLM}[cfg.llm_backend](cache, model)
