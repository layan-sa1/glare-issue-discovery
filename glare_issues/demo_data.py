"""Synthetic Arabic reviews with known issue labels. DEMO ONLY: it lets us check that the code runs; it produces no research result."""
import numpy as np
import pandas as pd

ISSUES = {
    "login":    ["ما اقدر ادخل حسابي", "مشكلة في تسجيل الدخول", "يطلب مني كلمة المرور كل مرة", "التطبيق ما يقبل رمز التحقق"],
    "payment":  ["مشكلة في الدفع", "الدفع بالبطاقة يفشل", "خصم المبلغ مرتين", "ما يقبل بطاقة مدى"],
    "crash":    ["التطبيق يقفل لحاله", "يتعطل بعد ما افتحه", "ينغلق فجأة", "ما يفتح ويطلعني برا"],
    "slow":     ["التطبيق بطيء جدا", "يعلق كثير وبطيء", "التحميل ياخذ وقت طويل", "بطيء وما يستجيب"],
    "update":   ["بعد التحديث صار فيه مشاكل", "التحديث الاخير خرب التطبيق", "ارجعوا النسخة القديمة بعد التحديث", "التحديث الجديد ما يشتغل"],
    "delivery": ["الطلب تاخر ساعتين", "التوصيل متاخر دايم", "السائق ما وصل الطلب", "الطلب وصل ناقص"],
    "support":  ["خدمة العملاء ما ترد", "الدعم الفني بطيء في الرد", "ارسلت شكوى وما احد رد", "ما فيه تجاوب من الدعم"],
}
PRAISE = ["تطبيق ممتاز", "رائع جدا", "خدمة حلوة وسريعة", "شكرا لكم", "افضل تطبيق", "سهل الاستخدام والله يعطيكم العافية"]
VAGUE = ["سيئ جدا", "ما عجبني", "مو حلو"]
PRE = ["", "والله ", "للاسف ", "كل مرة ", "من امس "]
POST = ["", " ارجو الحل", " ضيعتوا وقتي", " جدا", " ياريت تصلحونها"]


def make_demo_reviews(n=4000, n_apps=12, seed=0):
    rng = np.random.RandomState(seed)
    apps = [f"com.demo.app{i:02d}" for i in range(n_apps)]
    weights = np.array([1 / (i + 1) for i in range(n_apps)]); weights /= weights.sum()
    rows, keys = [], list(ISSUES)
    for i in range(n):
        r = rng.rand()
        if r < 0.42:
            issue = keys[rng.randint(len(keys))]
            text = rng.choice(PRE) + rng.choice(ISSUES[issue]) + rng.choice(POST)
            rating = int(rng.choice([1, 1, 2, 3]))
        elif r < 0.92:
            issue, text, rating = "none", rng.choice(PRAISE) + rng.choice(POST), 5
        else:
            issue, text, rating = "none", rng.choice(VAGUE), 1
        rows.append({"review_id": f"d{i}", "app_id": str(rng.choice(apps, p=weights)), "content": text,
                     "rating": rating, "true_issue": issue})
    return pd.DataFrame(rows)
