"""Hàm dùng chung cho GUI ITViec — chép NGUYÊN logic từ notebook project_2
(project_2_recommendation_ITViec_dachay.ipynb, Mục 3, 6, 7, 8) để app và notebook
xử lý văn bản giống hệt nhau."""
import re
import unicodedata

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    from pyvi import ViTokenizer
    HAS_PYVI = True
except Exception:          # thiếu pyvi -> vẫn chạy, chỉ mất bước ghép từ tiếng Việt
    HAS_PYVI = False

RANDOM_STATE = 42
TOP_K = 5

# ======================================================================
# 1. Từ điển chuẩn hoá (Mục 3.1 notebook)
# ======================================================================
TECH_MAP = {   # chạy TRƯỚC khi bỏ dấu câu
    "c++": " cplusplus ", "c#": " csharp ", "asp.net": " aspdotnet ", ".net": " dotnet ",
    "node.js": " nodejs ", "vue.js": " vuejs ", "react.js": " reactjs ", "next.js": " nextjs ",
    "objective-c": " objectivec ", "ci/cd": " cicd ", "qa/qc": " qaqc ", "ui/ux": " uiux ",
    "r&d": " rnd ", "e-commerce": " ecommerce ", "front-end": " frontend ", "back-end": " backend ",
    "full-stack": " fullstack ",
}

TEENCODE = {   # chỉ áp dụng cho văn bản tiếng Việt
    "ko": "không", "k": "không", "kh": "không", "khong": "không", "hok": "không", "hem": "không",
    "dc": "được", "đc": "được", "duoc": "được",
    "cty": "công ty", "cti": "công ty", "nv": "nhân viên", "ns": "nhân sự",
    "mn": "mọi người", "mng": "mọi người", "vs": "với", "j": "gì", "r": "rồi",
    "sp": "sản phẩm", "tg": "thời gian", "bhxh": "bảo hiểm xã hội", "trc": "trước",
    "bt": "bình thường", "oke": "ok", "okie": "ok", "oki": "ok", "okay": "ok",
}

VI_STOP_LIGHT = {
    "và", "của", "là", "các", "những", "thì", "mà", "để", "một", "cũng", "đã", "đang", "sẽ",
    "khi", "như", "về", "từ", "ra", "vào", "với", "cho", "ở", "này", "đó", "đây", "em", "anh",
    "chị", "tôi", "mình", "bạn", "ạ", "nhé", "nha", "thế", "vậy", "thì", "lại", "nữa", "do",
}
VI_STOP_BT1 = VI_STOP_LIGHT | {
    "có", "được", "trong", "rất", "nhiều", "nên", "hay", "hoặc", "còn", "bị", "nếu", "vì",
    "theo", "tại", "trên", "đến", "cả", "chỉ", "đều", "hơn", "khá", "quá", "luôn", "vẫn",
    "có_thể", "cần", "phải", "làm", "sau", "trước", "không", "chúng_tôi", "chúng_ta", "hãy",
}
DOMAIN_STOP = {
    "company", "companies", "công_ty", "vietnam", "việt_nam", "https", "www", "com", "vn",
}
ADDRESS_STOP = {
    "ho", "chi", "minh", "hcm", "hcmc", "tp", "ha", "noi", "hanoi", "hà_nội", "hồ_chí_minh",
    "da", "nang", "đà_nẵng", "district", "ward", "street", "floor", "building", "tower", "road",
    "quận", "phường", "đường", "tầng", "toà", "tòa", "tòa_nhà", "toà_nhà", "số", "lầu", "thành_phố",
}
NEGATIONS = {"không", "chưa", "chẳng", "chả", "not", "no", "nor", "never", "cannot",
             "nothing", "none", "without", "but", "nhưng"}

EN_STOP  = frozenset(ENGLISH_STOP_WORDS)
STOP_BT1 = frozenset(EN_STOP | VI_STOP_BT1 | DOMAIN_STOP | ADDRESS_STOP)
STOP_BT2 = frozenset((EN_STOP | VI_STOP_LIGHT) - NEGATIONS)

# ======================================================================
# 2. clean_text (Mục 3.2 notebook)
# ======================================================================
URL_RE     = re.compile(r"https?://\S+|www\.\S+|\S+@\S+")
VI_RE      = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]")
TECH_RE    = re.compile("|".join(re.escape(k) for k in sorted(TECH_MAP, key=len, reverse=True)))
NONWORD_RE = re.compile(r"[^\w\s]|_")
NUM_RE     = re.compile(r"\b\d+\b")


def is_vietnamese(text):
    return bool(VI_RE.search(str(text).lower()))


def clean_text(text, stop_words=frozenset(), tokenize_vi=True, min_len=2):
    """Chuẩn hoá 1 văn bản -> chuỗi token cách nhau bởi dấu cách."""
    if not isinstance(text, str):
        return ""
    s = unicodedata.normalize("NFC", text).lower()
    s = URL_RE.sub(" ", s)
    s = TECH_RE.sub(lambda m: TECH_MAP[m.group(0)], s)
    s = NONWORD_RE.sub(" ", s)
    s = NUM_RE.sub(" ", s)
    if is_vietnamese(s):
        s = " ".join(TEENCODE.get(w, w) for w in s.split())
        if tokenize_vi and HAS_PYVI:
            s = ViTokenizer.tokenize(s)
    tokens = [t for t in s.split() if len(t) >= min_len and t not in stop_words]
    return " ".join(tokens)


# ======================================================================
# 3. BT2 — feature & cách lấy điểm (Mục 6, 8 notebook)
# ======================================================================
TEXT_COL = "text_clean"
TARGET = "recommend"
NUM_FEATURES = ["rating", "salary", "training", "management", "culture", "office"]
NUM_LABEL = {"rating": "Đánh giá chung", "salary": "Salary & benefits",
             "training": "Training & learning", "management": "Management cares about me",
             "culture": "Culture & fun", "office": "Office & workspace"}
SCENARIOS = {"text_num": NUM_FEATURES, "text_only": []}


def get_score(model, X):
    """Xác suất lớp 1 nếu có, không thì decision_function (LinearSVC)."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


# ======================================================================
# 4. BT1 — các phương pháp gợi ý (Mục 7 notebook)
# ======================================================================
METHODS = ["Gensim TF-IDF", "TF-IDF + cosine (sklearn)", "Gensim LSI (+)"]
DEFAULT_METHOD = "Gensim TF-IDF"      # P@5 cùng lĩnh vực cao nhất (Mục 7.9)


def top_k_indices(sim_row, i=None, k=TOP_K):
    s = np.asarray(sim_row, dtype=float).copy()
    if i is not None:
        s[i] = -np.inf                # không gợi ý chính nó
    return np.argsort(-s)[:k]


def query_scores(bt1, text, method):
    """Điểm tương đồng giữa 1 đoạn mô tả tự do và 478 công ty."""
    q = clean_text(text, STOP_BT1)
    g = bt1["gensim"]
    if method == "TF-IDF + cosine (sklearn)":
        from sklearn.metrics.pairwise import cosine_similarity
        sk = bt1["sklearn"]
        return cosine_similarity(sk["vectorizer"].transform([q]), sk["X"]).ravel(), q
    bow = g["dictionary"].doc2bow(q.split())
    if method == "Gensim TF-IDF":
        return np.asarray(g["index"][g["tfidf"][bow]]), q
    return np.asarray(g["index_lsi"][g["lsi"][g["tfidf"][bow]]]), q
