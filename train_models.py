"""Huấn luyện lại model cho GUI ITViec từ các file parquet sinh ở Mục 4 & 6
notebook project_2, lưu vào models/. Chạy:  python train_models.py

Đầu vào (KHÔNG đưa lên GitHub vì chứa review gốc):
    ../../project_2/df_companies_clean.parquet
    ../../project_2/df_model_bt2.parquet
Đầu ra (đưa lên GitHub):
    models/itviec_bt1.joblib · models/itviec_bt2.joblib · data/companies.parquet
"""
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import itviec_core as core

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "project_2"))

SHOW_COLS = ["id", "company_name", "company_type", "industry", "company_size", "country",
             "working_days", "ot_policy", "overview", "key_skills", "why_love", "href", "cities",
             "n_reviews", "overall_rating", "recommend_pct",
             "c_salary", "c_training", "c_management", "c_culture", "c_office"]


def p_at_5(sim, labels):
    p = []
    for i in range(len(labels)):
        if pd.isna(labels[i]):
            continue
        idx = core.top_k_indices(sim[i], i)
        p.append(np.mean([labels[j] == labels[i] for j in idx]))
    return float(np.mean(p))


def train_bt1():
    from gensim import corpora, models, similarities
    comp = pd.read_parquet(os.path.join(SRC, "df_companies_clean.parquet"))
    docs = comp["doc_clean"].fillna("")

    tfidf_bt1 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True)
    X_bt1 = tfidf_bt1.fit_transform(docs)

    docs_tok = [d.split() for d in docs]
    dictionary = corpora.Dictionary(docs_tok)
    dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=None)
    corpus = [dictionary.doc2bow(d) for d in docs_tok]
    tfidf_g = models.TfidfModel(corpus)
    index_g = similarities.SparseMatrixSimilarity(tfidf_g[corpus], num_features=len(dictionary))
    lsi = models.LsiModel(tfidf_g[corpus], id2word=dictionary, num_topics=100,
                          random_seed=core.RANDOM_STATE)
    index_lsi = similarities.MatrixSimilarity(lsi[tfidf_g[corpus]], num_features=100)

    sims = {"TF-IDF + cosine (sklearn)": cosine_similarity(X_bt1).astype(np.float32),
            "Gensim TF-IDF": np.asarray(index_g[tfidf_g[corpus]], dtype=np.float32),
            "Gensim LSI (+)": np.asarray(index_lsi[lsi[tfidf_g[corpus]]], dtype=np.float32)}

    ind = comp["industry"].to_numpy(dtype=object)
    for m, s in sims.items():
        print(f"  BT1 {m:28s} P@5 cùng lĩnh vực = {p_at_5(s, ind):.3f}")

    bt1 = {"default_method": core.DEFAULT_METHOD, "sims": sims,
           "sklearn": {"vectorizer": tfidf_bt1, "X": X_bt1},
           "gensim": {"dictionary": dictionary, "tfidf": tfidf_g, "index": index_g,
                      "lsi": lsi, "index_lsi": index_lsi}}
    joblib.dump(bt1, os.path.join(HERE, "models", "itviec_bt1.joblib"), compress=3)
    comp[SHOW_COLS].to_parquet(os.path.join(HERE, "data", "companies.parquet"), index=False)


def make_pre(scenario):
    parts = [("txt", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9,
                                     sublinear_tf=True, max_features=30000), core.TEXT_COL)]
    if core.SCENARIOS[scenario]:
        parts.append(("num", MinMaxScaler(), core.SCENARIOS[scenario]))
    return ColumnTransformer(parts)


# Model triển khai theo notebook v5 mục 10.4: chọn theo PR-AUC lớp No out-of-fold trên train
# (text_num: LogisticRegression 0,802 > LinearSVC 0,787 > LightGBM 0,783; text_only: hoà -> ưu tiên LogisticRegression).
# Ngưỡng chọn trên out-of-fold của train (tối đa F1-macro); giá trị lấy đúng từ lần chạy notebook v5.
DEPLOY = {"text_num": ("LogisticRegression", LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced")),
          "text_only": ("LogisticRegression", LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced"))}
THRESHOLDS = {"text_num": 0.199, "text_only": 0.476}


def train_bt2():
    df = pd.read_parquet(os.path.join(SRC, "df_model_bt2.parquet"))
    tr, te = df[df["_split"] == "train"], df[df["_split"] == "test"]
    out = {"models": {}, "metrics": {}, "thresholds": dict(THRESHOLDS)}
    for scen, (name, est) in DEPLOY.items():
        pipe = Pipeline([("pre", make_pre(scen)), ("clf", est)]).fit(tr, tr[core.TARGET])
        y, score = te[core.TARGET].values, core.get_score(pipe, te)
        pred = (score > THRESHOLDS[scen]).astype(int)
        m = {"Accuracy": accuracy_score(y, pred),
             "Precision_No": precision_score(y, pred, pos_label=0),
             "Recall_No": recall_score(y, pred, pos_label=0),
             "F1_macro": f1_score(y, pred, average="macro"),
             "ROC_AUC": roc_auc_score(y, score),
             "PR_AUC_No": average_precision_score(1 - y, -score)}
        print(f"  BT2 {scen:9s} {name:20s} ngưỡng {THRESHOLDS[scen]} " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))
        out["models"][scen] = pipe
        out["metrics"][scen] = {"model": name, **m}

    # Từ đẩy về No / Yes (hệ số LogisticRegression text_only) -> giải thích trên web
    lr = out["models"]["text_only"]
    vocab = lr.named_steps["pre"].named_transformers_["txt"].get_feature_names_out()
    coef = pd.Series(lr.named_steps["clf"].coef_.ravel()[:len(vocab)], index=vocab)
    out["top_words"] = {"no": coef.nsmallest(20), "yes": coef.nlargest(20)}
    out["coef_text_only"] = coef[coef.abs() > 0.5].astype("float32")   # cho tóm tắt từ khoá theo công ty
    out["n_train"], out["n_test"] = len(tr), len(te)
    joblib.dump(out, os.path.join(HERE, "models", "itviec_bt2.joblib"), compress=3)

    # Review hiển thị trên web + dự đoán của model text_num (df_model_bt2 cùng thứ tự với df_reviews_clean)
    rev = pd.read_parquet(os.path.join(SRC, "df_reviews_clean.parquet"))
    assert len(rev) == len(df) and (rev["recommend"].values == df["recommend"].values).all()
    rev["_split"] = df["_split"].values
    rev["pred"] = (core.get_score(out["models"]["text_num"], df) > THRESHOLDS["text_num"]).astype(int)
    keep = ["id", "company_name", "cmt_date", "title", "liked", "suggestion", *core.NUM_FEATURES,
            "recommend", "text_clean", "_split", "pred"]
    rev[keep].to_parquet(os.path.join(HERE, "data", "reviews.parquet"), index=False)


def main():
    t0 = time.time()
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    train_bt1()
    train_bt2()
    print(f"Xong trong {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
