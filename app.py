import os
from collections import Counter
from html import escape

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import itviec_core as core
import csv_io
import ui

HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda *a: os.path.join(HERE, *a)

st.set_page_config(page_title="ITViec — Đánh giá & gợi ý công ty", page_icon="💼", layout="wide")
ui.inject_css()

TEAM = [
    {"ten": "Phạm Võ Khánh Thư", "email": "phamvokhanhthu@gmail.com",
     "viec": "EDA, gợi ý công ty (BT1), phân loại scikit-learn (BT2)"},
    {"ten": "Chế Quang Dương", "email": "duongdino8x@gmail.com",
     "viec": "Pipeline dữ liệu & tiền xử lý văn bản, phân loại PySpark, bảng so sánh, deployment"},
]


# ======================================================================
# Nạp model + dữ liệu (cache 1 lần cho mọi người dùng)
# ======================================================================
@st.cache_resource(show_spinner="Đang nạp model…")
def load_models():
    return joblib.load(P("models", "itviec_bt1.joblib")), joblib.load(P("models", "itviec_bt2.joblib"))


@st.cache_data
def load_data():
    comp = pd.read_parquet(P("data", "companies.parquet"))
    rev = pd.read_parquet(P("data", "reviews.parquet"))
    stats = rev.groupby("id").agg(n=("rating", "size"), avg=("rating", "mean"), rec=("recommend", "mean"))
    comp = comp.join(stats, on="id")
    comp["n"] = comp["n"].fillna(0).astype(int)
    return comp, rev


BT1, BT2 = load_models()
COMP, REV = load_data()
NAMES = COMP["company_name"].tolist()
COEF = BT2["coef_text_only"]
SIM = BT1["sims"][core.DEFAULT_METHOD]


@st.cache_resource(show_spinner=False)
def load_lexicon():
    """Từ điển âm tiết tiếng Việt (từ review + tên công ty) để khôi phục chữ bị mất dấu trong file upload."""
    return csv_io.build_lexicon(pd.concat([REV.title, REV.liked, REV.suggestion, COMP.company_name]).fillna(""))

MENU = ["🏠 Trang chủ", "🏢 Hồ sơ công ty", "🔎 Tìm công ty", "✍️ Viết review",
        "📊 Dữ liệu & mô hình", "👥 Nhóm thực hiện"]
ss = st.session_state
ss.setdefault("nav", MENU[0])
ss.setdefault("company", "FPT Software" if "FPT Software" in NAMES else NAMES[0])


def go_company(name):
    ss.company, ss.nav = name, MENU[1]
    ss._toast = ("🏢", f"Đã mở hồ sơ: {name}")


def go_write(name):
    ss.write_company, ss.nav = name, MENU[3]
    ss._toast = ("✍️", f"Viết review cho: {name}")


# Từ đơn mang dấu hiệu cảm xúc trong model nhưng đứng riêng thì vô nghĩa khi hiển thị
WEAK_WORDS = set(core.NEGATIONS) | {"chỉ", "hơi", "thêm", "khá", "rất", "lắm", "đôi_khi", "hay", "bị",
                                    "the", "is", "are", "very", "so", "too", "just", "only"}


def keywords(texts, min_df=3, k=6):
    """Từ / cụm từ hay gặp trong review của 1 công ty, xếp theo hệ số LogisticRegression (text_only)."""
    cnt = Counter()
    for t in texts:
        toks = t.split()
        cnt.update(set(toks) | {f"{a} {b}" for a, b in zip(toks, toks[1:])})
    words = pd.Series({w: c for w, c in cnt.items()
                       if c >= min_df and w in COEF.index and w not in WEAK_WORDS}, dtype=float)
    if words.empty:
        return [], []
    score = COEF[words.index] * np.log1p(words)
    pos = [w for w in score.nlargest(k).index if score[w] > 0]
    neg = [w for w in score.nsmallest(k).index if score[w] < 0]
    return pos, neg


def chips(words, cls):
    return "".join(f'<span class="it-chip {cls}">{escape(w.replace("_", " "))}</span>' for w in words)


def mini_company(r, sim=None, key="", desc=None):
    """Thẻ công ty thu gọn — cả thẻ bấm được: 1 nút trong suốt phủ lên thẻ (CSS st-key-open_*)."""
    score = (f'{ui.stars(r["avg"], 14)} <span class="it-meta">{ui.vn(r["avg"])} · {r["n"]} review</span>'
             if r["n"] else '<span class="it-meta">chưa có review</span>')
    extra = f' · tương đồng {sim:.2f}' if sim is not None else ""
    body = f'<div class="it-meta" style="margin-top:8px">{escape(desc)}</div>' if desc else ""
    with st.container(key=f"cc_{key}_{r['id']}"):
        ui.html(f'<div class="it-ccard"><div style="display:flex;gap:10px;align-items:flex-start">'
                f'<div class="it-logo" style="width:36px;height:36px;font-size:13px;border-radius:8px">'
                f'{ui.logo_text(r["company_name"])}</div><div><div class="it-name">{escape(r["company_name"])}</div>'
                f'<div>{score}</div><div class="it-meta">{escape(str(r["industry"] or ""))}{extra}</div></div></div>'
                f'{body}</div>')
        st.button(f"Xem hồ sơ {r['company_name']}", key=f"open_{key}_{r['id']}",
                  on_click=go_company, args=(r["company_name"],))


def explain(model, X, k=8):
    """Đóng góp của từng từ / điểm số = giá trị feature × hệ số (model tuyến tính)."""
    pre, clf = model.named_steps["pre"], model.named_steps["clf"]
    x = pre.transform(X)
    x = x.toarray().ravel() if hasattr(x, "toarray") else np.asarray(x).ravel()
    names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
    contrib = pd.Series(x * clf.coef_.ravel(), index=names)
    contrib = contrib[contrib != 0].rename(index=lambda n: core.NUM_LABEL.get(n, n.replace("_", " ")))
    return contrib.nsmallest(k), contrib.nlargest(k)


# ======================================================================
# Sidebar
# ======================================================================
st.sidebar.image(P("images", "channels4_banner.jpg"), width="stretch")
st.sidebar.radio("Menu", MENU, key="nav", label_visibility="collapsed")
st.sidebar.divider()
ui.team_sidebar(TEAM)
st.sidebar.caption("Đồ án tốt nghiệp Data Science — TTTH ĐH KHTN")
st.sidebar.caption("Sản phẩm học tập, không phải website chính thức của ITViec. Kết quả mô hình chỉ mang tính tham khảo.")
choice = ss.nav

# ----------------------------------------------------------------------
if choice == MENU[0]:
    ui.card('<div class="it-hero"><div class="it-logo" style="background:#e5484d">it</div><div>'
            '<div class="it-title">Đọc review thật. Tìm công ty IT phù hợp.</div>'
            '<div class="it-sub">478 công ty · 8.415 review của nhân viên trên ITViec (2016–2025)</div>'
            f'<div class="it-score"><b>{ui.vn(REV["rating"].mean())}</b>{ui.stars(REV["rating"].mean(), 24)}'
            f'<span class="it-meta">điểm trung bình toàn bộ review · {REV["recommend"].mean():.0%} recommend</span>'
            '</div></div></div>')
    mn, mo = BT2["metrics"]["text_num"], BT2["metrics"]["text_only"]
    ui.kpis([("Gợi ý cùng lĩnh vực (P@5) · ngẫu nhiên 13%", "23,5%"),
             ("F1-macro · text + điểm", ui.vn(mn["F1_macro"], 3)),
             ("F1-macro · chỉ text", ui.vn(mo["F1_macro"], 3)),
             ("Bắt được review “No”", f"{mn['Recall_No']:.0%}")])
    c1, c2 = st.columns(2)
    with c1:
        ui.card('<div class="it-h">🔎 Bài toán 1 — Content-Based Company Similarity</div>'
                '<div class="it-txt">Từ mô tả công ty (overview + key skills + why you\'ll love), gợi ý các công ty '
                '<b>tương tự</b> — tìm đối thủ, đối tác hoặc nơi làm việc giống nơi bạn thích. So sánh TF-IDF + '
                'cosine (scikit-learn), Gensim TF-IDF, Gensim LSI → chọn <b>Gensim TF-IDF</b>.</div>')
    with c2:
        ui.card('<div class="it-h">👍👎 Bài toán 2 — Recommend or Not</div>'
                '<div class="it-txt">Từ nội dung review (+ 6 điểm số), dự đoán người viết có <b>recommend</b> công ty '
                'không. Dữ liệu lệch 88/12 → đo bằng F1-macro & Recall lớp No. Triển khai <b>LinearSVC</b> '
                '(text + điểm) và <b>LogisticRegression</b> (chỉ text).</div>')

    st.markdown('<div class="it-h">Công ty được đánh giá cao (≥ 30 review)</div>', unsafe_allow_html=True)
    top = COMP[COMP.n >= 30].sort_values("avg", ascending=False).head(6)
    cols = st.columns(3)
    for i, (_, r) in enumerate(top.iterrows()):
        with cols[i % 3]:
            mini_company(r, key="home")

# ----------------------------------------------------------------------
elif choice == MENU[1]:
    st.selectbox("Chọn hoặc gõ tên công ty", NAMES, key="company")
    r = COMP.iloc[NAMES.index(ss.company)]
    rv = REV[REV.id == r["id"]]

    top, right = st.columns([1.7, 1])
    with top:
        ui.company_hero(r, len(rv), rv["rating"].mean() if len(rv) else None)
    with right:
        if len(rv):
            ui.card(f'<div class="it-kpi">Điểm trung bình</div><div class="it-big">{ui.vn(rv["rating"].mean())}</div>'
                    + ui.dist_bars(rv["rating"])
                    + f'<div class="it-meta">{rv["recommend"].mean():.0%} nhân viên recommend công ty này</div>')
        else:
            ui.card('<div class="it-kpi">Chưa có review chi tiết</div><div class="it-txt">'
                    'Công ty chưa có review trong dữ liệu — vẫn xem được giới thiệu và công ty tương tự.</div>')
        st.button("✍️ Viết review cho công ty này", type="primary", width="stretch",
                  on_click=go_write, args=(ss.company,))

    if len(rv):
        subs = rv[[c for c, _ in ui.SUB]].mean()
        allm = REV[[c for c, _ in ui.SUB]].mean()
        diff = (subs - allm).sort_values()
        lab = dict(ui.SUB)
        pos, neg = keywords(rv["text_clean"], min_df=max(2, min(5, len(rv) // 20)))
        hi, lo = diff.index[-1], diff.index[0]
        txt = (f"Dựa trên <b>{ui.vn(len(rv), 0)}</b> review: <b>{rv['recommend'].mean():.0%}</b> recommend "
               f"(trung bình ITViec {REV['recommend'].mean():.0%}). So với mặt bằng ITViec, tiêu chí nổi bật nhất là "
               f"<b>{lab[hi]}</b> ({ui.vn(subs[hi])}★, {'+' if diff[hi] >= 0 else ''}{ui.vn(diff[hi])}), kém nhất là "
               f"<b>{lab[lo]}</b> ({ui.vn(subs[lo])}★, {'+' if diff[lo] >= 0 else ''}{ui.vn(diff[lo])}).")
        if pos or neg:
            txt += ("<br>Nhân viên hay nhắc tích cực: " + chips(pos, "it-pos")
                    + "<br>Hay nhắc tiêu cực: " + chips(neg, "it-neg"))
        ui.summary("Tóm tắt review", txt)
        st.write("")

    with st.expander("Giới thiệu công ty"):
        st.write(" ".join(str(r["overview"] or "").split()))
        if isinstance(r["key_skills"], str) and r["key_skills"]:
            st.markdown("**Key skills:** " + r["key_skills"])
        if isinstance(r["why_love"], str) and r["why_love"].strip():
            st.markdown("**Why you'll love working here:** " + " ".join(r["why_love"].split()))

    main, side = st.columns([1.9, 1.1])
    with main:
        if len(rv):
            star_sel = st.pills("Lọc theo số sao", [5, 4, 3, 2, 1], selection_mode="multi",
                                format_func=lambda k: f"{k}★")
            f1, f2, f3 = st.columns(3)
            rec_sel = f1.selectbox("Recommend", ["Tất cả", "Recommend", "Không recommend"])
            sort = f2.selectbox("Sắp xếp", ["Mới nhất", "Điểm cao nhất", "Điểm thấp nhất"])
            q = f3.text_input("Tìm trong review", placeholder="vd: lương, OT, onsite…")
            lst = rv
            if star_sel:
                lst = lst[lst.rating.isin(star_sel)]
            if rec_sel != "Tất cả":
                lst = lst[lst.recommend == (1 if rec_sel == "Recommend" else 0)]
            if q.strip():
                m = (lst.title.fillna("") + " " + lst.liked.fillna("") + " " + lst.suggestion.fillna("")
                     ).str.contains(q.strip(), case=False, regex=False)
                lst = lst[m]
            key, asc = {"Mới nhất": (["cmt_date"], [False]), "Điểm cao nhất": (["rating", "cmt_date"], [False, False]),
                        "Điểm thấp nhất": (["rating", "cmt_date"], [True, False])}[sort]
            lst = lst.sort_values(key, ascending=asc)
            PER = 10
            n_page = max(1, int(np.ceil(len(lst) / PER)))
            pkey = f"pg_{ss.company}_{star_sel}_{rec_sel}_{sort}_{q}"
            page = ss.get(pkey, 1)
            st.caption(f"{ui.vn(len(lst), 0)} review · trang {page}/{n_page}")
            for i, (_, x) in enumerate(lst.iloc[(page - 1) * PER: page * PER].iterrows()):
                ui.review_card(x, i)
            if n_page > 1:
                st.number_input(f"Trang (1–{n_page})", 1, n_page, key=pkey)
        else:
            st.info("Chưa có review cho công ty này.")

    with side:
        st.markdown('<div class="it-h">Công ty tương tự</div>'
                    '<div class="it-meta">Gợi ý content-based (Gensim TF-IDF) từ mô tả công ty</div>',
                    unsafe_allow_html=True)
        i = NAMES.index(ss.company)
        for j in core.top_k_indices(SIM[i], i, 6):
            mini_company(COMP.iloc[j], float(SIM[i, j]), key="side")

# ----------------------------------------------------------------------
elif choice == MENU[2]:
    st.markdown('<div class="it-title">Tìm công ty phù hợp với bạn</div>'
                '<div class="it-sub">Gõ mô tả công việc / công ty mong muốn (tiếng Việt hoặc tiếng Anh) — '
                'hệ thống tìm công ty có mô tả gần nhất.</div>', unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns([3, 1.3, 1])
    ex = ["Công ty outsourcing Java cho khách hàng Nhật Bản",
          "fintech mobile banking app, React Native, payment",
          "game studio Unity C# mobile games"]
    q = c1.text_input("Mô tả", value=ex[0], label_visibility="collapsed")
    method = c2.selectbox("Phương pháp", core.METHODS, label_visibility="collapsed")
    k = c3.selectbox("Số kết quả", [6, 9, 12], label_visibility="collapsed")
    st.caption("Ví dụ: " + " · ".join(f"`{e}`" for e in ex))
    if q.strip():
        scores, q_clean = core.query_scores(BT1, q, method)
        if not q_clean or scores.max() <= 0:
            st.warning("Không có từ khoá nào khớp dữ liệu — thử mô tả cụ thể hơn (công nghệ, lĩnh vực, thị trường…).")
        else:
            st.caption(f"Từ khoá sau làm sạch: `{q_clean}`")
            idx = core.top_k_indices(scores, None, k)
            cols = st.columns(3)
            for n, j in enumerate(idx):
                with cols[n % 3]:
                    r = COMP.iloc[j]
                    mini_company(r, float(scores[j]), key="search",
                                 desc=" ".join(str(r["overview"] or "").split())[:180] + "…")

# ----------------------------------------------------------------------
elif choice == MENU[3]:
    st.markdown('<div class="it-title">Viết review</div><div class="it-sub">Mô hình sẽ đọc review của bạn và dự '
                'đoán bạn có <b>recommend</b> công ty hay không — trước khi bạn chọn.</div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.3, 1])
    with left, st.container(border=True):
        default = NAMES.index(ss.get("write_company", ss.company))
        company = st.selectbox("Công ty", NAMES, index=default)
        st.markdown("**Đánh giá chung**")
        fb = st.feedback("stars", key="w_star")
        use_num = st.toggle("Chấm thêm 5 tiêu chí (giống form ITViec)", True)
        ratings = {}
        if use_num:
            cols = st.columns(5)
            for n, (c, lab) in enumerate(ui.SUB):
                ratings[c] = cols[n].select_slider(lab, [1, 2, 3, 4, 5], 4, key=f"w_{c}")
        title = st.text_input("Tiêu đề", "Môi trường tốt")
        liked = st.text_area("Điều bạn thích", "Đồng nghiệp thân thiện, sếp support nhiệt tình, được học nhiều công nghệ mới")
        sugg = st.text_area("Điều cần cải thiện", "Nên tăng lương định kỳ")
        ok = st.button("Dự đoán", type="primary")

    with right:
        if ok:
            rating = (fb + 1) if fb is not None else 4
            scen = "text_num" if use_num else "text_only"
            row = {core.TEXT_COL: core.clean_text(f"{title}. {liked}. {sugg}", core.STOP_BT2)}
            if use_num:
                row.update({"rating": float(rating), **{c: float(v) for c, v in ratings.items()}})
            X = pd.DataFrame([row])
            model = BT2["models"][scen]
            yes = int(model.predict(X)[0]) == 1
            score = float(core.get_score(model, X)[0])
            prob = (f"xác suất recommend <b>{score:.0%}</b>" if hasattr(model, "predict_proba")
                    else f"điểm quyết định <b>{score:+.2f}</b> (&gt; 0 là recommend)")
            preview = pd.Series({"company_name": company, "cmt_date": pd.Timestamp.today(), "title": title,
                                 "liked": liked, "suggestion": sugg, "rating": rating,
                                 "recommend": int(yes), **ratings})
            st.markdown('<div class="it-h">Xem trước review của bạn</div>', unsafe_allow_html=True)
            ui.review_card(preview, 2)
            ui.html(f'<div class="it-reply" style="margin-top:-6px"><div class="who">🤖 Mô hình Recommend or Not</div>'
                    f'{"👍 <b>RECOMMEND</b>" if yes else "👎 <b>KHÔNG RECOMMEND</b>"} — {prob} · '
                    f'{BT2["metrics"][scen]["model"]} ({scen}), F1-macro {BT2["metrics"][scen]["F1_macro"]:.3f}</div>')
            if fb is None and use_num:
                st.caption("Chưa chọn sao đánh giá chung → tạm dùng 4★.")
            neg, pos = explain(model, X)
            e1, e2 = st.columns(2)
            with e1:
                st.markdown("🔻 Kéo về **không recommend**")
                st.bar_chart(neg.abs().rename("mức ảnh hưởng"), horizontal=True, color="#e5484d")
            with e2:
                st.markdown("🔺 Kéo về **recommend**")
                st.bar_chart(pos.rename("mức ảnh hưởng"), horizontal=True, color="#2e9e6a")
            st.caption(f"Văn bản sau làm sạch: `{row[core.TEXT_COL]}`")
        else:
            ui.card('<div class="it-h">Cách hoạt động</div><div class="it-txt">'
                    '1. Văn bản được chuẩn hoá: teencode (ko → không, cty → công ty), thuật ngữ IT (C# → csharp), '
                    'tách từ tiếng Việt bằng pyvi, giữ lại từ phủ định.<br>'
                    '2. TF-IDF (từ đơn + cụm 2 từ) + 6 điểm số → <b>LinearSVC</b>; tắt chấm điểm → '
                    '<b>LogisticRegression</b> chỉ đọc văn bản.<br>'
                    '3. Biểu đồ cho biết từ / điểm số nào kéo kết quả về mỗi phía.</div>')

        with st.expander("📄 Dự đoán hàng loạt từ file CSV / Excel"):
            st.caption("Cột `title`, `liked`, `suggestion`; có đủ 6 cột điểm "
                       f"`{', '.join(core.NUM_FEATURES)}` thì dùng model text + điểm.")
            d1, d2 = st.columns(2)
            with open(P("data", "sample_reviews.xlsx"), "rb") as f:
                d1.download_button("⬇️ File mẫu Excel", f, "sample_reviews.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            with open(P("data", "sample_reviews.csv"), "rb") as f:
                d2.download_button("⬇️ File mẫu CSV", f, "sample_reviews.csv", "text/csv")
            up = st.file_uploader("Chọn file CSV hoặc Excel (.xlsx)", type=["csv", "xlsx"])
            if up is not None:
                try:
                    df = csv_io.load_upload(st, up, load_lexicon())
                    for c in ["title", "liked", "suggestion"]:
                        df[c] = df.get(c, pd.Series("", index=df.index)).fillna("").astype(str)
                    df[core.TEXT_COL] = (df["title"] + ". " + df["liked"] + ". " + df["suggestion"]).map(
                        lambda s: core.clean_text(s, core.STOP_BT2))
                    has_num = set(core.NUM_FEATURES).issubset(df.columns) and df[core.NUM_FEATURES].notna().all().all()
                    scen = "text_num" if has_num else "text_only"
                    model = BT2["models"][scen]
                    df["du_doan"] = np.where(model.predict(df) == 1, "Recommend", "Not recommend")
                    df["diem"] = np.round(core.get_score(model, df), 3)
                    st.write(f"Model **{BT2['metrics'][scen]['model']}** ({scen}) · {len(df)} review · "
                             f"{(df['du_doan'] == 'Not recommend').sum()} Not recommend")
                    show = ["title", "liked", "suggestion", *(core.NUM_FEATURES if has_num else []), "du_doan", "diem"]
                    st.dataframe(df[show], width="stretch")
                    st.download_button("⬇️ Tải kết quả", df[show].to_csv(index=False).encode("utf-8-sig"),
                                       "ket_qua_recommend.csv", "text/csv")
                except Exception as e:
                    st.error(f"Không xử lý được file: {e}")

# ----------------------------------------------------------------------
elif choice == MENU[4]:
    st.markdown('<div class="it-title">Dữ liệu & kết quả mô hình</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["Khám phá dữ liệu", "BT1 — Gợi ý công ty", "BT2 — Recommend or Not"])
    with t1:
        st.image(P("images", "eda_companies.png"), caption="Chân dung 478 công ty")
        st.image(P("images", "eda_company_ratings.png"), caption="Điểm đánh giá cấp công ty")
        st.image(P("images", "eda_reviews_label.png"),
                 caption="Nhãn Recommend lệch 88/12 — Rating gần như quyết định nhãn")
        st.image(P("images", "eda_reviews_company_time.png"), caption="FPT Software chiếm ~24% review")
        st.image(P("images", "wordcloud_companies.png"), caption="Wordcloud mô tả công ty")
        st.image(P("images", "wordcloud_reviews_yes_no.png"), caption="Từ đặc trưng review Yes vs No")
    with t2:
        st.markdown("Không có nhãn \"công ty tương tự\" → đánh giá bằng **P@5 cùng lĩnh vực**: "
                    "trong 5 công ty được gợi ý, bao nhiêu % cùng ngành với công ty gốc "
                    "(ngành **không** nằm trong văn bản đưa vào model). Chọn **Gensim TF-IDF**.")
        st.dataframe(pd.read_csv(P("data", "bt1_so_sanh_phuong_phap.csv")), width="stretch", hide_index=True)
        st.image(P("images", "bt1_danh_gia.png"))
    with t3:
        st.markdown("""
Hai kịch bản: **text_num** (nội dung review + 6 điểm số, khớp form thật của ITViec) và **text_only** (chỉ nội dung).
Thước đo chính là **F1-macro** và **Recall lớp No** vì dữ liệu lệch 88/12.
Model triển khai: **LinearSVC** (text_num) và **LogisticRegression** (text_only).
PySpark (chạy trong notebook, không nằm trong bảng bên dưới): LogisticRegression tốt nhất — F1-macro 0,795 (text_num) và 0,720 (text_only). Lưu ý: F1-macro của Spark và sklearn có thể tính hơi khác nhau nên chỉ so sánh tương đối.
""")
        st.dataframe(pd.read_csv(P("data", "bang_so_sanh_model_bt2.csv")).round(3), width="stretch", hide_index=True)
        st.markdown("**Mốc so sánh & kiểm tra độ tin cậy** (tính lại trên cùng tập test 1.683 review; 88% review là Recommend)")
        st.dataframe(pd.DataFrame([
            ["Luôn đoán “Recommend”", "0,878", "0,467", "0%", "Accuracy 88% nhưng không bắt được review No nào"],
            ["Chỉ 6 điểm số (không text)", "0,844", "0,743", "89%", "Precision No chỉ 43% — nhiều báo động giả"],
            ["Chỉ text (LogisticRegression)", "0,867", "0,730", "63%", "Mô hình triển khai cho text_only"],
            ["Text + 6 điểm số (LinearSVC)", "0,915", "0,822", "78%", "Mô hình triển khai chính"],
            ["Text + điểm số — chia theo công ty (5 lần)", "—", "0,842", "79%", "±0,040; không thấp hơn chia ngẫu nhiên 5 lần (0,830) → không thấy dấu hiệu học thuộc công ty"],
        ], columns=["Mô hình / kiểm tra", "Accuracy", "F1-macro", "Recall No", "Ghi chú"]),
            width="stretch", hide_index=True)
        st.caption("Điểm số riêng đã đạt F1-macro 0,743; text giúp thêm ~0,08. Dùng F1-macro / Recall No, không dùng Accuracy.")
        st.image(P("images", "bang_so_sanh_model_bt2.png"))
        st.image(P("images", "bt2_confusion_roc.png"), caption="Confusion matrix + ROC")
        st.markdown("**Xử lý mất cân bằng** (LogisticRegression, text_only)")
        st.dataframe(pd.read_csv(P("data", "bt2_imbalance.csv")), width="stretch", hide_index=True)
        st.image(P("images", "bt2_top_words.png"), caption="Từ đẩy review về Not recommend / Recommend")

# ----------------------------------------------------------------------
else:
    st.markdown('<div class="it-title">Nhóm thực hiện</div><div class="it-sub">Đồ án tốt nghiệp Data Science — '
                'Trung tâm Tin học, ĐH KHTN TP.HCM</div>', unsafe_allow_html=True)
    st.write("")
    for m in TEAM:
        ini = "".join(w[0] for w in m["ten"].split()[-2:]).upper()
        ui.card(f'<div class="it-top"><div class="it-ava" style="width:52px;height:52px;font-size:1rem;'
                f'background:#e8eefc;color:#1f3f8f">{escape(ini)}</div><div><div class="it-name" style="font-size:1.15rem">'
                f'{escape(m["ten"])}</div><div class="it-meta">📧 {escape(m["email"])}</div></div></div>'
                f'<hr class="it-hr"><div class="it-txt"><b>Phụ trách:</b> {escape(m["viec"])}</div>')

# ======================================================================
# Phản hồi điều hướng: thông báo + cuộn lên đầu trang khi đổi trang / đổi công ty
# ======================================================================
if "_toast" in ss:
    icon, msg = ss.pop("_toast")
    st.toast(msg, icon=icon)
view = (ss.nav, ss.company if ss.nav == MENU[1] else None)
if ss.get("_last_view") != view:
    ss._last_view = view
    ss._scroll_n = ss.get("_scroll_n", 0) + 1     # nội dung khác nhau mỗi lần -> script chạy lại
    components.html(
        f"<script>/* {ss._scroll_n} */ const d = window.parent.document;"
        "for (const s of ['[data-testid=\"stMain\"]', '[data-testid=\"stAppViewContainer\"]', 'section.main']) {"
        "  const el = d.querySelector(s); if (el) el.scrollTo({top: 0, behavior: 'smooth'}); }"
        "window.parent.scrollTo({top: 0, behavior: 'smooth'});</script>", height=0)
