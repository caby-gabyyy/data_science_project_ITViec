"""Thành phần giao diện (HTML/CSS) cho GUI ITViec — bố cục trang review:
hồ sơ công ty, thang sao, thanh phân bố, thẻ review, cột công ty tương tự."""
from html import escape

import numpy as np
import pandas as pd
import streamlit as st

STAR_COLOR = {5: "#2e9e6a", 4: "#7cc242", 3: "#f5b800", 2: "#f76b15", 1: "#e5484d"}
STAR_LABEL = {5: "Xuất sắc", 4: "Tốt", 3: "Trung bình", 2: "Kém", 1: "Rất tệ"}

CSS = """
<style>
:root{--card:#ffffff;--line:#e6e3da;--ink:#1c1c1c;--muted:#6b6b6b;--soft:#f3f1ea;--link:#1f6feb}
.block-container,[data-testid="stMainBlockContainer"]{padding-top:4.2rem;max-width:1180px}
.it-card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin-bottom:14px}
.it-hero{display:flex;gap:18px;align-items:center}
.it-logo{width:84px;height:84px;border-radius:14px;background:#1c1c1c;color:#fff;display:flex;align-items:center;
  justify-content:center;font-size:28px;font-weight:800;flex:none;letter-spacing:-.02em}
.it-title{font-size:1.7rem;font-weight:700;margin:0;line-height:1.2;color:var(--ink)}
.it-sub{color:var(--muted);font-size:.92rem;margin-top:4px}
.it-score{display:flex;align-items:center;gap:10px;margin-top:8px;flex-wrap:wrap}
.it-score b{font-size:1.25rem}
.it-tag{display:inline-block;background:var(--soft);border:1px solid var(--line);border-radius:999px;
  padding:2px 10px;font-size:.8rem;margin:6px 6px 0 0;color:var(--ink)}
.it-stars{display:inline-flex;gap:2px;vertical-align:middle}
.it-star{display:inline-flex;align-items:center;justify-content:center;color:#fff;border-radius:3px;font-size:.8em;line-height:1}
.it-bar{display:flex;align-items:center;gap:10px;margin:7px 0;font-size:.9rem;color:var(--ink)}
.it-bar .lab{width:110px;flex:none}
.it-bar .trk{flex:1;height:10px;background:var(--soft);border-radius:999px;overflow:hidden}
.it-bar .fil{display:block;height:100%;border-radius:999px}
.it-bar .pct{width:42px;text-align:right;color:var(--muted);flex:none}
.it-sum{background:var(--soft);border-radius:12px;padding:14px 16px;font-size:.95rem;line-height:1.6}
.it-sum .hd{font-weight:700;margin-bottom:4px}
.it-chip{display:inline-block;border-radius:999px;padding:1px 9px;font-size:.8rem;margin:2px 4px 2px 0}
.it-pos{background:#e7f6ee;color:#1d7a50}.it-neg{background:#fdeceb;color:#b42318}
.it-top{display:flex;align-items:center;gap:12px}
.it-ava{width:40px;height:40px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  font-weight:700;font-size:.85rem;flex:none}
.it-name{font-weight:600;color:var(--ink)}
.it-meta{color:var(--muted);font-size:.82rem}
.it-badge{display:inline-block;font-size:.75rem;border-radius:4px;padding:1px 7px;margin-left:6px;vertical-align:middle}
.it-yes{background:#e7f6ee;color:#1d7a50}.it-no{background:#fdeceb;color:#b42318}.it-info{background:#eef3ff;color:#1f3f8f}
.it-card hr.it-hr{border:none;border-top:1px solid var(--line);margin:12px 0 !important}
.it-h{font-weight:700;font-size:1.05rem;margin:10px 0 4px;color:var(--ink)}
.it-txt{color:#333;font-size:.93rem;line-height:1.55;margin-top:6px}
.it-txt .k{font-weight:600;color:var(--ink)}
.it-txt summary{color:var(--link);cursor:pointer;font-size:.85rem;margin-top:4px}
.it-sub5{display:flex;gap:14px;flex-wrap:wrap;font-size:.8rem;color:var(--muted);margin-top:10px}
.it-reply{border-left:3px solid var(--line);background:#fafaf7;border-radius:0 10px 10px 0;padding:10px 14px;
  margin-top:12px;font-size:.88rem;line-height:1.5}
.it-reply .who{font-weight:600;margin-bottom:2px}
.it-side{display:flex;justify-content:space-between;align-items:center;gap:8px}
.it-big{font-size:2.6rem;font-weight:700;line-height:1;color:var(--ink)}
.it-kpi{font-size:.85rem;color:var(--muted)}
.it-kpi b{display:block;font-size:1.35rem;color:var(--ink);margin-top:2px}
.it-team{font-size:.85rem;line-height:1.45}
.it-ccard{background:#fff;border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin-bottom:10px}
/* Phản hồi khi đang tải: thanh xanh chạy ở đầu trang + nội dung cũ mờ đi */
.stApp[data-test-script-state="running"]::before{content:"";position:fixed;top:0;left:0;height:4px;width:40%;
  z-index:1000000;border-radius:0 4px 4px 0;background:linear-gradient(90deg,#4b8df8,#1f6feb);
  animation:it-load .9s ease-in-out infinite}
@keyframes it-load{0%{left:-40%}100%{left:100%}}
.stApp[data-test-script-state="running"] [data-stale="true"]{opacity:.45;transition:opacity .15s}
/* Cả thẻ công ty bấm được: nút trong suốt phủ kín thẻ; nổi lên khi rê chuột, lún xuống khi nhấn */
[class*="st-key-cc_"]{position:relative}
[class*="st-key-cc_"] .it-ccard{transition:box-shadow .15s ease,transform .15s ease}
[class*="st-key-cc_"]:hover .it-ccard{box-shadow:0 6px 18px rgba(0,0,0,.10);transform:translateY(-2px)}
[class*="st-key-cc_"]:hover .it-name{color:var(--link)}
[class*="st-key-cc_"]:active .it-ccard{transform:scale(.97);border-color:var(--link);box-shadow:0 2px 6px rgba(31,111,235,.3)}
[class*="st-key-open_"]{position:absolute !important;inset:0;z-index:5;width:100% !important;margin:0 !important}
[class*="st-key-open_"] div[data-testid="stButton"],[class*="st-key-open_"] button{width:100% !important;
  height:100% !important;min-height:100%;opacity:0;cursor:pointer;padding:0;border:0}
</style>
"""

AVA_COLORS = ["#e8eefc", "#fdeceb", "#e7f6ee", "#fff6dc", "#f1e9fd", "#e6f4f8"]
AVA_TEXT = ["#1f3f8f", "#b42318", "#1d7a50", "#8a6200", "#5b2a9e", "#0f6170"]


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def vn(x, d=1):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def html(s):
    """Gộp 1 dòng — tránh Markdown hiểu '- …' trong text là danh sách."""
    st.markdown(" ".join(s.split("\n")), unsafe_allow_html=True)


def card(s):
    html(f'<div class="it-card">{s}</div>')


def stars(value, size=20):
    if value is None or pd.isna(value):
        return f'<span class="it-meta">chưa có đánh giá</span>'
    lv = int(min(5, max(1, round(value))))
    color = STAR_COLOR[lv]
    out = []
    for k in range(5):
        frac = min(1.0, max(0.0, value - k))
        bg = (color if frac >= 1 else "#dcdbd3" if frac <= 0
              else f"linear-gradient(90deg,{color} {frac * 100:.0f}%,#dcdbd3 {frac * 100:.0f}%)")
        out.append(f'<span class="it-star" style="width:{size}px;height:{size}px;background:{bg}">★</span>')
    return f'<span class="it-stars">{"".join(out)}</span>'


def dist_bars(ratings):
    r = pd.Series(ratings).dropna().round().astype(int)
    n = max(len(r), 1)
    rows = []
    for k in [5, 4, 3, 2, 1]:
        p = (r == k).sum() / n * 100
        rows.append(f'<div class="it-bar"><span class="lab">{k}★ {STAR_LABEL[k]}</span>'
                    f'<span class="trk"><span class="fil" style="width:{p:.1f}%;background:{STAR_COLOR[k]}">'
                    f'</span></span><span class="pct">{p:.0f}%</span></div>')
    return "".join(rows)


def logo_text(name):
    words = [w for w in str(name).replace("(", " ").replace(")", " ").split() if w[:1].isalnum()]
    return escape("".join(w[0] for w in words[:2]).upper() or "IT")


def company_hero(r, n_rev, avg):
    tags = [x for x in [r["company_type"], r["industry"], r["company_size"], r["country"]]
            if isinstance(x, str) and x]
    cities = (r["cities"] or "").replace("|", ", ") if isinstance(r["cities"], str) else ""
    link = (f' · <a href="{escape(r["href"])}" target="_blank">Xem trên ITViec ↗</a>'
            if isinstance(r["href"], str) and r["href"].startswith("http") else "")
    score = (f'<div class="it-score"><b>{vn(avg)}</b>{stars(avg, 26)}'
             f'<span class="it-meta">{vn(n_rev, 0)} review</span></div>'
             if n_rev else '<div class="it-score"><span class="it-meta">Chưa có review chi tiết</span></div>')
    card(f'<div class="it-hero"><div class="it-logo">{logo_text(r["company_name"])}</div><div>'
         f'<div class="it-title">{escape(r["company_name"])}</div>'
         f'<div class="it-sub">{escape(cities)}{link}</div>{score}'
         f'<div>{"".join(f"<span class=it-tag>{escape(t)}</span>" for t in tags)}</div></div></div>')


def kpis(items):
    cols = "".join(f'<div class="it-kpi">{escape(k)}<b>{v}</b></div>' for k, v in items)
    card(f'<div style="display:grid;grid-template-columns:repeat({len(items)},1fr);gap:12px">{cols}</div>')


def summary(title, text):
    html(f'<div class="it-sum"><div class="hd">✨ {escape(title)}</div>{text}</div>')


def _para(label, text, limit=320):
    text = " ".join(str(text or "").split())
    if not text:
        return ""
    if len(text) <= limit:
        return f'<div class="it-txt"><span class="k">{label}</span> {escape(text)}</div>'
    return (f'<div class="it-txt"><span class="k">{label}</span> {escape(text[:limit])}…'
            f'<details><summary>Xem thêm</summary>{escape(text)}</details></div>')


SUB = [("salary", "Lương & phúc lợi"), ("training", "Đào tạo"), ("management", "Quản lý"),
       ("culture", "Văn hoá"), ("office", "Văn phòng")]


def review_card(r, idx=0, show_company=False):
    rating = int(r["rating"]) if pd.notna(r["rating"]) else 3
    yes = r["recommend"] == 1
    badge = ('<span class="it-badge it-yes">👍 Recommend</span>' if yes
             else '<span class="it-badge it-no">👎 Không recommend</span>')
    date = r["cmt_date"].strftime("Tháng %m/%Y") if pd.notna(r["cmt_date"]) else ""
    who = escape(r["company_name"]) if show_company else "Nhân viên ẩn danh"
    k = idx % len(AVA_COLORS)
    subs = "".join(f'<span>{lab} <b>{int(r[c])}★</b></span>' for c, lab in SUB if pd.notna(r.get(c)))
    pred_yes = r.get("pred") == 1
    ok = pred_yes == yes
    reply = ""
    if r.get("_split") == "test" and "pred" in r and pd.notna(r["pred"]):
        # Chỉ hiện ✅/❌ cho review thuộc tập test (mô hình chưa từng thấy); review train bị "học thuộc" nên không phản ánh chất lượng thật
        reply = (f'<div class="it-reply"><div class="who">🤖 Mô hình Recommend or Not</div>'
                 f'Dự đoán: <b>{"Recommend" if pred_yes else "Không recommend"}</b> — '
                 f'{"✅ khớp" if ok else "❌ khác"} với lựa chọn thật của người viết'
                 f' · <span class=it-meta>review thuộc tập test (mô hình chưa thấy khi huấn luyện)</span></div>')
    elif "pred" in r and pd.notna(r["pred"]):
        reply = ('<div class="it-reply"><div class="who">🤖 Mô hình Recommend or Not</div>'
                 '<span class=it-meta>Review này nằm trong tập huấn luyện nên không dùng để minh hoạ độ chính xác. '
                 'Dùng trang “Viết review” để thử review mới.</span></div>')
    card(f'<div class="it-top"><div class="it-ava" style="background:{AVA_COLORS[k]};color:{AVA_TEXT[k]}">'
         f'{"NV" if not show_company else logo_text(r["company_name"])}</div><div>'
         f'<div class="it-name">{who}</div><div class="it-meta">{date}</div></div></div>'
         f'<hr class="it-hr"><div>{stars(rating, 20)}{badge}</div>'
         f'<div class="it-h">{escape(str(r["title"] or ""))}</div>'
         f'{_para("👍 Điều thích:", r["liked"])}{_para("💡 Cần cải thiện:", r["suggestion"])}'
         f'<div class="it-sub5">{subs}</div>{reply}')


def team_sidebar(team):
    st.sidebar.markdown(
        '<div class="it-team"><b>Nhóm thực hiện</b><br>'
        + "".join(f'{escape(m["ten"])}<br><span style="color:#6b6b6b">{escape(m["email"])}</span><br>'
                  for m in team) + "</div>", unsafe_allow_html=True)
