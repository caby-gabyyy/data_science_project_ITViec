"""Đọc file người dùng upload (CSV nhiều bảng mã hoặc Excel .xlsx) + khôi phục chữ tiếng Việt bị mất dấu.

File CSV mở bằng Excel rồi Save thường bị đổi từ UTF-8 sang bảng mã Windows: cp1258 (Windows tiếng Việt,
giữ được dấu) hoặc cp1252 (Windows tiếng Anh) — với cp1252, mỗi chữ không có trong bảng mã (ư, ơ, ă, đ,
ạ, ả, ờ, ố…) bị thay bằng đúng 1 dấu '?'. Hàm repair_df đoán lại chữ đó bằng từ điển âm tiết xây từ dữ liệu
của đồ án: '?' chỉ có thể là 1 trong ~90 chữ bị mất, các chữ còn lại của âm tiết phải khớp chính xác,
chọn ứng viên theo tần suất + ngữ cảnh từ đứng trước (bigram)."""
import io
import re
import unicodedata
from collections import Counter, defaultdict
from html import escape

import pandas as pd

GUIDE = ("Cách tránh: lưu file dạng **Excel (.xlsx)** rồi upload — hoặc trong Excel chọn **File → Save As → "
         "CSV UTF-8 (Comma delimited) (*.csv)**.")

_VI = ("aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵđ")
_VI = _VI + _VI.upper()


def _lost_in_cp1252(c):
    try:
        c.encode("cp1252")
        return False
    except UnicodeEncodeError:
        return True


LOST_CHARS = "".join(c for c in _VI if _lost_in_cp1252(c))        # chữ bị Excel cp1252 thay bằng '?'
_LOST_CLASS = f"[{re.escape(LOST_CHARS)}]"
LOST_RE = re.compile(r"[^\W\d_]\?+[^\W\d_]|[^\W\d_]\?+(?=\s|$)|(?:^|(?<=\s))\?+[^\W\d_]")
_WORD = re.compile(r"[^\W\d_]+")


# ----------------------------------------------------------------------
# Đọc file
# ----------------------------------------------------------------------
def _decode(raw):
    """UTF-8 trước; không được thì chọn giữa cp1258 và cp1252.
    cp1258 ghi dấu thanh bằng ký tự tổ hợp đứng sau nguyên âm -> sau NFC phải ghép hết; nếu còn
    dấu tổ hợp đứng lẻ thì file thực ra là cp1252 (các byte đó là à/ì/ò… chứ không phải dấu thanh)."""
    try:
        return raw.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        pass
    t1258 = unicodedata.normalize("NFC", raw.decode("cp1258", errors="replace"))
    if not any(unicodedata.category(c) == "Mn" for c in t1258):
        return t1258, "cp1258"
    return raw.decode("cp1252", errors="replace"), "cp1252"


def count_lost(df):
    obj = df.select_dtypes(include="object")
    if not len(obj.columns):
        return 0
    return int(obj.apply(lambda s: s.astype(str).str.contains(LOST_RE)).to_numpy().sum())


def read_any(uploaded):
    """CSV (mọi bảng mã) hoặc Excel .xlsx -> (DataFrame, định dạng/bảng mã, số ô có chữ bị mất dấu)."""
    name = getattr(uploaded, "name", "")
    raw = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
    if name.lower().endswith((".xlsx", ".xls")):
        df, enc = pd.read_excel(io.BytesIO(raw)), "xlsx"
    else:
        text, enc = _decode(raw)
        df = pd.read_csv(io.StringIO(unicodedata.normalize("NFC", text)))
    return df, enc, count_lost(df)


# ----------------------------------------------------------------------
# Khôi phục chữ bị mất dấu
# ----------------------------------------------------------------------
def build_lexicon(texts):
    """Từ điển âm tiết (chữ thường) + bigram, chỉ cần các âm tiết có chứa chữ dễ bị mất."""
    uni, bi = Counter(), Counter()
    for t in texts:
        words = _WORD.findall(unicodedata.normalize("NFC", str(t)).lower())
        uni.update(words)
        bi.update(zip(words, words[1:]))
    by_len = defaultdict(list)
    for w in uni:
        if any(c in LOST_CHARS for c in w):
            by_len[len(w)].append(w)
    lost_words = {w for ws in by_len.values() for w in ws}
    bi = {k: v for k, v in bi.items() if k[0] in lost_words or k[1] in lost_words}
    return {"uni": uni, "bi": bi, "by_len": dict(by_len)}


def _match_case(src, word, sentence_start):
    letters = [c for c in src if c.isalpha()]
    if len(letters) > 1 and all(c.isupper() for c in letters):
        return word.upper()
    if src[:1].isupper() or (src[:1] == "?" and (src[1:2].isupper() or sentence_start)):
        return word[:1].upper() + word[1:]
    return word


def _cands(token, lex):
    pat = re.compile("^" + "".join(_LOST_CLASS if c == "?" else re.escape(c) for c in token.lower()) + "$")
    return [w for w in lex["by_len"].get(len(token), ()) if pat.match(w)]


def _pick(cands, prev, nxt, lex):
    uni, bi = lex["uni"], lex["bi"]
    return max(cands, key=lambda w: uni[w] * (1 + 30 * bi.get((prev, w), 0)) * (1 + 30 * bi.get((w, nxt), 0)))


_TOKEN = re.compile(r"[\w?]+")


def repair_text(s, lex):
    """Trả về (chuỗi đã sửa, số từ sửa được, số từ không sửa được).
    Lượt 1 chọn ứng viên theo từ đứng trước; lượt 2 chọn lại theo cả từ trước lẫn từ sau."""
    if not isinstance(s, str) or "?" not in s:
        return s, 0, 0
    toks = list(_TOKEN.finditer(s))
    items = []                        # mỗi token: [start, end, chữ thường dùng làm ngữ cảnh, ứng viên, phần '?' thật ở cuối]
    for k, m in enumerate(toks):
        t = m.group()
        cands, tail = None, ""
        if "?" in t:
            gap_l = s[toks[k - 1].end():m.start()] if k else ""
            gap_r = s[m.end():toks[k + 1].start()] if k + 1 < len(toks) else "\n"
            only_q = not re.search(r"[^\W\d_]", t)
            # token toàn '?' chỉ coi là chữ bị mất khi nằm giữa 2 từ, cách nhau bằng khoảng trắng
            if not only_q or (k + 1 < len(toks) and len(t) <= 3 and gap_r.isspace()
                              and re.fullmatch(r"[\s.,;:!()\"'\-–]*", gap_l)):
                cands = _cands(t, lex)
                if not cands and t.endswith("?") and not only_q:          # '?' cuối có thể là dấu hỏi thật
                    core_ = t.rstrip("?")
                    tail = t[len(core_):]
                    cands = _cands(core_, lex) if "?" in core_ else None
                    if cands is None:                                    # từ còn nguyên + dấu hỏi thật
                        tail = ""
        items.append([m.start(), m.end(), t.lower().rstrip("?") if cands is None else "", cands, tail])

    def ctx(k, step):
        j = k + step
        return items[j][2] if 0 <= j < len(items) else ""

    for _ in range(2):                                                   # 2 lượt: trái -> phải, rồi có cả ngữ cảnh phải
        for k, it in enumerate(items):
            if it[3]:
                it[2] = _pick(it[3], ctx(k, -1), ctx(k, +1), lex)

    out, pos, fixed, miss = [], 0, 0, 0
    for k, (st_, en, word, cands, tail) in enumerate(items):
        out.append(s[pos:st_])
        src = s[st_:en]
        if cands is None:
            out.append(src)
        elif not cands:
            miss += 1
            out.append(src)
        else:
            fixed += 1
            before = s[:st_].rstrip(" \t")
            # đầu câu: đầu chuỗi, sau . ! xuống dòng, hoặc sau dấu hỏi thật (không dính chữ -> không phải chữ bị mất)
            sent = (not before or before[-1] in ".!\n"
                    or (before[-1] == "?" and (len(before) < 2 or not before[-2].isalpha() and before[-2] != "?")))
            out.append(_match_case(src, word, sent) + tail)
        pos = en
    out.append(s[pos:])
    return "".join(out), fixed, miss

def repair_df(df, lex):
    """Sửa mọi ô chữ có dấu hiệu mất dấu. Trả về (df mới, số từ sửa được, số từ chưa sửa được)."""
    df = df.copy()
    fixed = miss = 0
    for c in df.select_dtypes(include="object").columns:
        res = df[c].map(lambda v: repair_text(v, lex))
        df[c] = res.map(lambda t: t[0])
        fixed += int(res.map(lambda t: t[1]).sum())
        miss += int(res.map(lambda t: t[2]).sum())
    return df, fixed, miss


# ----------------------------------------------------------------------
# Giao diện: đọc + sửa + thông báo
# ----------------------------------------------------------------------
def load_upload(st, uploaded, lex):
    """Đọc file upload, tự khôi phục chữ mất dấu nếu có và hiển thị thông báo phù hợp."""
    df, enc, n_lost = read_any(uploaded)
    if not n_lost:
        if enc == "cp1258":
            st.info("File được lưu bằng bảng mã Windows tiếng Việt (`cp1258`) — app đã tự chuyển đổi.")
        return df
    before = df
    df, fixed, miss = repair_df(df, lex)
    total = fixed + miss
    msg = (f"🔧 File được lưu bằng bảng mã `{enc}` nên **{n_lost} ô** bị mất chữ có dấu (thành '?'). "
           f"App đã **tự khôi phục {fixed}/{total} từ** ({fixed / max(total, 1):.0%}) bằng từ điển tiếng Việt "
           f"xây từ dữ liệu đồ án.")
    (st.warning if miss else st.success)(msg + (f" {miss} từ chưa khôi phục được. {GUIDE}" if miss else ""))
    with st.expander("Xem trước / sau khi khôi phục"):
        # 1 danh sách duy nhất (Trước ở trên, Sau ở dưới) — vừa cả khung hẹp; tối đa 5 ô, mỗi dòng 1 ô
        items = []
        for i in before.index:
            for c in before.select_dtypes(include="object").columns:
                v = str(before.at[i, c])
                if LOST_RE.search(v) and len(items) < 5:
                    items.append((i, c, " ".join(v.split()), " ".join(str(df.at[i, c]).split())))
                    break
        cut = lambda s: escape(s[:110] + ("…" if len(s) > 110 else ""))
        st.markdown("".join(
            f'<div style="font-size:.85rem;line-height:1.45;padding:6px 0;border-bottom:1px solid #eee">'
            f'<span style="color:#8c8c8c">Dòng {i + 1} · {escape(c)}</span><br>'
            f'<span style="color:#b42318">Trước:</span> {cut(a)}<br>'
            f'<span style="color:#1d7a50">Sau:</span> <b>{cut(b)}</b></div>'
            for i, c, a, b in items), unsafe_allow_html=True)
    return df
