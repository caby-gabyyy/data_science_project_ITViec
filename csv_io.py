"""Đọc file CSV người dùng upload với nhiều bảng mã.

File mở bằng Excel rồi Save thường bị đổi từ UTF-8 sang bảng mã Windows: cp1258 (Windows tiếng Việt)
hoặc cp1252 (Windows tiếng Anh). Với cp1252, chữ có dấu tiếng Việt không có trong bảng mã sẽ bị Excel
thay bằng '?' — mất hẳn, app chỉ có thể cảnh báo."""
import io
import re
import unicodedata

import pandas as pd

# '?' kẹp giữa 2 chữ cái (vd 'M?i tr??ng') = dấu hiệu chữ có dấu đã bị thay thế khi lưu file
LOST_RE = re.compile(r"[^\W\d_]\?+[^\W\d_]")

GUIDE = ("Mở file bằng Excel → **File → Save As** → chọn kiểu **CSV UTF-8 (Comma delimited) (*.csv)** "
         "rồi upload lại — hoặc dùng lại file mẫu tải từ app (chưa mở bằng Excel).")


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


def read_csv_any(uploaded):
    """Trả về (DataFrame, bảng mã đã dùng, số ô có dấu hiệu mất chữ có dấu)."""
    raw = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
    text, enc = _decode(raw)
    df = pd.read_csv(io.StringIO(unicodedata.normalize("NFC", text)))
    obj = df.select_dtypes(include="object")
    n_lost = int(obj.apply(lambda s: s.astype(str).str.contains(LOST_RE)).to_numpy().sum()) if len(obj.columns) else 0
    return df, enc, n_lost


def warn_encoding(st, enc, n_lost):
    """Hiển thị cảnh báo phù hợp sau khi đọc file."""
    if n_lost:
        st.warning(f"⚠️ File có **{n_lost} ô** chứa chữ tiếng Việt bị mất dấu thành '?' "
                   f"(file được lưu bằng bảng mã `{enc}`, thường do mở bằng Excel rồi Save). "
                   f"Kết quả cho các dòng này sẽ kém chính xác. {GUIDE}")
    elif enc != "utf-8":
        st.info(f"File được lưu bằng bảng mã Windows (`{enc}`) — app đã tự chuyển đổi.")
