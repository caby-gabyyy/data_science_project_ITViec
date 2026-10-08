# GUI ITViec — Đánh giá công ty, gợi ý công ty tương tự & Recommend or Not

Ứng dụng Streamlit cho project 2 (đồ án tốt nghiệp Data Science, TTTH ĐH KHTN), bố cục kiểu trang review:
hồ sơ công ty (điểm, phân bố sao, tóm tắt, thẻ review), cột công ty tương tự, trang viết review có dự đoán.

## Chạy trên máy

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cấu trúc

| File | Vai trò |
|---|---|
| `app.py` | Trang chủ · Hồ sơ công ty · Tìm công ty · Viết review (Recommend or Not) · Dữ liệu & mô hình · Nhóm |
| `ui.py` | Thành phần HTML/CSS: thang sao, thanh phân bố, thẻ review, khối hồ sơ công ty |
| `itviec_core.py` | `clean_text`, stopwords, teencode, hàm gợi ý — chép từ notebook project_2 |
| `train_models.py` | Train lại BT1 (TF-IDF sklearn, Gensim TF-IDF, Gensim LSI) + BT2 (LinearSVC, LogisticRegression) từ parquet của project_2 |
| `models/` | `itviec_bt1.joblib`, `itviec_bt2.joblib` |
| `data/` | `companies.parquet`, `reviews.parquet` (review + dự đoán của model), bảng so sánh, `sample_reviews.csv` (review mẫu tự viết) |
| `images/` | Biểu đồ EDA / đánh giá xuất từ notebook |
| `.streamlit/config.toml` | Màu giao diện |

Dữ liệu ITViec chỉ dùng cho mục đích học tập. `train_models.py` cần `project_2/df_companies_clean.parquet`,
`df_reviews_clean.parquet` và `df_model_bt2.parquet` ở máy local.

## Deploy Streamlit Cloud

1. Tạo repo GitHub (public), upload **toàn bộ nội dung thư mục này**, kể cả thư mục `.streamlit` (không có file > 100MB).
2. share.streamlit.io → Create app → Deploy a public app from GitHub → chọn repo, branch `main`, main file `app.py`.
3. Advanced settings → Python **3.12** → Deploy.
