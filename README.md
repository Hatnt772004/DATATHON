# Dự Án DATATHON 2026 — Phân Tích Dữ Liệu Kinh Doanh & Dự Báo Doanh Thu

Dự án này là bộ giải pháp toàn diện cho cuộc thi DATATHON 2026, bao gồm 3 phần chính: Xử lý các câu hỏi phân tích dữ liệu kinh doanh (Part 1), xây dựng hệ thống Business Intelligence Dashboard (Part 2), và thiết lập một Pipeline học máy (Machine Learning) toàn diện để dự báo Doanh thu (Revenue) và Giá vốn hàng bán (COGS). 

---

## 🛠 Yêu Cầu Cài Đặt (Requirements)

Đảm bảo bạn đã cài đặt Python (khuyến nghị 3.8+). Các thư viện cần thiết có thể được cài đặt thông qua lệnh sau:

```bash
pip install pandas numpy matplotlib statsmodels lightgbm xgboost prophet scikit-learn shap optuna
```

## 📂 Cấu trúc thư mục định tuyến

📁 dataset/
   
   └── sales.csv, customers.csv, orders.csv...   # File dữ liệu 

📁 output/                        # Thư mục chứa toàn bộ kết quả đầu ra của Machine Learning
   
   ├── 📁 models/                 # Chứa các mô hình đã huấn luyện (.pkl) và file cấu hình (.json)
   
   ├── 📁 submission/             # Chứa kết quả dự báo cuối cùng (CSV & PDF báo cáo)
   
   ├── 📁 shap_analysis/          # Chứa các biểu đồ giải thích mô hình từ SHAP
   
   ├── train.csv, val.csv...      # Dữ liệu đã chia tập
   
   ├── train_fe.csv...            # Dữ liệu sau Feature Engineering
   
   ├── feature_columns.txt        # Danh sách các đặc trưng (features) được sử dụng
   
   └── *.png                      # Các biểu đồ EDA, đánh giá metrics, walk-forward

📄 Part1.ipynb     # Trả lời 10 câu hỏi trắc nghiệm

📄 Part2.html      # Hệ thống Business Intelligence Dashboard

📄 Part3.ipynb                      # Pipeline Machine Learning dự báo chuỗi thời gian

📄 submission.csv        # File kết quả dự báo doanh thu 2023-2024

📄 README.md                      # Tài liệu hướng dẫn dự án (File này)

## 🚀 Cấu trúc dự án

### Phân tích số liệu và giao diện trực quan (Part 1 & 2)
Part 1 — Truy vấn & thống kê (part1.ipynb): Sử dụng pandas và numpy để xử lý lượng lớn dữ liệu bán hàng. Trả lời 10 câu hỏi cốt lõi của doanh nghiệp.

Part 2 — Master Dashboard (dashboard for part2.html): Một hệ thống báo cáo BI (Business Intelligence) tĩnh nhưng có tính tương tác cao được xây dựng trực tiếp bằng HTML và Chart.js. Dashboard bao gồm 8 phân hệ chuyên sâu: Overview, Revenue, Web Traffic, Customers, Profitability, Returns, Inventory, và Operations. Tính năng nổi bật bao gồm P&L Waterfall, phân tích phân tán lợi nhuận (Scatter), và Ma trận sức khỏe hàng tồn kho (Inventory Health Matrix).

### Xây dựng mô hình dự báo doanh thu (Part 3)

### Kiến Trúc Pipeline (6 Giai Đoạn)
Mã nguồn được chia thành 6 giai đoạn:

- Giai đoạn 1: Khám phá dữ liệu & phân tích chuỗi thời gian (EDA & Time Series Analysis): Tải và làm sạch dữ liệu, kiểm tra giá trị thiếu (missing values) và dữ liệu bất thường, tạo biến Covid_Decay_Shock với hàm phân rã mũ (Exponential Decay) để giúp mô hình hiểu được tác động giảm dần của đại dịch Covid-19, kiểm định tính dừng (Stationarity) bằng Augmented Dickey-Fuller (ADF) test, chia tập dữ liệu (Train/Validation/Test nội bộ).

- Giai đoạn 2: Trích xuất Đặc trưng (Feature Engineering)
Lag Features: Độ trễ 1, 7, 14, 30 và 365 ngày để bắt chu kỳ ngắn và dài hạn.
Rolling Statistics: Trung bình, độ lệch chuẩn, max, min trượt trên các khung thời gian (7, 14, 30, 90 ngày).
Calendar Features: Khai thác tính mùa vụ (DOW, Month) kết hợp mã hóa vòng (Sin/Cos encoding) và Chuỗi Fourier.
Growth & Ratio Features: Động lượng (momentum), tốc độ tăng trưởng YoY, MoM và Z-score ngắn hạn.
Interaction Features: Tương tác giữa các biến độ trễ, biến rolling với cú sốc COVID-19 và hiệu ứng cuối tuần.

- Giai đoạn 3: Lựa chọn & huấn luyện mô hình cơ sở: áp dụng Recency Weighting (gán trọng số mẫu cao hơn cho dữ liệu gần đây), huấn luyện và dò tìm siêu tham số tự động với Optuna (TPE Sampler) cho LightGBM và XGBoost, huấn luyện mô hình cơ sở Ridge Regression, tạo mô hình Ensemble dựa trên nghịch đảo của chỉ số MAPE, đánh giá và lưu các mô hình tốt nhất.

- Giai đoạn 4: Walk-forward Validation: Thực hiện kiểm định chéo trên chuỗi thời gian (Time-series Cross Validation) qua 6 Fold mở rộng theo năm, đánh giá khả năng dự báo của mô hình khi đi qua cú sốc cấu trúc năm 2020, chấm điểm độ ổn định của mô hình thông qua công thức: Score = Mean MAPE + 0.5 * Std MAPE để chọn ra mô hình tối ưu nhất.

- Giai đoạn 5: Đánh giá cuối cùng & dự báo Đệ quy (Recursive Forecasting): mô phỏng bài toán thực tế: đánh giá cuốn chiếu theo từng tháng trên tập Test 2022, phân tích lỗi chi tiết (theo tháng, theo thứ trong tuần, kiểm tra bias), retrain toàn bộ dữ liệu (2012 - 2022), dự báo đệ quy (Recursive Forecasting) cho 2023-2024: dự báo từng ngày, đắp kết quả dự báo vào làm dữ liệu lịch sử để tính toán lại toàn bộ Feature Engineering (Lag, Rolling, Growth) cho ngày tiếp theo, xuất báo cáo dưới định dạng CSV và PDF.

- Giai đoạn 6: Giải thích mô hình với SHAP: sử dụng TreeExplainer và LinearExplainer để tính giá trị SHAP cho từng mô hình thành viên, tính toán ma trận SHAP tổng hợp (Ensemble SHAP) dựa trên trọng số của từng mô hình, xuất các biểu đồ phân tích.

### 📊 Hiệu suất và đánh giá (Metrics)
Hệ thống sử dụng các thang đo tiêu chuẩn trong kinh tế lượng và học máy: MAPE, RMSE, MAE, R² 

### 💡 Cấu hình tham số chính (Configuration)
Bạn có thể thay đổi các biến toàn cục ở đầu script để điều chỉnh hành vi của pipeline:
- DATA_PATH: Đường dẫn file dữ liệu.
- TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END: Các mốc thời gian chia tập dữ liệu.
- LAMBDA_DECAY: Tốc độ phân rã cho trọng số mẫu (Sample Weight).
- LAMBDA_DECAY_RATE: Tốc độ phân rã của biến tác động COVID-19.

## 📌 Hướng dẫn sử dụng

Đảm bảo các file dữ liệu CSV (customers.csv, orders.csv...) nằm cùng một thư mục với file .ipynb và .html. 

1. Chạy Script phân Tích (Part 1)
Mở file Part1.ipynb và chạy toàn bộ các cell (Run All) để xem kết quả trích xuất dữ liệu cho 10 câu hỏi của cuộc thi.

2. Xem Business Dashboard (Part 2)
Nhấp đúp chuột vào Part2.html để mở bằng trình duyệt web của bạn, hoặc click chuột phải trong VS Code và chọn Open with Live Server.

3. Chạy Pipeline dự Báo học Máy
Mở file Part3.ipynb và chạy toàn bộ các cell (Run All)

Mở thư mục output/ để xem toàn bộ biểu đồ phân tích, tệp cấu hình JSON, báo cáo PDF, và tệp dự báo predictions_2023_2024.csv.
