# Explainable Software Defect Prediction Prototype

Prototype codebase for the project **Explainable Software Defect Prediction Using Machine Learning and Commit Message Analysis**.

## Goals
- Build a reproducible data pipeline for software defect datasets.
- Engineer features from software metrics and commit messages.
- Train and compare Random Forest, XGBoost, and LightGBM.
- Explain predictions with SHAP.
- Provide a Streamlit demo for inspection and presentation.

## Project Structure
`	ext
prototype/
├── data/
├── notebooks/
├── src/
├── scripts/
├── models/
├── results/
├── paper/
└── tests/
`

## Phase 1 Status
Phase 1 is the foundation/setup stage. The current baseline includes:
- a stable project scaffold,
- reusable path and seed helpers,
- YAML-based configuration,
- ingest/clean/unify/validate scaffolds,
- smoke-test style script entry points,
- documentation for the baseline workflow.

## Setup
### 1. Create environment
`ash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
`

On Windows PowerShell, use:
`powershell
.venv\Scripts\Activate.ps1
`

### 2. Run the foundation checks
`ash
python scripts/run_data_pipeline.py
python scripts/run_feature_pipeline.py
`

These commands should:
- discover raw datasets under data/raw/,
- standardize and clean them,
- save processed outputs under data/processed/,
- generate inventory / summary tables under data/interim/.

### 3. Optional downstream checks
Once baseline artifacts exist, you can also run:
`ash
python scripts/run_train_metrics_only.py
python scripts/run_evaluation.py
python scripts/run_shap.py
`

### 4. Run the demo app
`ash
streamlit run src/app/streamlit_app.py
`

## Current Scope
This repository is currently focused on the **metrics-based baseline** first.

The stable foundation covers:
- directory layout and path helpers,
- reproducible configuration and seed utilities,
- data ingestion, cleaning, schema unification, and validation scaffolds,
- CLI scripts for data/feature/training/evaluation stages,
- paper support notes.

## Recommended Next Steps
1. Build the final metrics-only feature matrix in src/features/metrics_features.py
2. Connect the metrics-only experiment flow end-to-end in scripts/run_train_metrics_only.py
3. Finalize evaluation outputs in src/evaluation/compare.py
4. Add SHAP outputs for the best model
5. Wire the Streamlit demo to the saved artifacts

## Notes
- Keep experiment settings in src/config/config.yaml
- Save intermediate artifacts under data/processed/
- Save result tables and figures under 
esults/
- Save trained models under models/
- Do not commit generated caches, logs, or local data artifacts


# BỐI CẢNH DỰ ÁN (PROJECT CONTEXT)
Đây là dự án xây dựng baseline pipeline chuẩn mực cho các mô hình học máy. Hệ thống xử lý toàn bộ quy trình từ tiền xử lý dữ liệu, giải quyết vấn đề mất cân bằng (ví dụ: áp dụng kỹ thuật over-sampling như SMOTE), cho đến việc huấn luyện và đánh giá các thuật toán cốt lõi (như Decision Trees, Linear Regression). Mục tiêu là tạo ra một mã nguồn module hóa cao, dễ tái sử dụng và mở rộng.

# CẤU TRÚC THƯ MỤC CHUẨN (ARCHITECTURE)
Dự án phải tuân thủ nghiêm ngặt cấu trúc sau:
*   data/: Chứa dữ liệu đầu vào (raw) và dữ liệu đã qua xử lý (processed). Không push data lên git.
*   scripts/: Các file thực thi độc lập dùng để khởi chạy các bước trong pipeline (ví dụ: run_training.py).
*   src/: Thư mục chứa mã nguồn cốt lõi (logic chính).
    *   src/config/: File cấu hình (config.yaml).
    *   src/data_processing/: Chứa các hàm làm sạch, scale và cân bằng dữ liệu.
    *   src/models/: Các class/hàm định nghĩa mô hình.
*   
esults/: Nơi lưu trữ output như ma trận đánh giá (metrics), logs và model weights.

# QUY TẮC LẬP TRÌNH VÀ LOGIC (CLEAN CODE RULES)

1. Phân tách trách nhiệm (Modularity):
Tách biệt hoàn toàn phần xử lý dữ liệu khỏi thuật toán huấn luyện. Không viết code gộp nhiều chức năng phức tạp vào chung một hàm.

2. Tiêu chuẩn viết hàm và biến:
*   Mọi hàm bắt buộc phải có Docstring giải thích rõ ràng tham số đầu vào (args) và đầu ra (returns).
*   Sử dụng Type Hinting (ví dụ: def train_model(data: pd.DataFrame) -> dict:).
*   Tên biến phải mang ý nghĩa rõ ràng, phản ánh đúng loại dữ liệu nó chứa.

3. Kiểm soát logic luồng và vòng lặp (Strict Logic):
*   Đặc biệt cẩn thận khi sử dụng các biến cờ (boolean flag) bên trong các vòng lặp kiểm tra điều kiện. Phải đảm bảo vị trí cập nhật trạng thái True/False và lệnh 
eturn/reak được đặt đúng chỗ, tránh việc trạng thái đúng bị ghi đè sai lệch ở các bước lặp cuối cùng.
*   Luôn có các bước kiểm tra (validation) đầu vào trước khi chạy qua các vòng lặp tốn tài nguyên.

4. Xử lý lỗi (Error Handling):
Sử dụng 	ry-except ở các luồng đọc/ghi file hoặc các bước chuyển đổi dữ liệu dễ xảy ra lỗi định dạng.