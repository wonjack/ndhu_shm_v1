# project2026 — 結構主頻演化分析流程

結合解迴旋與時頻分析之結構主頻演化研究的資料處理流程（東華大學校園建物觀測）。

## Pipeline 版本

- `main_v5_D001.py` — NDHU 人社院（4F RC，sensor at 2F）
- `main_v5_D002.py` / `D003.py` / `D005.py` / `D006.py` — 其他測站

v5 包裝 `src.pipeline_common_v5.run_pipeline`，在 v4 基礎上加入 reviewer_v4 的程式碼修正：
- (3) 對稱 bandpass `f_min` 敏感度檢查
- (5) 顯式 `segment_source` 欄位
- (6) `baseline_comparison_pga_matched`
- (10) `baseline_comparison_deseasonalized`

## 結構

```
main_v5_*.py        各測站進入點
src/
  pipeline_common_v5.py   v5 主流程
  pipeline_common.py      v4 主流程
  processor.py            訊號處理
  matcher*.py             事件比對
  plotter.py              繪圖
  station_config.py       測站設定
  ...
requirements.txt
```

## 安裝

```bash
pip install -r requirements.txt
```

## 執行

```bash
python main_v5_D001.py
```

輸出寫到 `output_v5_<station>/`（已被 `.gitignore` 排除）。
