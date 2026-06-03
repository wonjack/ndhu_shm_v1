# project2026 — 結構主頻演化分析流程 (v5)

結合解迴旋（water-level deconvolution）與時頻分析（CWT / sliding-window TF），追蹤花蓮地震序列前後、東華大學校園 RC 建物的主頻演化。NCREE 強震站為自由場參考輸入，D-系列 / W10F 測站為結構反應輸出。

## 內容

- [Pipeline 版本](#pipeline-版本)
- [處理步驟總覽](#處理步驟總覽)
- [Step 1 — 事件目錄與訊號載入](#step-1--事件目錄與訊號載入)
- [Step 2 — 重採樣與時窗切片](#step-2--重採樣與時窗切片)
- [Step 3 — 轉移函數 (Water-level Deconvolution)](#step-3--轉移函數-water-level-deconvolution)
- [Step 4 — STA/LTA + Arias 時段切分](#step-4--stalta--arias-時段切分)
- [Step 5 — 三段操作模態頻率 (Astorga 2018)](#step-5--三段操作模態頻率-astorga-2018)
- [Step 6 — Sliding-window 時變 TF](#step-6--sliding-window-時變-tf)
- [Step 7 — CWT 連續小波響應](#step-7--cwt-連續小波響應)
- [Step 8 — 阻尼比 (Half-power)](#step-8--阻尼比-half-power)
- [Step 9 — 每事件輸出彙整](#step-9--每事件輸出彙整)
- [Step 10 — 跨事件趨勢分析](#step-10--跨事件趨勢分析)
- [Step 11 — Baseline 比較 (v5 新增)](#step-11--baseline-比較-v5-新增)
- [檔案結構](#檔案結構)
- [安裝與執行](#安裝與執行)
- [參考文獻](#參考文獻)

---

## Pipeline 版本

| Entry point | 測站 | 建物 | Sensor 位置 |
|---|---|---|---|
| `main_v5_D001.py` | D001 | NDHU 人社院 (4F RC) | 2F |
| `main_v5_D002.py` | D002 | NDHU 花師教育學院 | — |
| `main_v5_D003.py` | D003 | NDHU 圖書館 | — |
| `main_v5_D005.py` | D005 | NDHU 學生活動中心 | — |
| `main_v5_D006.py` | D006 | NDHU 行政大樓 | 1F + 4F（唯一可做 inter-story SSI 解迴旋） |

v5 包裝 `src.pipeline_common_v5.run_pipeline`，在 v4 基礎上加入 reviewer_v4 四項程式碼修正：

- **§3** 對稱 bandpass `f_min_sym` 偏差檢查（新增 CSV 欄位 + `summary_fmin_symmetric_vs_asymmetric.{png,json}`）
- **§5** 顯式 `segment_source` 欄位，標明 STA/LTA + Arias 切分跑在結構記錄
- **§6** `baseline_comparison_pga_matched.{png,json}` — 以 log-PGA 最近鄰配對 pre/post-mainshock 事件
- **§10** `baseline_comparison_deseasonalized.{png,json}` — 月中位數 + STL (period=12) 去季節後再做 bootstrap

主震 (Mainshock)：2024-04-02 23:58:11 UTC，event id `20240402235754`。

---

## 處理步驟總覽

```
事件目錄 → 載入 NCREE + 結構雙站三分量
        ↓
   重採樣 100 Hz, 切 [-30, +120] s 視窗, demean
        ↓
   ┌─ Transfer Function (water-level) → f1, ζ
   │   └─ Sliding-window TF (10s/5s)  → H(f, t)
   ├─ STA/LTA + Arias 5/95/99%        → t_first, t5, t95, t99, arias_total
   │   └─ 三段 FFT/IF                  → f_iapp, f_min (asym + sym), f_99app
   ├─ CWT 連續小波                    → ridge(t) in code band
   └─ FFT 平滑頻譜 (Konno-Ohmachi)    → fft_peak_ncree, fft_peak_<sid>
        ↓
   寫入 structural_history_log_v5_<sid>.csv （每事件 × 3 分量 = 一列）
        ↓
   跨事件彙整：
     - 線性趨勢 (OLS + Newey-West HAC SE + BH-FDR)
     - Pre/Post mainshock bootstrap (±90 d)
     - PGA-matched 配對比較        ← v5 §6
     - STL 去季節後比較             ← v5 §10
     - f_min 對稱 vs 不對稱對照     ← v5 §3
```

---

## Step 1 — 事件目錄與訊號載入

- 目錄比對於 `src/matcher.py` 與各測站專屬的 `matcher_ncree_d00X.py`
- 每事件以 `event.event_id` 命名子資料夾
- 載入：
  - **NCREE**：自由場輸入（borehole 地表伴隨站，距各 D 站 230–1077 m）
  - **結構站**：依 `cfg.palert_attr` 取得 `{E, N, Z}` 三分量 SAC

關鍵程式碼：`src/pipeline_common_v5.py::_load_traces`

---

## Step 2 — 重採樣與時窗切片

| 參數 | 值 | 來源 |
|---|---|---|
| `TARGET_FS` | 100 Hz | 統一所有測站 |
| `PRE_EVENT_SEC` | 30 s | NCREE trigger 前 |
| `POST_EVENT_SEC` | 120 s | NCREE trigger 後 |
| Tukey α | 0.05 | FFT/TF 前端錐形窗 |

`_resample_to` 用 ObsPy `Trace.resample`；`_slice_window` 以 NCREE Z trigger 時刻為 `t=0`，視窗外補 NaN，最後做 demean。

---

## Step 3 — 轉移函數 (Water-level Deconvolution)

$$H(f) = \frac{Y(f)\,\overline{X(f)}}{\max(|X(f)|^2,\ \varepsilon \cdot |X|^2_{\max})}$$

- `WATERLEVEL_EPS = 0.05`（防止低能量頻段炸開）
- **平滑**：Konno-Ohmachi (`bandwidth=30`, `KO_LOG_BINS=300`)
- **峰值**：
  - `tf_peak_freq` — 無限制 [0.4, 15] Hz 的全域峰值
  - `tf_f1` — 約束在規範經驗頻帶 `[f_emp_min, f_emp_max]` 內
  - `tf_peak_in_codeband` — 布林欄位：無限制峰值是否落在規範頻帶內（破除循環邏輯）

輸出：每事件 `transfer_fn_{E,N,Z}.png`

關鍵程式碼：`_transfer_function`, `_plot_transfer_function`

---

## Step 4 — STA/LTA + Arias 時段切分

跑在**結構站**記錄 (`segment_source = "structure"`)：

| 參數 | 值 |
|---|---|
| STA 視窗 | 1.0 s |
| LTA 視窗 | 10.0 s |
| STA/LTA 觸發門檻 | 2.0 |
| Arias 重力常數 | 9.80665 m/s² |

輸出時間點（相對 NCREE trigger，秒）：
- `t_first_arrival_sec` — STA/LTA 首觸發
- `t5_sec` / `t95_sec` / `t99_sec` — Arias 5% / 95% / 99% 累積能量到達時刻
- `arias_total` — $I_a = \tfrac{\pi}{2g}\int a^2\,dt$（cm/s）

關鍵程式碼：`_event_segments`

---

## Step 5 — 三段操作模態頻率 (Astorga 2018)

| 名稱 | 時段 | 方法 |
|---|---|---|
| `f_iapp` | `[0, t_first]`（pre-event） | Konno-Ohmachi 平滑 FFT 在規範頻帶內取峰 |
| `f_min` | `[t5, t95]`（強震段） | 對稱/不對稱 bandpass + Hilbert 瞬時頻率中位數 |
| `f_99app` | `[t95, t99]`（衰減段） | 同 `f_iapp` |

**`f_min` 帶通模式（reviewer §3）**

```
asymmetric (v4 預設):  [f_iapp / 1.7,    f_iapp * 1.15]   → 偏軟化偏差
symmetric  (v5 新增):  [f_iapp / 1.398,  f_iapp * 1.398]  → r = √(1.7·1.15)
```

`f_min_diff_pct = (f_min_sym − f_min) / f_iapp × 100%` 記錄到 CSV，並於彙整階段繪出分佈對照。

**Hilbert 瞬時頻率流程**：6 階 Butterworth bandpass → `hilbert()` → unwrap 相位 → 一階差分 → 1 秒 box smoothing → 取 envelope > 50% peak 且 IF 在頻帶內的樣本，回傳中位數。

關鍵程式碼：`_modal_in_segment`, `_f_min_strong_motion`

---

## Step 6 — Sliding-window 時變 TF

回應 reviewer §2.1：事件平均 TF 混合了 pre-event 線性彈性、共震非線性、post-event 恢復三段，違反 LTI 假設。改用滑動視窗解迴旋（Nakata & Snieder 2014; Mikael et al. 2013）：

- Window = 10 s, overlap = 5 s
- 對每個 window 重複 water-level deconvolution
- 在規範頻帶內取 ridge，疊上 STA/LTA + Arias 標記線

輸出：每事件 `transfer_fn_tv_{E,N,Z}.png`

關鍵程式碼：`_sliding_window_tf`, `_plot_sliding_tf`

---

## Step 7 — CWT 連續小波響應

- 對數頻率網格 120 bins, `fmin=0.4` Hz, `fmax=15` Hz
- 在規範頻帶內逐時間取最大功率位置 → 一階模態 ridge
- 圖上疊放 `f_iapp / f_min / f_99app` 三點與切分時間線

輸出：每事件 `cwt_{E,N,Z}.png`

關鍵程式碼：`_cwt_response`（呼叫 `src/processor.py::compute_cwt`）, `_plot_cwt_annotated`

---

## Step 8 — 阻尼比 (Half-power)

於 `f1`（規範頻帶內 TF 峰值）附近 ±1.5 Hz 取半功率寬度：

$$\zeta = \frac{f_2 - f_1}{2 f_0}$$

`damping_ratio` 欄位限制 $0 < \zeta < 0.5$，否則記為 NaN。

關鍵程式碼：`_damping_from_raw_H`, `_half_power_damping`

---

## Step 9 — 每事件輸出彙整

每事件 × 3 分量 (E/N/Z) = 一列寫入 `structural_history_log_v5_<sid>.csv`。

主要欄位（完整列表見 `csv_columns()`）：

| 欄位 | 說明 |
|---|---|
| `event_id`, `utc_time`, `local_time` | 事件識別 |
| `magnitude`, `depth_km`, `epicenter` | CWA 目錄欄位 |
| `pga_0m / 20m / 30m / 58m / 78m` | 對應深度的 PGA |
| `comp` | E / N / Z |
| `tf_peak_freq`, `tf_peak_amp`, `tf_f1`, `tf_peak_in_codeband` | 解迴旋峰值 |
| `f_empirical`, `f_emp_min`, `f_emp_max` | 規範經驗頻率（依結構高度） |
| `f_iapp`, `f_min`, `f_99app`, `f_min_sym`, `f_min_diff_pct` | 三段操作模態 |
| `t_first_arrival_sec`, `t5_sec`, `t95_sec`, `t99_sec`, `arias_total` | 切分時間 + Arias |
| `damping_ratio` | 半功率阻尼 |
| `fft_peak_ncree`, `fft_peak_<sid>` | 兩站獨立 FFT 峰值 |
| `ncree_distance_m`, `vs30_ms`, `site_class` | 部署元資料 |
| `segment_source` | 固定 `"structure"`（v5 §5） |
| `status` | `OK` / `too_short` / `feat_fail:<msg>` |

每事件圖片：`raw_{comp}.png`, `fft_{comp}.png`, `transfer_fn_{comp}.png`, `transfer_fn_tv_{comp}.png`, `cwt_{comp}.png`

---

## Step 10 — 跨事件趨勢分析

每測站 CSV 跑完之後做：

- **OLS 線性趨勢** `slope per year` + **Newey-West HAC SE**（餘震叢集造成殘差時間相關，平實 OLS SE 有偏；lag $L \approx \lfloor 4(T/100)^{2/9} \rfloor$）
- **Benjamini-Hochberg FDR** — 多重檢定（每測站 × 每特徵）的 q-value
- 滾動 30 點中位數曲線

關鍵程式碼：`_linear_fit`, `_bh_fdr`, `_rolling_trend`

---

## Step 11 — Baseline 比較 (v5 新增)

針對主震前後的中位數差異，給出三種互補檢定：

| 輸出 | 邏輯 | 對應 reviewer 點 |
|---|---|---|
| `baseline_comparison.{png,json}` | ±90 d 視窗中位數 + 1000× bootstrap CI | v4 既有 |
| `baseline_comparison_pga_matched.{png,json}` | 每個 post 事件配對 log-PGA 最近的 pre 事件，匹配子樣本中位差 | §6 |
| `baseline_comparison_deseasonalized.{png,json}` | 月中位數 + STL (period=12) 殘差+趨勢，再跑 ±90 d bootstrap | §10 |
| `summary_fmin_symmetric_vs_asymmetric.{png,json}` | $\Delta f / f$ 在對稱 vs 不對稱 bandpass 下的分佈 | §3 |

額外限制說明 (`SSI_LIMITATION_NOTE`) 會印在主要 summary 圖的 caption：
- NCREE 與各 D 站有 230–1077 m 距離，存在 path effects
- 觀測到的 coseismic `f_min` 下降混合了結構非線性與土壤剪切模量退化（Todorovska 2009; Stewart et al. 1999）
- 只有 D006 有 1F+4F sensors，可做 inter-story 解迴旋直接消去 SSI（Snieder & Şafak 2006）；其餘測站僅以 Veletsos-Meek 上限估 SSI（NIST GCR 12-917-21 §2），輸出於 `ssi_summary/`

---

## 檔案結構

```
project2026/
├── main_v5_D001.py … main_v5_D006.py    # 各測站進入點
├── src/
│   ├── pipeline_common_v5.py   # v5 主流程（本檔重點）
│   ├── pipeline_common.py      # v4 主流程（保留供回溯比對）
│   ├── processor.py            # CWT, signal processing primitives
│   ├── matcher.py              # 通用事件目錄比對
│   ├── matcher_ncree_d00X.py   # 各測站專屬目錄/路徑解析
│   ├── plotter.py              # 共用繪圖工具
│   ├── station_config.py       # STATIONS dict, code_band, empirical_frequency
│   ├── ssi_bounds.py           # Veletsos-Meek SSI 上限估計
│   ├── d006_interstory.py      # D006 inter-story 解迴旋
│   ├── sensitivity_sweep.py    # 參數敏感度掃描
│   ├── backfill_pga.py         # 補算 PGA 欄位
│   └── epicenter_plots.py      # 震央分佈圖
├── requirements.txt
├── .gitignore
└── README.md (本檔)
```

執行後產生（已在 `.gitignore` 排除）：

```
output_v5_<sid>/<event_id>/{raw,fft,transfer_fn,transfer_fn_tv,cwt}_{E,N,Z}.png
structural_history_log_v5_<sid>.csv
processing_errors_v5_<sid>.log
skipped_events_v5_<sid>.log
```

---

## 安裝與執行

```bash
pip install -r requirements.txt
# 依賴：obspy, PyWavelets, scipy, pandas, tqdm, matplotlib, numpy
# 額外需要：statsmodels (for HAC), 自行 pip install statsmodels
```

執行單一測站：

```bash
python main_v5_D001.py
```

輸出目錄寫死在 `BASE_DIR = D:\PHD\project2026`，跨機器執行請修改 `src/pipeline_common_v5.py::BASE_DIR`。

---

## 參考文獻

- Astorga, A. et al. (2018) — operational modal `f_iapp / f_min / f_99app` 三段定義
- Nakata, N. & Snieder, R. (2014) BSSA — sliding-window deconvolution
- Mikael, A. et al. (2013) — time-varying TF for SHM
- Snieder, R. & Şafak, E. (2006) BSSA — inter-story deconvolution (D006)
- Todorovska, M. I. (2009) BSSA — soil nonlinearity contribution
- Stewart, J. P. et al. (1999) — SSI analytical methods
- NIST GCR 12-917-21 — Veletsos-Meek SSI bounds
- Newey, W. K. & West, K. D. (1994) Econometrica — HAC lag rule
- Benjamini, Y. & Hochberg, Y. (1995) JRSS — FDR control
