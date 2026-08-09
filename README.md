# LighterGlue Edge Deployment: Transformer PTQ, SmoothQuant & HPatches Benchmark

[![ONNX](https://img.shields.io/badge/ONNX_Runtime-1.16+-blue.svg)](https://onnxruntime.ai/)
[![Quantization](https://img.shields.io/badge/PTQ-Dynamic_%7C_Static_QDQ_%7C_SmoothQuant-green.svg)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Target](https://img.shields.io/badge/Target-Edge_ARM_%2F_Drones_%2F_Robotics-orange.svg)]()

This tutorial demonstrates how to optimize, quantize, and deploy **LighterGlue** (*Local Feature Matching at Light Speed*, ICCV 2023 by Lindenberger et al.) for resource-constrained edge robotics platforms using **ONNX Runtime Post-Training Quantization (PTQ)** and **SmoothQuant**.

---

## Executive Summary & Motivation

In Visual SLAM and 3D perception pipelines, two-view temporal feature matching is allocated a strict **$\le 10\text{--}12$ ms** latency budget per frame to maintain high-frequency pose estimation:

```text
Camera Frame
   ├── Feature Detection & Extraction  --> [XFeat ONNX INT8 / FP16]
   ├── Two-View / Temporal Matching    --> [LighterGlue Dynamic INT8 / SmoothQuant]
   └── Spatial Tracking / PnP / BA/ GO --> [C++ Ceres / g2o Backend]
```

While CNNs (like XFeat) quantize straightforwardly, Transformer-based matchers present severe quantization hurdles due to the **"Attention Paradox"**—where heavy-tailed dynamic attention outlier activations stretch uniform 8-bit quantization grids, causing an accuracy collapse in standard Static QDQ.

In this module, we:
1. **Decouple the Transformer Architecture**: Export PyTorch LighterGlue to ONNX FP32 with dynamic keypoint axes.
2. **Resolve the Attention Paradox**: Contrast **Dynamic PTQ** with **SmoothQuant ($\alpha = 0.5$)**, migrating activation outliers into static weights.
3. **Validate Robustness Across 5 Degradations**: Stress-test models under Clean, Noise, Brightness, Blur, and Occlusion.
4. **Benchmark on HPatches & Compute SPE**: Measure sub-pixel localization accuracy ($\text{AUC@1px}$) against classical and deep baselines using the **SLAM Perception Efficiency (SPE)** Figure of Merit.

---

## Key Benchmark Results

### 1. Multi-Pair Robustness Benchmark (20 Test Pairs, 1-Core ARM Simulation)

The following table reflects the exact experimental logs recorded in [`ptq_onnx_lighterGlue.ipynb`](ptq_onnx_lighterGlue.ipynb) (Cell 26):

| Model Variant | Disk Size | Latency (1-Thread Sim) | Average Success Rate | MAE Numerical Drift |
| :--- | :---: | :---: | :---: | :---: |
| **FP32 (Base Reference)** | `5.29 MB` *(total)* / `1.18 MB` | `~508 ms` | `53.1%` | `0.000` *(Baseline)* |
| **INT8 (Dynamic)** | `2.16 MB` | **`~455 ms`** | `51.1%` | **`0.320`** *(Lowest drift)* |
| **INT8 (Static QDQ)** | `2.32 MB` | `~520 ms` | `44.4%` | `0.520` *(Activation clipping)* |
| **INT8 (SmoothQuant)** | `2.37 MB` | `~522 ms` | `49.0%` | `0.560` *(Noise resilient)* |

---

### 2. Standardized HPatches Benchmark (Cell 41 & Cell 46)

Evaluated on 10 HPatches sequences (5 illumination `i_*` + 5 viewpoint `v_*`) capped at 700 features:

| Method-Architecture | AUC@1px (%) | AUC@3px (%) | AUC@5px (%) | Time (ms) | SPE Score (↑) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FAST + BRIEF + BF** | `15.6449` | `35.8120` | `47.9231` | `5.53` | **`283.1446`** |
| **FAST + ORB + BF** | `18.4862` | `42.1150` | `55.6200` | `17.38` | `106.3926` |
| **SuperPoint + BF** | `22.6583` | `51.4820` | `66.8340` | `1603.97` | `1.4126` |
| **XFeat + BF** | `18.8501` | `44.9200` | `59.7300` | `1321.70` | `1.4262` |
| **XFeat + LighterGlue INT8** | **`24.5014`** | **`56.3400`** | **`72.1800`** | `1478.73` | `1.6569` |

$$\text{SPE} = \frac{\text{AUC@1px}}{\text{Total Latency (ms)}} \times 100$$

> [!NOTE]
> Classical binary matchers (FAST+BRIEF) have high raw FPS but inadequate sub-pixel precision for complex visual odometry. **XFeat + LighterGlue delivers the highest localization accuracy (24.50% AUC@1px)**, outperforming all classical and deep matching combinations.

---

## Architectural & Theoretical Insights

### 1. The "Attention Paradox" & Quantization Strategy

```text
The Transformer Quantization Landscape
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Input Keypoints & Descriptors                                                  │
│         │                                                                       │
│         ▼                                                                       │
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │  Multi-Head Cross/Self-Attention Layers (LighterGlue)                       │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
│         │                                                                       │
│         ├─── 1. Static QDQ (Collapsed) ──> Outlier spikes crush INT8 step Δ     │
│         │                                                                       │
│         ├─── 2. Dynamic INT8 (Fastest)  ──> On-the-fly scales catch spikes      │
│         │                                                                       │
│         └─── 3. SmoothQuant (Robust)    ──> Migrates activation spikes to weights│
└─────────────────────────────────────────────────────────────────────────────────┘
```

* **The Problem:** In cross-attention layers, dot-product activations $\text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)$ generate sparse, high-magnitude outlier spikes. In standard Static QDQ, these spikes stretch the 8-bit dynamic range ($\Delta = \frac{\max - \min}{255}$), compressing the remaining 99% of fine-grained matching weights into 2–3 discrete integer levels and dropping success from 53.1% to 44.4%.
* **The Dynamic Solution:** Dynamic INT8 recalculates scale factors per inference step, capturing spikes as they occur and delivering the fastest execution (**~455 ms**).
* **The SmoothQuant Solution:** Applies an equivalence transformation that scales activation outliers by per-channel migration factors $s_j$ and absorbs them into the weight matrix:

$$
\boxed{\displaystyle W' = \text{diag}(s) \cdot W, \quad X' = X \cdot \text{diag}(s)^{-1}, \quad s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}}
$$

Using $\alpha = 0.5$, SmoothQuant splits the difficulty evenly between activations and weights, recovering **49.0% matching success** without runtime scale-calculation overhead.

---

### 2. The Noise Regularization Phenomenon
Under severe sensor noise stress testing (Cell 26):
* **FP32 Baseline:** `34.1%` success rate.
* **SmoothQuant INT8:** **`35.8%` success rate**.
* **Insight:** The discrete 8-bit quantization grid acts as a subtle high-frequency denoising filter, suppressing low-amplitude background noise before attention score computation.

---

### 3. Engineering Post-Mortem: Hurdle Resolutions (Cell 28)

1. **Symbolic Shape `IndexError` Resolution:**
   * *The Issue:* During `quant_pre_process`, ONNX Runtime's symbolic shape inferrer crashed with `IndexError: index 4 is out of bounds for axis 0`.
   * *The Fix:* Implemented an automated fallback wrapper with `skip_symbolic_shape=True`, bypassing unstable rank trackers on dynamic attention reshaping.
2. **Operator Targeting (`MatMul` vs. `Gemm`):**
   * PyTorch Linear layers export as both `MatMul` and `Gemm`. Quantization explicitly targeted `['MatMul', 'Gemm', 'Softmax']` to convert the entire attention mechanism to 8-bit arithmetic.
3. **Apple Neural Engine (ANE) / Core ML Integration (Cell 43):**
   * Clamped keypoint extraction buffers to a static 1024-point dimension to enable deterministic memory pre-allocation on Apple Silicon Neural Engine hardware.

---

## Quickstart & Usage

### 1. Automated Setup
```bash
bash setup.sh
```

### 2. Run the Interactive Jupyter Walkthrough
```bash
jupyter notebook ptq_onnx_lighterGlue.ipynb
```

### 3. Run via Standalone CLI Pipeline
```bash
# 1. Export PyTorch model to ONNX FP32
python3 scripts/01_export_fp32.py

# 2. Apply Dynamic INT8 Quantization
python3 scripts/02_quantize_dynamic.py

# 3. Apply SmoothQuant Calibration
python3 scripts/03_quantize_smooth.py

# 4. Run Multi-Pair Drift & Robustness Benchmark
python3 scripts/04_benchmark_drift.py

# 5. Run HPatches Benchmark
python3 scripts/05_benchmark_hpatches.py

# 6. Calculate SLAM Perception Efficiency (SPE)
python3 scripts/06_figure_of_merit.py
```

---

## 📂 Directory Layout

```text
tutorials/lighterglue-ptq/
├── README.md                      # Technical report & benchmark comparisons
├── LICENSE                        # Apache 2.0 License
├── .gitignore                     # Ignores weights and caches
├── requirements.txt               # Module dependencies
├── setup.sh                       # Dependency & dataset bootstrap
├── ptq_onnx_lighterGlue.ipynb     # Interactive tutorial notebook
├── scripts/
│   ├── common.py                  # Evaluation, calibration & AUC calculators
│   ├── 01_export_fp32.py          # PyTorch -> ONNX FP32
│   ├── 02_quantize_dynamic.py     # Dynamic INT8 Quantization
│   ├── 03_quantize_smooth.py      # SmoothQuant Calibration
│   ├── 04_benchmark_drift.py      # MAE drift across 5 conditions
│   ├── 05_benchmark_hpatches.py   # HPatches MMA & AUC benchmark
│   └── 06_figure_of_merit.py      # SLAM Perception Efficiency (SPE)
└── output/                        # Pre-exported ONNX models (*.onnx)
```

---

## 📖 References & Acknowledgments

* **LighterGlue**: Lindenberger, P., Sarlin, P. E., & Pollefeys, M. (2023). *LightGlue: Local Feature Matching at Light Speed*. ICCV 2023. [[Paper](https://arxiv.org/abs/2306.13643)] [[GitHub](https://github.com/cvg/LightGlue)]
* **XFeat**: Potje, G., Dahl, A. L., Cavalheiro, G. G., & Christiansen, C. (2024). *Accelerated Features for Lightweight Image Matching*. CVPR 2024. [[Paper](https://arxiv.org/abs/2404.19174)] [[GitHub](https://github.com/verlab/accelerated_features)]
* **SmoothQuant**: Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., & Han, S. (2023). *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*. ICML 2023. [[Paper](https://arxiv.org/abs/2211.10438)]
* **HPatches**: Balntas, V., Lenc, K., Vedaldi, A., & Mikolajczyk, K. (2017). *HPatches: A benchmark and evaluation protocol for local feature descriptors*. CVPR 2017. [[Paper](https://arxiv.org/abs/1704.05939)]
* **ONNX Runtime**: Microsoft. *High-performance cross-platform inferencing and training accelerator*. [[Documentation](https://onnxruntime.ai/)]

---

## 📜 License
This deployment tutorial and benchmark suite is licensed under the [Apache License 2.0](LICENSE) in alignment with upstream LightGlue and XFeat licensing.
