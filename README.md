#BTS Finder: Cellular Base Station Localization & Propagation Modeling Demo

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Functional Demo](https://img.shields.io/badge/Status-Reproducible%20Demo-brightgreen.svg)]()
[![Citation](https://img.shields.io/badge/Cite-CITATION.cff-orange.svg)](CITATION.cff)

A standalone, reproducible research and demonstration repository for **reverse-engineering physical LTE base transceiver station (eNodeB) coordinates** from crowdsourced smartphone drive-test logs.

Developed as a worked example for academic citation, thesis benchmarking, and empirical cellular network evaluation.

---

## 📌 Research Problem & Overview

Commercial cellular operators (Digi/RCS&RDS, Orange, Vodafone, Telekom) do not publicly publish the exact geographical coordinates, antenna azimuths, or tower locations of their cellular infrastructure.

This repository provides an autonomous mathematical pipeline that takes crowd-sourced RF drive-test measurements collected via the Android **Network Survey** app (GPS coordinates, Reference Signal Received Power $RSRP$, and Timing Advance $TA$) and reconstructs:
1. **Physical eNodeB Tower Coordinates** $(\text{Latitude}, \text{Longitude})$ using non-linear robust multilateration with Huber loss.
2. **Empirical Radio Propagation Parameters** ($PL_0$ and path-loss exponent $n$) calibrated in the linear power domain.
3. **Antenna Beam Coverage Polygons** (Convex Hull) and estimated boresight azimuth angles for each sector.
4. **Interactive GIS Cartography** and publication-ready diagnostic charts.

---

## 🚀 Key Features

* **Universal Network Survey Ingestion**: Directly ingests raw CSV logs exported from the Android [Network Survey (Craxiom)](https://github.com/christianrowlands/android-network-survey) application (automatically handles comment lines `# Created by Network Survey` and derives $\text{eNodeB ID} = \lfloor \text{ECI} / 256 \rfloor$).
* **User Custom Data Support**: Place your own survey CSV into `data/user_input/` or pass `--input path/to/file.csv` to instantly analyze your own drive tests.
* **No Database Required**: Decoupled from heavy PostgreSQL / PostGIS infrastructure — runs out-of-the-box using standard scientific Python (`numpy`, `scipy`, `pandas`, `matplotlib`, `pyproj`, `folium`).
* **Slant-Range Antenna Correction**: Compensates for tower height differentials ($h_{\text{diff}} \approx 35.5\text{ m}$) to prevent severe near-field estimation bias.
* **Anti-Clustering Weighting**: Prevents vehicle stops at traffic lights from skewing the spatial optimization via inverse-density cell weights:
  $$w_{\text{density}} = \max\left(0.2, \frac{1}{\text{count}^{0.3}}\right)$$
* **Physical Timing Advance Boundary**: Asymmetric cost ensuring actual distance satisfies Line-of-Sight propagation physics ($d \ge d_{\text{radio}}$ with building shadow tolerance).
* **Multi-Start Optimization**: Avoids local minima using power-weighted centroids, peak RSRP seeds, and orthogonal PCA projections.

---

## 📊 Included Worked Example Dataset

The repository includes a curated sample dataset (`data/sample_lte_measurements.csv`, 1,870 real measurements from Timișoara, Romania) covering **9 representative eNodeBs**:

| eNodeB ID | Operator / Equipment Typology | Raw Records | TA Valid Records | Mean RSRP |
| :--- | :--- | :---: | :---: | :---: |
| **80609** | Ericsson Macro (Orange) | 648 | 611 | -84.6 dBm |
| **101811** | Nokia Macro (Digi Mobil) | 391 | 362 | -95.7 dBm |
| **240151** | Nokia Macro (Digi Mobil) | 290 | 270 | -99.6 dBm |
| **200643** | Nokia Macro (Digi Mobil) | 227 | 183 | -103.6 dBm |
| **950063** | Micro Cell Cluster (Digi Mobil) | 227 | 203 | -91.1 dBm |
| **82039** | Ericsson Macro (Orange) | 40 | 39 | -89.8 dBm |
| **201813** | Nokia Macro (Digi Mobil) | 32 | 28 | -93.0 dBm |
| **950064** | Micro Cell Cluster (Digi Mobil) | 13 | 13 | -84.1 dBm |
| **950062** | Micro Cell Cluster (Digi Mobil) | 2 | 2 | -80.5 dBm |

---

## ⚙️ Mathematical Formulation

### 1. Timing Advance Ground Distance & Slant Correction
In 3GPP LTE, Timing Advance ($TA \in [0, 63]$) represents propagation delay in units of $16 \times T_s \approx 78.125\text{ m}$.
Accounting for tower height $h_{\text{mast}} = 37\text{ m}$ and mobile receiver $h_{\text{ue}} = 1.5\text{ m}$:
$$d_{\text{ground}} = \sqrt{\max\left(0, (TA \times 78.125)^2 - (37.0 - 1.5)^2\right)}$$

### 2. Log-Distance Path Loss Propagation
When Timing Advance is unavailable or for signal calibration:
$$PL(d) = PL_0 + 10 \cdot n \cdot \log_{10}(d) + X_\sigma$$
$$d_{\text{est}}(RSRP) = 10^{\frac{-RSRP - PL_0}{10 \cdot n}}$$
Parameters $PL_0$ and exponent $n$ are calibrated via Robust Iteratively Reweighted Least Squares with Median Absolute Deviation (MAD) residual weighting.

### 3. Huber Loss Multilateration
The Cartesian tower position $(x_0, y_0)$ in metric UTM coordinates minimizes:
$$\min_{x_0, y_0} \sum_{i=1}^N w_i \cdot L_\delta\left(\sqrt{(x_0 - x_i)^2 + (y_0 - y_i)^2} - d_{\text{target}, i}\right) + \text{Cost}_{\text{physics}} + \text{Cost}_{\text{repulsion}}$$
where $L_\delta(r)$ is the Huber loss:
$$L_\delta(r) = \begin{cases} \frac{1}{2} r^2 & \text{for } |r| \le \delta \\ \delta \left(|r| - \frac{1}{2}\delta\right) & \text{for } |r| > \delta \end{cases}$$

---

## 🛠️ Quickstart Guide

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-username/nokia-bts-localization-demo.git
cd nokia-bts-localization-demo
pip install -r requirements.txt
```

### 2. Run the Worked Example (One-Click)
```bash
# Processes the sample dataset across all 9 target eNodeBs:
python run_demo.py
```
Execution takes **~15-20 seconds** and generates all figures in `output/`.

### 3. Target a Specific eNodeB
```bash
python run_demo.py --enodeb 80609
```

### 4. Analyze Your Own Data from Network Survey
Drop your exported CSV file from the **Network Survey** app into `data/user_input/` or run:
```bash
python run_demo.py --input path/to/craxiom-lte-survey.csv
```

---

## 📈 Generated Artifacts & Visualizations

Running `run_demo.py` produces the following files in `output/`:

1. **`fig1_bts_localization_<eNodeB>.png`**: Multi-lateration map with drive-test track, RSRP colorbar, concentric Timing Advance circular distance rings, initial RF centroid seed, and final estimated eNodeB location with confidence metrics.
2. **`fig2_path_loss_model_<eNodeB>.png`**: Empirical radio propagation calibration curve ($RSRP$ vs. distance), free-space comparison line ($n = 2.0$), and shadow-fading error distribution histogram.
3. **`fig3_sector_coverage_<eNodeB>.png`**: Multi-sector Convex Hull coverage polygons color-coded by Physical Cell ID (PCI) and estimated main antenna azimuth arrows.
4. **`interactive_map.html`**: Self-contained interactive Leaflet/Folium web map featuring layer controls, interactive measurement tooltips, tower markers, and sector bounds.
5. **`metrics_summary.json`**: Structured JSON report with estimated coordinates, solver RMSE, confidence grades (A–D), and calibrated path-loss coefficients.

---

## 📂 Repository Structure

```
nokia-bts-localization-demo/
│
├── .gitignore
├── LICENSE                          # MIT Open-Source License
├── CITATION.cff                     # Citation File Format for GitHub / Zenodo
├── requirements.txt                 # Minimal dependencies (numpy, scipy, pandas, matplotlib, pyproj, folium)
├── README.md                        # Complete project documentation and quickstart
├── run_demo.py                      # Master command-line runner script
├── worked_example.ipynb             # Step-by-step Jupyter Notebook with explanations
│
├── data/
│   ├── sample_lte_measurements.csv  # 1,870 drive-test records for the 9 specified eNodeBs
│   ├── ground_truth_reference.json  # Reference benchmark localizations
│   └── user_input/                  # Drop-in folder for custom user Network Survey CSV files
│
├── src/                             # Core modular algorithms
│   ├── __init__.py
│   ├── loader.py                    # Universal loader: Network Survey (# headers, ECI->eNodeB) & standard CSVs
│   ├── preprocessing.py             # UTM projection, linear power domain averaging, spatial deduplication
│   ├── path_loss.py                 # Robust IRLS Log-Distance Path Loss fitting
│   ├── solver.py                    # Multi-start Huber loss multilateration solver
│   ├── sectors.py                   # Sector grouping, Convex Hull polygons, and azimuth estimation
│   └── visualization.py             # Matplotlib figures and Folium interactive map builder
│
└── output/                          # Output directory for generated PNG figures and HTML map
```

---

## 📝 How to Cite

If you use this code, methodology, or empirical dataset in an academic paper, graduation thesis, or technical report, please cite it using:

### BibTeX
```bibtex
@software{nokia_bts_finder_2026,
  author = {Nokia BTS Research Project},
  title = {Nokia BTS Finder: Robust Multilateration and Path Loss Modeling from Crowdsourced Cellular Drive-Test Measurements},
  year = {2026},
  url = {https://github.com/your-username/nokia-bts-localization-demo},
  version = {1.0.0}
}
```

Or click the native **"Cite this repository"** button on GitHub using the provided `CITATION.cff`.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - feel free to use, modify, and integrate it in academic and commercial projects.
