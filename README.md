# 🌿 GreenScan — Drone-Based Precision Farm Analysis

<div align="center">

![GreenScan Banner](https://img.shields.io/badge/GreenScan-Precision%20Farming-2d6a2d?style=for-the-badge&logo=leaf&logoColor=7FFF00)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![No ML](https://img.shields.io/badge/ML%20Models-None%20(Pure%20Math)-yellow?style=for-the-badge)

**AI-free, pure-math drone image analysis for smart farming decisions.**  
Upload multispectral drone imagery → Get actionable field health maps → Estimate treatment costs.

[🚀 Live Demo](#) · [📖 Docs](#how-it-works) · [🐛 Report Bug](../../issues) · [💡 Request Feature](../../issues)

</div>



## 🌾 What is GreenScan?

**GreenScan** is a web-based drone imagery analysis tool built for precision agriculture. It takes multispectral images captured by agricultural drones and breaks them into thousands of micro-cells, computing vegetation health indices for each cell — then tells the farmer **exactly where** to reseed, irrigate, fertilize, or treat for pests.

> **No machine learning. No cloud APIs. No black box.**  
> Just pure spectral math (NDVI, NDRE, LCI, GNDVI) running locally in your browser via Streamlit.

---

## ✨ Features

### 📤 Upload & Configure
- Upload **5 spectral bands** separately — RGB, Red, Green, NIR, RedEdge
- **Missing bands?** GreenScan auto-fills with smart synthetic defaults so you still get usable results
- Configure **camera & flight parameters** — sensor width, focal length, flight height, image resolution
- **GSD (Ground Sampling Distance)** auto-calculated live — know exactly how many cm² each pixel covers
- Adjustable **grid size** (up to 256×192 = ~49,000 micro-cells)
- Advanced **threshold sliders** per action type (Reseeding, Water, Urea, Stress, Growth)
- **Field notes** section for manual observations

### 📊 Results Dashboard
- **5 dedicated action tabs** — click any action to see its full map
  - 🟤 **Reseeding** — bare soil / failed germination zones
  - 💧 **Water** — moisture stress areas
  - 🟡 **Urea** — nitrogen deficiency zones
  - 🟣 **Stress** — pest / disease pressure
  - 🟢 **Growth** — stunted growth areas
- Each tab shows: **color-coded overlay map** + **severity bar chart** + **affected area in sqm**
- 🌡️ **NDVI Heatmap** — full field vegetation density view
- 📈 **Overview donut chart** + complete statistics table

### 💰 Cost Analysis
- Input your **local treatment rates** (₹ per sqm) for each action
- Specify **labor cost** and **overhead percentage**
- Control **coverage sliders** — what % of critical/moderate/low cells to treat
- Get **total estimated treatment cost** with **priority-ordered action list**
- All powered by **GSD-derived real-world area** — not guesswork

### 📥 Export & Reports
- **CSV** — raw data for all cells (indices + action flags)
- **PDF Report** — all maps, statistics, cost breakdown, and field notes in one document
- **JSON** — machine-readable summary for integrations

---

## 🔬 How It Works

GreenScan processes drone imagery in 4 steps:

```
Drone Images (RGB + 4 Bands)
         │
         ▼
  ┌─────────────────┐
  │  Grid Division  │  → Field split into N×M micro-cells
  └────────┬────────┘
           │
           ▼
  ┌─────────────────────────────────────────┐
  │     Vegetation Index Calculation        │
  │                                         │
  │  NDVI  = (NIR - Red) / (NIR + Red)     │  ← Crop health
  │  NDRE  = (NIR - RE)  / (NIR + RE)      │  ← Nitrogen level
  │  LCI   = (NIR - RE)  / (NIR + Red)     │  ← Crop stress
  │  GNDVI = (NIR - Green)/(NIR + Green)   │  ← Water content
  └────────┬────────────────────────────────┘
           │
           ▼
  ┌──────────────────────────┐
  │  Threshold Classification│  → Each cell flagged:
  │  Critical / Moderate / Low│   Critical=3, Moderate=2, Low=1
  └────────┬─────────────────┘
           │
           ▼
  ┌──────────────────────────┐
  │  GSD × Cells = Real Area │  → Cost estimation in ₹
  └──────────────────────────┘
```

### GSD Formula
```
GSD (m/px) = (Sensor Width (m) × Flight Height (m)) / (Focal Length (m) × Image Width (px))

Cell Area (sqm) = GSD_width × GSD_height × (Image_px / Grid_cells)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/GreenScan.git
cd GreenScan

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📁 Project Structure

```
GreenScan/
│
├── app.py                  # Main Streamlit application (single file)
├── requirements.txt        # Python dependencies
├── config.json             # Default threshold & camera configuration
├── README.md               # This file
│
├── Sample/                 # (Optional) Sample drone images for testing
    ├── Drone_Image_001.jpg
    ├── Drone_Image_001_MS_R.TIF
    ├── Drone_Image_001_MS_G.TIF
    ├── Drone_Image_001_MS_NIR.TIF
    └── Drone_Image_001_MS_RE.TIF

```

---

## ⚙️ Configuration (`config.json`)

```json
{
  "camera": {
    "flight_height_meters": 50,
    "sensor_width_mm": 6.17,
    "focal_length_mm": 4.5,
    "image_width_px": 4000,
    "image_height_px": 3000
  },
  "thresholds": {
    "RESEEDING": { "critical": 0.16, "moderate": 0.22, "low": 0.30 },
    "WATER":     { "critical": 0.45, "moderate": 0.55, "low": 0.65 },
    "UREA":      { "critical": 0.15, "moderate": 0.22, "low": 0.30 },
    "STRESS":    { "critical": 0.10, "moderate": 0.20, "low": 0.30 },
    "GROWTH":    { "critical": 0.35, "moderate": 0.45, "low": 0.55 }
  },
  "costs": {
    "RESEEDING_PER_SQM": 5.0,
    "WATER_PER_SQM": 2.0,
    "UREA_PER_SQM": 3.0,
    "PESTICIDE_PER_SQM": 4.0
  }
}
```

All thresholds can also be adjusted live from the **Advanced Settings** panel in the UI.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `numpy` | Array math & index calculations |
| `opencv-python-headless` | Image loading, resizing, overlay rendering |
| `tifffile` | Reading multispectral `.TIF` band images |
| `matplotlib` | Maps, heatmaps, charts, PDF export |
| `pandas` | Data tables & CSV export |
| `Pillow` | Image processing utilities |

> **No scikit-learn. No TensorFlow. No PyTorch. No external AI APIs.**

---

## 🌱 Supported Drone & Camera Types

GreenScan works with any multispectral drone setup where you can export individual band images:

| Drone / Sensor | Compatibility |
|---|---|
| DJI Phantom 4 Multispectral | ✅ Full support |
| DJI Mavic 3 Multispectral | ✅ Full support |
| Parrot Sequoia | ✅ Full support |
| MicaSense RedEdge | ✅ Full support |
| Any drone with TIF band export | ✅ Compatible |
| RGB-only drone | ⚠️ Partial (limited indices) |

---

## 📊 Vegetation Indices Reference

| Index | Formula | What It Tells You |
|---|---|---|
| **NDVI** | `(NIR - Red) / (NIR + Red)` | Overall crop health & biomass |
| **NDRE** | `(NIR - RedEdge) / (NIR + RedEdge)` | Nitrogen / chlorophyll content |
| **LCI** | `(NIR - RedEdge) / (NIR + Red)` | Crop stress & disease |
| **GNDVI** | `(NIR - Green) / (NIR + Green)` | Water content & canopy density |

---

## 🤝 Contributing

Contributions are welcome! Here's how:

```bash
# 1. Fork the repo
# 2. Create your feature branch
git checkout -b feature/AmazingFeature

# 3. Commit your changes
git commit -m 'Add some AmazingFeature'

# 4. Push to the branch
git push origin feature/AmazingFeature

# 5. Open a Pull Request
```

### Ideas for Contribution
- [ ] Time-series comparison (upload 2 dates, compare field changes)
- [ ] GPS/geo-tagging support for map overlay
- [ ] Multi-language support (Hindi, Punjabi, etc.)
- [ ] Mobile-responsive UI improvements
- [ ] Batch processing multiple fields

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙏 Acknowledgements

- Vegetation index formulas based on established remote sensing research
- UI built with [Streamlit](https://streamlit.io)
- Inspired by real precision agriculture challenges faced by Indian farmers

---

<div align="center">

**Made with 💚 for Indian Farmers**

*GreenScan — See your field, save your crop.*

</div>
