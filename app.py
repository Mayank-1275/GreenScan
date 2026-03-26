"""
╔══════════════════════════════════════════════════════════════╗
║          🌾 GreenScan — Drone Farm Analysis System       ║
║          Streamlit Web App | Pure Python + Math Only         ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import cv2
import os
import io
import json
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be FIRST streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GreenScan 🌾",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CONSTANTS & DEFAULTS
# ─────────────────────────────────────────────
DEFAULT_GRID_COLS = 256
DEFAULT_GRID_ROWS = 192

# ─────────────────────────────────────────────
# UPDATED SCIENTIFIC THRESHOLDS (RECOMMENDED)
# ─────────────────────────────────────────────
DEFAULT_THRESHOLDS = {
    "RESEEDING": {"critical": 0.12, "moderate": 0.18, "low": 0.25},
    "WATER":     {"critical": 0.30, "moderate": 0.40, "low": 0.50},
    "UREA":      {"critical": 0.15, "moderate": 0.25, "low": 0.35},
    "STRESS":    {"critical": 0.10, "moderate": 0.20, "low": 0.30},
    "GROWTH":    {"critical": 0.30, "moderate": 0.40, "low": 0.50},
}

ACTION_META = {
    "RESEEDING": {"icon": "🟤", "color": "#A0522D", "desc": "Bare soil / failed germination — seeds need replanting"},
    "WATER":     {"icon": "💧", "color": "#1E90FF", "desc": "Moisture stress — irrigation required"},
    "UREA":      {"icon": "🟡", "color": "#DAA520", "desc": "Nitrogen deficiency — fertilizer needed"},
    "STRESS":    {"icon": "🟣", "color": "#8A2BE2", "desc": "Crop stress / pest or disease pressure"},
    "GROWTH":    {"icon": "🟢", "color": "#228B22", "desc": "Stunted growth — general intervention needed"},
}

SEVERITY_COLORS_MPL = {3.0: "#FF2222", 2.0: "#FF8C00", 1.0: "#FFD700"}

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Nunito:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Rajdhani', sans-serif !important;
        letter-spacing: 0.03em;
    }

    /* Top header bar */
    .kd-hero {
        background: linear-gradient(135deg, #1a3c1a 0%, #2d6a2d 50%, #1a4a2a 100%);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 24px;
        border: 1px solid #3a8a3a;
        box-shadow: 0 4px 24px rgba(0,150,0,0.15);
    }
    .kd-hero h1 {
        color: #7FFF00;
        font-size: 2.6rem;
        margin: 0;
        text-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }
    .kd-hero p {
        color: #b8f0b8;
        font-size: 1.05rem;
        margin: 4px 0 0 0;
    }

    /* Metric cards */
    .kd-metric {
        background: linear-gradient(135deg, #1e2e1e, #253525);
        border: 1px solid #3a5a3a;
        border-radius: 12px;
        padding: 14px 18px;
        text-align: center;
    }
    .kd-metric .label {
        font-size: 0.78rem;
        color: #88aa88;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .kd-metric .value {
        font-size: 1.7rem;
        font-weight: 700;
        font-family: 'Rajdhani', sans-serif;
        color: #7FFF00;
        line-height: 1.1;
    }
    .kd-metric .sub {
        font-size: 0.82rem;
        color: #aacaaa;
    }

    /* Band upload card */
    .band-card {
        background: #182818;
        border: 2px dashed #3a5a3a;
        border-radius: 12px;
        padding: 18px 12px;
        text-align: center;
        min-height: 80px;
        margin-bottom: 6px;
    }
    .band-card .band-name {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: #b0d0b0;
    }
    .band-card .band-status {
        font-size: 0.8rem;
        margin-top: 4px;
    }
    .status-ok    { color: #7FFF00; }
    .status-warn  { color: #FFD700; }
    .status-error { color: #FF4444; }

    /* Section headers */
    .section-header {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #7FFF00;
        border-left: 4px solid #7FFF00;
        padding-left: 12px;
        margin: 18px 0 10px 0;
    }

    /* Summary box */
    .kd-summary {
        background: #182818;
        border: 1px solid #3a5a3a;
        border-radius: 10px;
        padding: 16px 20px;
        font-size: 0.92rem;
        color: #c0dcc0;
        line-height: 1.7;
    }

    /* Cost table highlight */
    .cost-total {
        background: #1a3a1a;
        border: 1px solid #7FFF00;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        font-family: 'Rajdhani', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #7FFF00;
    }

    /* Progress bar override */
    .stProgress > div > div {
        background-color: #7FFF00 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #0d1f0d;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #88aa88;
        padding: 6px 16px;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #2d6a2d !important;
        color: #7FFF00 !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #2d6a2d, #1a4a1a);
        color: #7FFF00;
        border: 1px solid #7FFF00;
        border-radius: 10px;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        padding: 10px 24px;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3d8a3d, #2a6a2a);
        box-shadow: 0 0 16px rgba(127,255,0,0.3);
        transform: translateY(-1px);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0d1f0d;
        border-right: 1px solid #2a4a2a;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2 {
        color: #7FFF00;
    }

    /* Streamlit default overrides */
    .stApp { background-color: #0a150a; }
    .main .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetricValue"] { color: #7FFF00 !important; }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "farm_map": None,
        "rgb_img": None,
        "stats": None,
        "total_cells": None,
        "health_score": None,
        "cell_area": None,
        "grid_rows": DEFAULT_GRID_ROWS,
        "grid_cols": DEFAULT_GRID_COLS,
        "thresholds": DEFAULT_THRESHOLDS.copy(),
        "camera_config": {},
        "cost_results": {},
        "field_notes": "",
        "analysis_done": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────
# MATH — INDEX CALCULATIONS
# ─────────────────────────────────────────────
def _safe_div(a, b):
    """
    Standard division with zero-check. 
    Purane code mein (a-b)/b ho raha tha, jo galat tha.
    Ab yeh sirf a/b karega.
    """
    d = b.copy().astype(float)
    d[d == 0] = 1e-9  # Zero division error se bachne ke liye
    return a.astype(float) / d

def calc_ndvi(nir, red):
    # Formula: (NIR - Red) / (NIR + Red)
    return _safe_div(nir - red, nir + red)

def calc_ndre(nir, re):
    # Formula: (NIR - RedEdge) / (NIR + RedEdge)
    return _safe_div(nir - re, nir + re)

def calc_lci(nir, re, red):
    # Formula: (NIR - RedEdge) / (NIR + Red)
    return _safe_div(nir - re, nir + red)

def calc_gndvi(nir, green):
    # Formula: (NIR - Green) / (NIR + Green)
    return _safe_div(nir - green, nir + green)

def calc_gsd(fh, sw, fl, iw, ih):
    """GSD in meters/pixel (width, height)"""
    gsd_w = (sw / 1000.0 * fh) / (fl / 1000.0 * iw)
    gsd_h = (sw / 1000.0 * fh) / (fl / 1000.0 * ih)
    return gsd_w, gsd_h

def calc_cell_area(fh, sw, fl, iw, ih, grid_r, grid_c):
    gsd_w, gsd_h = calc_gsd(fh, sw, fl, iw, ih)
    cell_w_m = gsd_w * (iw / grid_c)
    cell_h_m = gsd_h * (ih / grid_r)
    return cell_w_m * cell_h_m


# ─────────────────────────────────────────────
# IMAGE LOADING HELPERS
# ─────────────────────────────────────────────
def load_rgb(uploaded_file):
    data = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img  # BGR

def load_band(uploaded_file, target_h, target_w, default_val=0.5):
    """Load spectral band, normalize to 0.0-1.0, or return synthetic default."""
    if uploaded_file is None:
        # Default value ko 0.0-1.0 ki range mein hi rakhein
        base = np.full((target_h, target_w), default_val, dtype=np.float64)
        noise = np.random.normal(0, 0.04, (target_h, target_w))
        return np.clip(base + noise, 0, 1.0), True 

    fname = uploaded_file.name.lower()
    raw = uploaded_file.read()

    if fname.endswith((".tif", ".tiff")):
        try:
            import tifffile
            arr = tifffile.imread(io.BytesIO(raw)).astype(np.float64)
        except Exception:
            nparr = np.frombuffer(raw, np.uint8)
            arr = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    else:
        nparr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        arr = img.astype(np.float64)

    # 1. Flatten to 2D if multi-channel
    if arr.ndim == 3:
        arr = arr[:, :, 0]

    # 2. Resize if needed
    if arr.shape != (target_h, target_w):
        arr = cv2.resize(arr.astype(np.float32), (target_w, target_h),
                         interpolation=cv2.INTER_LINEAR).astype(np.float64)

    # 3. CRITICAL: Normalization Step (Yahan paste karein)
    max_val = np.max(arr)
    if max_val > 0:
        arr = arr / max_val # Sabhi pixels ko 0.0-1.0 range mein le aata hai

    return arr, False


# ─────────────────────────────────────────────
# ANALYSIS ENGINE
# ─────────────────────────────────────────────
def init_farm_map(rows, cols):
    fm = {}
    for r in range(rows):
        for c in range(cols):
            fm[(r, c)] = {
                "NDVI": 0.0, "NDRE": 0.0, "LCI": 0.0, "GNDVI": 0.0,
                "RESEEDING": 0.0, "WATER": 0.0, "UREA": 0.0,
                "STRESS": 0.0, "GROWTH": 0.0, "HEALTHY": 0.0,
            }
    return fm

def fill_indices(fm, band_r, band_g, band_nir, band_re, rows, cols, prog=None):
    ndvi_map  = calc_ndvi(band_nir, band_r)
    ndre_map  = calc_ndre(band_nir, band_re)
    lci_map   = calc_lci(band_nir, band_re, band_r)
    gndvi_map = calc_gndvi(band_nir, band_g)

    h, w = band_r.shape
    ch = h // rows
    cw = w // cols

    for r in range(rows):
        for c in range(cols):
            y1, y2 = r * ch, (r + 1) * ch
            x1, x2 = c * cw, (c + 1) * cw
            fm[(r, c)]["NDVI"]  = round(float(np.nanmean(ndvi_map[y1:y2, x1:x2])), 4)
            fm[(r, c)]["NDRE"]  = round(float(np.nanmean(ndre_map[y1:y2, x1:x2])), 4)
            fm[(r, c)]["LCI"]   = round(float(np.nanmean(lci_map[y1:y2, x1:x2])), 4)
            fm[(r, c)]["GNDVI"] = round(float(np.nanmean(gndvi_map[y1:y2, x1:x2])), 4)
        if prog:
            prog.progress(int((r + 1) / rows * 60),
                          text=f"⚡ Computing indices... row {r+1}/{rows}")
    return fm

def _level(val, t):
    if val < t["critical"]: return 3.0
    if val < t["moderate"]: return 2.0
    if val < t["low"]:      return 1.0
    return 0.0

def apply_flags(fm, thresholds, prog=None):
    total = len(fm)
    for i, (key, d) in enumerate(fm.items()):
        fr = _level(d["NDVI"], thresholds["RESEEDING"])
        if fr > 0:
            fw = fu = fs = fg = 0.0
        else:
            fw = _level(d["NDVI"],  thresholds["WATER"])
            fu = _level(d["NDRE"],  thresholds["UREA"])
            fs = _level(d["LCI"],   thresholds["STRESS"])
            fg = _level(d["GNDVI"], thresholds["GROWTH"])

        fh = 1.0 if (fr + fw + fu + fs + fg) == 0 else 0.0
        d.update({"RESEEDING": fr, "WATER": fw, "UREA": fu,
                  "STRESS": fs,  "GROWTH": fg, "HEALTHY": fh})
        if prog and i % 8000 == 0:
            prog.progress(60 + int(i / total * 38),
                          text=f"🏷️  Classifying cells... {i:,}/{total:,}")
    return fm

def build_stats(fm):
    total = len(fm)
    stats = {a: {"critical": 0, "moderate": 0, "low": 0, "total": 0}
             for a in ACTION_META}
    stats["HEALTHY"] = 0
    for d in fm.values():
        for a in ACTION_META:
            lv = d[a]
            if lv == 3.0:   stats[a]["critical"] += 1; stats[a]["total"] += 1
            elif lv == 2.0: stats[a]["moderate"] += 1; stats[a]["total"] += 1
            elif lv == 1.0: stats[a]["low"]      += 1; stats[a]["total"] += 1
        if d["HEALTHY"] == 1.0:
            stats["HEALTHY"] += 1
    return stats, total

def health_score(stats, total):
    w = sum(stats[a]["critical"] * 3 + stats[a]["moderate"] * 2 + stats[a]["low"]
            for a in ACTION_META)
    return round(max(0.0, 100.0 - w / (total * 3) * 100), 1)


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────
def build_overlay(fm, rgb_bgr, action, rows, cols):
    """Returns RGB numpy array with colored overlay."""
    h, w = rgb_bgr.shape[:2]
    sev = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            sev[r, c] = fm[(r, c)][action]

    sev_full = cv2.resize(sev, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay  = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[sev_full == 3.0] = (0,   34, 255)   # BGR red
    overlay[sev_full == 2.0] = (0,  140, 255)   # BGR orange
    overlay[sev_full == 1.0] = (0,  255, 255)   # BGR yellow

    blended = cv2.addWeighted(rgb_bgr, 0.58, overlay, 0.42, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

def fig_action_map(fm, rgb_bgr, action, rows, cols, figsize=(13, 7)):
    img_rgb = build_overlay(fm, rgb_bgr, action, rows, cols)
    meta    = ACTION_META[action]

    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d1f0d")
    ax.imshow(img_rgb)
    ax.axis("off")
    ax.set_title(f"{meta['icon']}  {action}  MAP  —  GreenScan",
                 fontsize=16, fontweight="bold", color="#7FFF00",
                 fontfamily="monospace", pad=10)

    patches = [
        mpatches.Patch(color="#FF2222", label="🔴 Critical"),
        mpatches.Patch(color="#FF8C00", label="🟠 Moderate"),
        mpatches.Patch(color="#FFD700", label="🟡 Low"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=11,
              framealpha=0.85, facecolor="#1a3a1a", labelcolor="white",
              edgecolor="#3a6a3a")
    fig.tight_layout(pad=0.5)
    return fig

def fig_severity_bar(stats, action, total, figsize=(7, 4)):
    d = stats[action]
    cats   = ["Critical", "Moderate", "Low"]
    vals   = [d["critical"], d["moderate"], d["low"]]
    colors = ["#FF3333", "#FF8C00", "#FFD700"]

    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d1f0d")
    ax.set_facecolor("#111f11")
    bars = ax.bar(cats, vals, color=colors, edgecolor="#0d1f0d", linewidth=1.5,
                  width=0.55, zorder=3)
    ax.yaxis.grid(True, color="#2a4a2a", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    for bar, v in zip(bars, vals):
        pct = v / total * 100 if total else 0
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.02,
                f"{v:,}\n({pct:.1f}%)",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="white")

    ax.set_title(f"{action} — Severity Breakdown",
                 fontsize=12, fontweight="bold", color="#7FFF00", pad=8)
    ax.tick_params(colors="#88aa88")
    ax.set_ylabel("Cells", color="#88aa88", fontsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a4a2a")
    fig.tight_layout()
    return fig

def fig_ndvi_heatmap(fm, rows, cols, figsize=(13, 6)):
    matrix = np.array([[fm[(r, c)]["NDVI"] for c in range(cols)] for r in range(rows)])
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d1f0d")
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-0.5, vmax=1.0)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("NDVI Value", color="#88aa88", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#88aa88")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#88aa88")
    ax.set_title("NDVI Heatmap  (Green = Healthy · Red = Bare/Dead)",
                 fontsize=13, fontweight="bold", color="#7FFF00", pad=10)
    ax.set_xlabel("Grid Column", color="#88aa88")
    ax.set_ylabel("Grid Row",    color="#88aa88")
    ax.tick_params(colors="#88aa88")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a4a2a")
    fig.tight_layout()
    return fig

def fig_overview_donut(stats, total, figsize=(5, 5)):
    """Donut chart of all action categories."""
    labels, sizes, colors = [], [], []
    palette = {"RESEEDING": "#A0522D", "WATER": "#1E90FF",
               "UREA": "#DAA520", "STRESS": "#8A2BE2", "GROWTH": "#228B22"}
    for a in ACTION_META:
        t = stats[a]["total"]
        if t > 0:
            labels.append(f"{ACTION_META[a]['icon']} {a}")
            sizes.append(t)
            colors.append(palette[a])
    healthy = stats["HEALTHY"]
    if healthy:
        labels.append("✅ HEALTHY")
        sizes.append(healthy)
        colors.append("#7FFF00")

    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d1f0d")
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=colors,
        autopct="%1.1f%%", pctdistance=0.78,
        wedgeprops=dict(width=0.5, edgecolor="#0d1f0d", linewidth=2),
        startangle=90,
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("white")
    ax.legend(wedges, labels, loc="center left",
              bbox_to_anchor=(1, 0.5), fontsize=9,
              facecolor="#1a3a1a", labelcolor="white",
              edgecolor="#3a6a3a")
    ax.set_title("Field Health Overview", fontsize=12,
                 fontweight="bold", color="#7FFF00", pad=10)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# EXPORT HELPERS
# ─────────────────────────────────────────────
def export_csv_bytes(fm):
    rows = []
    for (r, c), d in fm.items():
        rows.append({
            "Row": r, "Col": c,
            "NDVI": d["NDVI"], "NDRE": d["NDRE"],
            "LCI":  d["LCI"],  "GNDVI": d["GNDVI"],
            "RESEEDING": d["RESEEDING"], "WATER": d["WATER"],
            "UREA": d["UREA"], "STRESS": d["STRESS"],
            "GROWTH": d["GROWTH"], "HEALTHY": d["HEALTHY"],
        })
    return pd.DataFrame(rows).to_csv(index=False).encode()

def export_pdf_bytes(stats, total, score, cell_area, cost_results, notes,
                     fm, rgb_bgr, rows, cols):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # ── PAGE 1: Summary ───────────────────────────────────────────────
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="#0a150a")
        gs  = gridspec.GridSpec(2, 2, figure=fig,
                                left=0.06, right=0.96,
                                top=0.88, bottom=0.06,
                                hspace=0.42, wspace=0.35)

        # Title
        fig.text(0.5, 0.93, "🌾  GreenScan — FIELD ANALYSIS REPORT",
                 ha="center", fontsize=18, fontweight="bold",
                 color="#7FFF00", fontfamily="monospace")
        fig.text(0.5, 0.905,
                 f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}   |   "
                 f"Health Score: {score}/100   |   Total Cells: {total:,}",
                 ha="center", fontsize=9, color="#88aa88")

        # Donut chart
        ax1 = fig.add_subplot(gs[0, 0])
        _draw_donut_on_ax(ax1, stats, total)

        # Stats table
        ax2 = fig.add_subplot(gs[0, 1])
        _draw_stats_table_on_ax(ax2, stats, total, cell_area)

        # Cost table
        ax3 = fig.add_subplot(gs[1, 0])
        _draw_cost_table_on_ax(ax3, cost_results)

        # Notes
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")
        ax4.set_facecolor("#111f11")
        note_txt = notes if notes.strip() else "No field notes added."
        ax4.text(0.05, 0.95, "Field Notes:", fontsize=10,
                 fontweight="bold", color="#7FFF00", va="top",
                 transform=ax4.transAxes)
        ax4.text(0.05, 0.82, note_txt, fontsize=8.5,
                 color="#c0dcc0", va="top", wrap=True,
                 transform=ax4.transAxes)

        pdf.savefig(fig, bbox_inches="tight", facecolor="#0a150a")
        plt.close(fig)

        # ── PAGE 2: Action Maps ───────────────────────────────────────────
        for action in ACTION_META:
            f2 = fig_action_map(fm, rgb_bgr, action, rows, cols, figsize=(11.69, 6.5))
            pdf.savefig(f2, bbox_inches="tight", facecolor="#0d1f0d")
            plt.close(f2)

        # ── PAGE 3: NDVI Heatmap ──────────────────────────────────────────
        f3 = fig_ndvi_heatmap(fm, rows, cols, figsize=(11.69, 5))
        pdf.savefig(f3, bbox_inches="tight", facecolor="#0d1f0d")
        plt.close(f3)

    buf.seek(0)
    return buf.read()

def _draw_donut_on_ax(ax, stats, total):
    palette = {"RESEEDING": "#A0522D", "WATER": "#1E90FF",
               "UREA": "#DAA520", "STRESS": "#8A2BE2",
               "GROWTH": "#228B22", "HEALTHY": "#7FFF00"}
    sizes, colors, labels = [], [], []
    for a in list(ACTION_META.keys()) + ["HEALTHY"]:
        v = stats[a]["total"] if a != "HEALTHY" else stats["HEALTHY"]
        if v > 0:
            sizes.append(v); colors.append(palette[a])
            icon = ACTION_META[a]["icon"] if a in ACTION_META else "✅"
            labels.append(f"{icon} {a}")
    ax.pie(sizes, colors=colors,
           wedgeprops=dict(width=0.5, edgecolor="#0a150a", linewidth=1.5),
           startangle=90)
    ax.set_title("Overview", fontsize=10, fontweight="bold", color="#7FFF00")
    ax.legend(labels, loc="lower center", fontsize=7,
              facecolor="#1a3a1a", labelcolor="white",
              edgecolor="#3a6a3a", ncol=2,
              bbox_to_anchor=(0.5, -0.25))
    ax.set_facecolor("#111f11")

def _draw_stats_table_on_ax(ax, stats, total, cell_area):
    ax.axis("off")
    ax.set_facecolor("#111f11")
    cols_h = ["Action", "Critical", "Moderate", "Low", "Total", "Area(sqm)"]
    rows_d = []
    for a in ACTION_META:
        s = stats[a]
        rows_d.append([
            f"{ACTION_META[a]['icon']} {a}",
            f"{s['critical']:,}",
            f"{s['moderate']:,}",
            f"{s['low']:,}",
            f"{s['total']:,}",
            f"{s['total']*cell_area:.1f}",
        ])
    tbl = ax.table(cellText=rows_d, colLabels=cols_h,
                   cellLoc="center", loc="center",
                   bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#2a4a2a")
        if row == 0:
            cell.set_facecolor("#1a3a1a")
            cell.set_text_props(color="#7FFF00", fontweight="bold")
        else:
            cell.set_facecolor("#0d1f0d" if row % 2 else "#111f11")
            cell.set_text_props(color="#c0dcc0")
    ax.set_title("Detailed Statistics", fontsize=10,
                 fontweight="bold", color="#7FFF00", pad=6)

def _draw_cost_table_on_ax(ax, cost_results):
    ax.axis("off")
    if not cost_results:
        ax.text(0.5, 0.5, "No cost analysis data.\n(Run cost analysis on results page)",
                ha="center", va="center", color="#88aa88", fontsize=9,
                transform=ax.transAxes)
        ax.set_title("Cost Analysis", fontsize=10, fontweight="bold",
                     color="#7FFF00", pad=6)
        return
    rows_d = [[k, f"₹{v:,.2f}"] for k, v in cost_results.items()]
    total_cost = sum(cost_results.values())
    rows_d.append(["TOTAL", f"₹{total_cost:,.2f}"])
    tbl = ax.table(cellText=rows_d, colLabels=["Item", "Amount"],
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#2a4a2a")
        if row == 0:
            cell.set_facecolor("#1a3a1a")
            cell.set_text_props(color="#7FFF00", fontweight="bold")
        elif row == len(rows_d):
            cell.set_facecolor("#2d4a1a")
            cell.set_text_props(color="#7FFF00", fontweight="bold")
        else:
            cell.set_facecolor("#0d1f0d" if row % 2 else "#111f11")
            cell.set_text_props(color="#c0dcc0")
    ax.set_title("Cost Estimate", fontsize=10, fontweight="bold",
                 color="#7FFF00", pad=6)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 12px 0 6px 0;'>
          <span style='font-size:3rem;'>🌾</span>
          <h2 style='color:#7FFF00; margin:0; font-family:Rajdhani,sans-serif; font-size:1.6rem;'>
            GreenScan
          </h2>
          <p style='color:#88aa88; font-size:0.78rem; margin:2px 0 0 0;'>
            Drone-based Precision Farming
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        page = st.radio(
            "Navigate",
            ["📤  Upload & Configure", "📊  Results Dashboard"],
            label_visibility="collapsed",
        )

        st.divider()

        if st.session_state.analysis_done:
            score = st.session_state.health_score
            color = "#7FFF00" if score >= 70 else ("#FFD700" if score >= 40 else "#FF4444")
            st.markdown(f"""
            <div class='kd-metric'>
              <div class='label'>Farm Health</div>
              <div class='value' style='color:{color};'>{score}<span style='font-size:1rem'>/100</span></div>
              <div class='sub'>{'Healthy 🟢' if score>=70 else ('Moderate 🟡' if score>=40 else 'Critical 🔴')}</div>
            </div>
            """, unsafe_allow_html=True)

            total  = st.session_state.total_cells
            stats  = st.session_state.stats
            st.markdown(f"""
            <div style='font-size:0.82rem; color:#88aa88; margin-top:10px;'>
              <b style='color:#c0dcc0;'>Quick Stats</b><br>
              Total cells: <b style='color:#7FFF00;'>{total:,}</b><br>
              Healthy: <b style='color:#7FFF00;'>{stats['HEALTHY']:,}</b>
              ({stats['HEALTHY']/total*100:.0f}%)<br>
              Needs attention: <b style='color:#FFD700;'>
              {sum(stats[a]['total'] for a in ACTION_META):,}</b>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown("""
        <div style='font-size:0.75rem; color:#556655; text-align:center;'>
          Pure Python · No ML Models<br>
          NDVI · NDRE · LCI · GNDVI
        </div>
        """, unsafe_allow_html=True)

    return page


# ─────────────────────────────────────────────
# PAGE 1 — UPLOAD & CONFIGURE
# ─────────────────────────────────────────────
def page_upload():
    st.markdown("""
    <div class='kd-hero'>
      <h1>📤  Upload & Configure</h1>
      <p>Upload your multispectral drone images. Missing bands will be auto-filled with smart synthetic defaults.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── BAND UPLOAD ───────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🛰️ Spectral Band Images</div>",
                unsafe_allow_html=True)
    st.caption("**JPG / PNG** for RGB &nbsp;·&nbsp; **TIF / TIFF** for spectral bands &nbsp;·&nbsp; "
               "Missing optional bands will use synthetic defaults (results may vary).")

    col_rgb, col_r, col_g, col_nir, col_re = st.columns(5)

    def band_col(col, label, key, required=False, accept=None):
        if accept is None:
            accept = ["jpg", "jpeg", "png", "tif", "tiff"]
        with col:
            f = st.file_uploader(label, type=accept, key=key,
                                 label_visibility="collapsed")
            if f:
                st.markdown(f"<div class='band-card'><div class='band-name'>{label}</div>"
                            f"<div class='band-status status-ok'>✅ Uploaded</div></div>",
                            unsafe_allow_html=True)
            elif required:
                st.markdown(f"<div class='band-card'><div class='band-name'>{label}</div>"
                            f"<div class='band-status status-error'>⚠️ Required</div></div>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='band-card'><div class='band-name'>{label}</div>"
                            f"<div class='band-status status-warn'>🔄 Auto-default</div></div>",
                            unsafe_allow_html=True)
            return f

    rgb_file = band_col(col_rgb, "📸 RGB Image",   "up_rgb",  required=True,
                        accept=["jpg", "jpeg", "png", "tif", "tiff"])
    r_file   = band_col(col_r,   "🔴 Red Band",    "up_r")
    g_file   = band_col(col_g,   "🟢 Green Band",  "up_g")
    nir_file = band_col(col_nir, "🌿 NIR Band",    "up_nir")
    re_file  = band_col(col_re,  "🔵 RedEdge",     "up_re")

    st.divider()

    # ── CAMERA & GRID ─────────────────────────────────────────────────────
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("<div class='section-header'>📷 Camera & Flight Settings</div>",
                    unsafe_allow_html=True)
        st.caption("Used to auto-calculate **GSD** (Ground Sampling Distance) — real area per pixel.")

        c1, c2 = st.columns(2)
        with c1:
            fh  = st.number_input("Flight Height (m)",  value=50.0,  min_value=1.0,   step=1.0)
            sw  = st.number_input("Sensor Width (mm)",  value=6.17,  min_value=1.0,   step=0.01)
            fl  = st.number_input("Focal Length (mm)",  value=4.5,   min_value=0.5,   step=0.1)
        with c2:
            iw  = st.number_input("Image Width (px)",   value=4000,  min_value=100,   step=100)
            ih  = st.number_input("Image Height (px)",  value=3000,  min_value=100,   step=100)
            gsd_w, gsd_h = calc_gsd(fh, sw, fl, iw, ih)
            st.metric("GSD (width)",  f"{gsd_w * 100:.2f} cm/px")
            st.metric("GSD (height)", f"{gsd_h * 100:.2f} cm/px")

    with right_col:
        st.markdown("<div class='section-header'>⚙️ Grid Settings</div>",
                    unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            grid_c = st.number_input("Grid Columns", value=DEFAULT_GRID_COLS,
                                     min_value=16, max_value=512, step=16)
            grid_r = st.number_input("Grid Rows",    value=DEFAULT_GRID_ROWS,
                                     min_value=12, max_value=384, step=12)
        with c2:
            cell_a = calc_cell_area(fh, sw, fl, iw, ih, grid_r, grid_c)
            st.metric("Total Cells",    f"{grid_r * grid_c:,}")
            st.metric("Area per Cell",  f"{cell_a:.4f} sqm")
            st.metric("Total Coverage", f"{grid_r * grid_c * cell_a:.0f} sqm")

    st.divider()

    # ── THRESHOLD SETTINGS ────────────────────────────────────────────────
    with st.expander("🎛️  Advanced: Threshold Settings  (click to expand)",
                     expanded=False):
        st.caption("Lower values = more sensitive detection. Default values work well for most fields.")

        thresholds = {}
        t_tabs = st.tabs([f"{ACTION_META[a]['icon']} {a}"
                          for a in ACTION_META])
        for tab, key in zip(t_tabs, ACTION_META):
            with tab:
                defaults = DEFAULT_THRESHOLDS[key]
                st.caption(ACTION_META[key]["desc"])
                ca, cb, cc = st.columns(3)
                with ca:
                    crit = st.slider("🔴 Critical threshold", 0.0, 1.0,
                                     defaults["critical"], 0.01,
                                     key=f"t_{key}_c",
                                     help="Cells below this = Critical (immediate action)")
                with cb:
                    mod  = st.slider("🟠 Moderate threshold", 0.0, 1.0,
                                     defaults["moderate"], 0.01,
                                     key=f"t_{key}_m")
                with cc:
                    low  = st.slider("🟡 Low threshold",      0.0, 1.0,
                                     defaults["low"], 0.01,
                                     key=f"t_{key}_l")
                thresholds[key] = {"critical": crit, "moderate": mod, "low": low}

    if not thresholds:
        thresholds = {k: dict(v) for k, v in DEFAULT_THRESHOLDS.items()}

    st.divider()

    # ── FIELD NOTES ──────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📝 Field Notes (Optional)</div>",
                unsafe_allow_html=True)
    notes = st.text_area(
        "notes",
        value=st.session_state.field_notes,
        label_visibility="collapsed",
        placeholder="e.g., Heavy rain last week · Clay soil in NE corner · "
                    "Pest spotted near boundary · Last irrigation: 5 days ago…",
        height=90,
    )
    st.session_state.field_notes = notes

    st.divider()

    # ── RUN BUTTON ────────────────────────────────────────────────────────
    col_btn, col_msg = st.columns([1, 3])
    with col_btn:
        run = st.button("🚀  Start Analysis", type="primary", use_container_width=True)
    with col_msg:
        if rgb_file is None:
            st.error("⚠️ RGB image is required before analysis.")
        else:
            st.success("✅ Ready! Click **Start Analysis** to begin.")

    if run:
        if rgb_file is None:
            st.error("❌ Please upload the RGB image first.")
            return
        _run_analysis(
            rgb_file, r_file, g_file, nir_file, re_file,
            grid_r, grid_c, thresholds,
            fh, sw, fl, iw, ih,
        )


def _run_analysis(rgb_file, r_file, g_file, nir_file, re_file,
                  grid_r, grid_c, thresholds, fh, sw, fl, iw, ih):

    st.divider()
    st.markdown("<div class='section-header'>⚙️ Analysis Running…</div>",
                unsafe_allow_html=True)

    prog   = st.progress(0, text="Initializing…")
    status = st.empty()

    try:
        # 1. Load RGB
        prog.progress(3, text="📸 Loading RGB image…")
        rgb_bgr = load_rgb(rgb_file)
        if rgb_bgr is None:
            st.error("❌ Could not decode RGB image.")
            return
        img_h, img_w = rgb_bgr.shape[:2]
        status.info(f"✅ RGB loaded — {img_w}×{img_h} px")

        # 2. Load spectral bands
        prog.progress(8, text="🛰️ Loading spectral bands…")
        band_r,   dr = load_band(r_file,   img_h, img_w, 0.40)
        band_g,   dg = load_band(g_file,   img_h, img_w, 0.45)
        band_nir, dn = load_band(nir_file, img_h, img_w, 0.70)
        band_re,  de = load_band(re_file,  img_h, img_w, 0.38)

        defs = [name for name, flag in
                [("Red", dr), ("Green", dg), ("NIR", dn), ("RedEdge", de)] if flag]
        if defs:
            status.warning(f"⚠️ Using synthetic defaults for: **{', '.join(defs)}**. "
                           "Upload real bands for accurate results.")

        # 3. Init map
        prog.progress(12, text=f"💾 Initializing {grid_r}×{grid_c} grid map…")
        fm = init_farm_map(grid_r, grid_c)
        status.info(f"💾 Grid initialized — {grid_r * grid_c:,} cells")

        # 4. Compute indices
        status.info("🧠 Computing NDVI · NDRE · LCI · GNDVI…")
        fm = fill_indices(fm, band_r, band_g, band_nir, band_re,
                          grid_r, grid_c, prog)

        # 5. Apply flags
        prog.progress(62, text="🏷️ Classifying cells…")
        status.info("🏷️ Applying action flags…")
        fm = apply_flags(fm, thresholds, prog)

        # 6. Stats & score
        prog.progress(98, text="📊 Generating statistics…")
        stats, total = build_stats(fm)
        score = health_score(stats, total)
        cell_a = calc_cell_area(fh, sw, fl, iw, ih, grid_r, grid_c)

        # Save to state
        st.session_state.farm_map     = fm
        st.session_state.rgb_img      = rgb_bgr
        st.session_state.stats        = stats
        st.session_state.total_cells  = total
        st.session_state.health_score = score
        st.session_state.cell_area    = cell_a
        st.session_state.grid_rows    = grid_r
        st.session_state.grid_cols    = grid_c
        st.session_state.thresholds   = thresholds
        st.session_state.analysis_done = True

        prog.progress(100, text="✅ Done!")
        status.success(
            f"🎉 Analysis complete!  "
            f"Health Score: **{score}/100**  |  "
            f"Healthy cells: **{stats['HEALTHY']:,}** / {total:,}"
        )
        st.balloons()
        st.info("👉 Open **📊 Results Dashboard** from the sidebar to view maps and reports.")

    except Exception as exc:
        st.error(f"❌ Error during analysis: {exc}")
        with st.expander("🔍 Error Details"):
            import traceback
            st.code(traceback.format_exc())


# ─────────────────────────────────────────────
# PAGE 2 — RESULTS DASHBOARD
# ─────────────────────────────────────────────
def page_results():
    if not st.session_state.analysis_done:
        st.markdown("""
        <div class='kd-hero'>
          <h1>📊  Results Dashboard</h1>
          <p>No analysis data yet. Go to <b>Upload & Configure</b> and run analysis first.</p>
        </div>
        """, unsafe_allow_html=True)
        st.warning("⚠️ Please upload images and click **Start Analysis** first.")
        return

    fm       = st.session_state.farm_map
    rgb_bgr  = st.session_state.rgb_img
    stats    = st.session_state.stats
    total    = st.session_state.total_cells
    score    = st.session_state.health_score
    cell_a   = st.session_state.cell_area
    grid_r   = st.session_state.grid_rows
    grid_c   = st.session_state.grid_cols

    score_label = ("Excellent 🟢" if score >= 70 else
                   ("Moderate 🟡" if score >= 40 else "Critical 🔴"))
    score_col   = ("#7FFF00" if score >= 70 else
                   ("#FFD700" if score >= 40 else "#FF4444"))

    st.markdown(f"""
    <div class='kd-hero'>
      <h1>📊  Results Dashboard</h1>
      <p>Analysis complete — <b style='color:{score_col}'>{score}/100 Health Score ({score_label})</b>
         &nbsp;·&nbsp; {total:,} cells analyzed &nbsp;·&nbsp; {cell_a:.4f} sqm/cell</p>
    </div>
    """, unsafe_allow_html=True)

    # ── OVERVIEW METRICS ──────────────────────────────────────────────────
    m_cols = st.columns(7)
    metrics = [
        ("Health Score",  f"{score}/100", score_col),
        ("Total Cells",   f"{total:,}",   "#7FFF00"),
        ("Healthy",       f"{stats['HEALTHY']:,}", "#7FFF00"),
        ("Reseeding",     f"{stats['RESEEDING']['total']:,}", "#A0522D"),
        ("Water Stress",  f"{stats['WATER']['total']:,}",     "#1E90FF"),
        ("Urea Need",     f"{stats['UREA']['total']:,}",      "#DAA520"),
        ("Pest/Stress",   f"{stats['STRESS']['total']:,}",    "#8A2BE2"),
    ]
    for col, (label, val, color) in zip(m_cols, metrics):
        with col:
            st.markdown(f"""
            <div class='kd-metric'>
              <div class='label'>{label}</div>
              <div class='value' style='color:{color};'>{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── MAIN TABS ─────────────────────────────────────────────────────────
    tab_labels = [
        f"{ACTION_META[a]['icon']} {a}" for a in ACTION_META
    ] + ["🌡️ NDVI Heatmap", "📈 Overview", "💰 Cost Analysis", "📥 Export"]

    tabs = st.tabs(tab_labels)

    # Action tabs
    for i, (tab, action) in enumerate(zip(tabs[:5], ACTION_META)):
        with tab:
            _tab_action(fm, rgb_bgr, stats, total, action, cell_a, grid_r, grid_c)

    # NDVI Heatmap
    with tabs[5]:
        st.markdown("<div class='section-header'>🌡️ NDVI Distribution Heatmap</div>",
                    unsafe_allow_html=True)
        st.caption("**Green** = Healthy vegetation &nbsp;·&nbsp; **Red** = Bare soil / dead plants")
        with st.spinner("Generating heatmap…"):
            fig = fig_ndvi_heatmap(fm, grid_r, grid_c)
            st.pyplot(fig)
            plt.close(fig)

    # Overview donut
    with tabs[6]:
        st.markdown("<div class='section-header'>📈 Field Health Overview</div>",
                    unsafe_allow_html=True)
        col_d, col_t = st.columns([1, 2])
        with col_d:
            with st.spinner("Building overview chart…"):
                fig = fig_overview_donut(stats, total)
                st.pyplot(fig)
                plt.close(fig)
        with col_t:
            # Full summary table
            rows_tbl = []
            for a in ACTION_META:
                s = stats[a]
                rows_tbl.append({
                    "Action": f"{ACTION_META[a]['icon']} {a}",
                    "Description": ACTION_META[a]["desc"],
                    "🔴 Critical": s["critical"],
                    "🟠 Moderate": s["moderate"],
                    "🟡 Low":      s["low"],
                    "Total Cells": s["total"],
                    "% Field":     f"{s['total']/total*100:.2f}%",
                    "Area (sqm)":  f"{s['total']*cell_a:.1f}",
                })
            df = pd.DataFrame(rows_tbl)
            st.dataframe(df, use_container_width=True, hide_index=True)

    # Cost Analysis
    with tabs[7]:
        _tab_cost(stats, total, cell_a)

    # Export
    with tabs[8]:
        _tab_export(fm, rgb_bgr, stats, total, score, cell_a, grid_r, grid_c)


def _tab_action(fm, rgb_bgr, stats, total, action, cell_a, grid_r, grid_c):
    meta = ACTION_META[action]
    s    = stats[action]
    total_affected = s["total"]
    affected_area  = total_affected * cell_a

    st.markdown(f"<div class='section-header'>{meta['icon']} {action} — {meta['desc']}</div>",
                unsafe_allow_html=True)

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🔴 Critical",  f"{s['critical']:,}",
              f"{s['critical']/total*100:.1f}% of field")
    m2.metric("🟠 Moderate",  f"{s['moderate']:,}",
              f"{s['moderate']/total*100:.1f}% of field")
    m3.metric("🟡 Low",       f"{s['low']:,}",
              f"{s['low']/total*100:.1f}% of field")
    m4.metric("📐 Affected Area", f"{affected_area:.1f} sqm",
              f"{total_affected:,} cells")
    priority = ("🔴 HIGH" if s["critical"] / total > 0.10 else
                ("🟠 MEDIUM" if total_affected / total > 0.05 else "🟢 LOW"))
    m5.metric("🎯 Priority", priority)

    st.divider()

    col_map, col_bar = st.columns([2, 1])

    with col_map:
        with st.spinner(f"Rendering {action} overlay map…"):
            fig = fig_action_map(fm, rgb_bgr, action, grid_r, grid_c)
            st.pyplot(fig)
            plt.close(fig)

    with col_bar:
        with st.spinner("Building chart…"):
            fig2 = fig_severity_bar(stats, action, total)
            st.pyplot(fig2)
            plt.close(fig2)

        st.markdown(f"""
        <div class='kd-summary'>
          <b>📋 Summary</b><br>
          Affected cells: <b>{total_affected:,}</b><br>
          Area needing action: <b>{affected_area:.1f} sqm</b><br>
          Percentage of field: <b>{total_affected/total*100:.2f}%</b><br>
          Priority: <b>{priority}</b>
        </div>
        """, unsafe_allow_html=True)


def _tab_cost(stats, total, cell_a):
    st.markdown("<div class='section-header'>💰 Cost Analysis & Treatment Estimation</div>",
                unsafe_allow_html=True)
    st.caption("Enter your local rates below. Costs are estimated based on affected area per action.")

    col_in, col_out = st.columns([1, 1])

    with col_in:
        st.markdown("**⚙️ Treatment Rates (₹ per sqm)**")
        ca, cb = st.columns(2)
        with ca:
            rate_rs  = st.number_input("🟤 Reseeding",    value=5.0,  min_value=0.0, step=0.5)
            rate_wt  = st.number_input("💧 Irrigation",   value=2.0,  min_value=0.0, step=0.5)
            rate_ur  = st.number_input("🟡 Fertilizer",   value=3.0,  min_value=0.0, step=0.5)
        with cb:
            rate_pe  = st.number_input("🟣 Pesticide",    value=4.0,  min_value=0.0, step=0.5)
            labor    = st.number_input("👷 Labor",         value=1.5,  min_value=0.0, step=0.25)
            overhead = st.number_input("📦 Overhead (%)",  value=10.0, min_value=0.0, max_value=100.0)

        st.markdown("**📊 Treatment Coverage by Severity**")
        st.caption("What % of cells at each severity level will be treated?")
        wc = st.slider("Critical cells treated (%)", 50, 100, 100, 5)
        wm = st.slider("Moderate cells treated (%)", 25, 100,  75, 5)
        wl = st.slider("Low cells treated (%)",       0, 100,  50, 5)

    with col_out:
        def cost_for(action_stats, rate):
            area = (action_stats["critical"] * cell_a * wc / 100 +
                    action_stats["moderate"] * cell_a * wm / 100 +
                    action_stats["low"]      * cell_a * wl / 100)
            return area * (rate + labor)

        c_rs = cost_for(stats["RESEEDING"], rate_rs)
        c_wt = cost_for(stats["WATER"],     rate_wt)
        c_ur = cost_for(stats["UREA"],      rate_ur)
        c_pe = cost_for(stats["STRESS"],    rate_pe)
        subtotal = c_rs + c_wt + c_ur + c_pe
        ovh      = subtotal * overhead / 100
        grand    = subtotal + ovh

        # Save for PDF
        st.session_state.cost_results = {
            "🟤 Reseeding":   c_rs,
            "💧 Irrigation":  c_wt,
            "🟡 Fertilizer":  c_ur,
            "🟣 Pesticide":   c_pe,
            "📦 Overhead":    ovh,
        }

        rows_cost = [
            {"Treatment": "🟤 Reseeding",  "Cells": stats["RESEEDING"]["total"],
             "Area (sqm)": f"{stats['RESEEDING']['total']*cell_a:.1f}", "Cost (₹)": f"{c_rs:,.2f}"},
            {"Treatment": "💧 Irrigation", "Cells": stats["WATER"]["total"],
             "Area (sqm)": f"{stats['WATER']['total']*cell_a:.1f}",     "Cost (₹)": f"{c_wt:,.2f}"},
            {"Treatment": "🟡 Fertilizer", "Cells": stats["UREA"]["total"],
             "Area (sqm)": f"{stats['UREA']['total']*cell_a:.1f}",      "Cost (₹)": f"{c_ur:,.2f}"},
            {"Treatment": "🟣 Pesticide",  "Cells": stats["STRESS"]["total"],
             "Area (sqm)": f"{stats['STRESS']['total']*cell_a:.1f}",    "Cost (₹)": f"{c_pe:,.2f}"},
            {"Treatment": "📦 Overhead",   "Cells": "-",
             "Area (sqm)": "-",                                          "Cost (₹)": f"{ovh:,.2f}"},
        ]
        df = pd.DataFrame(rows_cost)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown(f"""
        <div class='cost-total'>
          💰 Total Estimated Cost<br>
          <span style='font-size:2.6rem;'>₹{grand:,.2f}</span>
        </div>
        """, unsafe_allow_html=True)

        # Priority order
        st.markdown("**🎯 Recommended Action Priority:**")
        priority_list = sorted(
            [("RESEEDING", c_rs, "🟤"), ("WATER", c_wt, "💧"),
             ("UREA", c_ur, "🟡"), ("STRESS", c_pe, "🟣")],
            key=lambda x: stats[x[0]]["critical"], reverse=True
        )
        for rank, (action, cost, icon) in enumerate(priority_list, 1):
            crit = stats[action]["critical"]
            if stats[action]["total"] > 0:
                st.markdown(
                    f"**{rank}.** {icon} `{action}` — "
                    f"{crit:,} critical cells — **₹{cost:,.2f}**"
                )


def _tab_export(fm, rgb_bgr, stats, total, score, cell_a, grid_r, grid_c):
    st.markdown("<div class='section-header'>📥 Export & Download</div>",
                unsafe_allow_html=True)

    notes = st.session_state.get("field_notes", "")
    cost_results = st.session_state.get("cost_results", {})

    col1, col2, col3 = st.columns(3)

    # CSV
    with col1:
        st.markdown("**📊 Raw Grid Data (CSV)**")
        st.caption(f"All {total:,} cells — indices & action flags")
        with st.spinner("Preparing CSV…"):
            csv_bytes = export_csv_bytes(fm)
        st.download_button(
            "⬇️  Download CSV",
            data=csv_bytes,
            file_name=f"kisan_drishti_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # PDF
    with col2:
        st.markdown("**📄 Full Analysis Report (PDF)**")
        st.caption("Maps · stats · cost breakdown · notes")
        with st.spinner("Generating PDF report…"):
            try:
                pdf_bytes = export_pdf_bytes(
                    stats, total, score, cell_a, cost_results, notes,
                    fm, rgb_bgr, grid_r, grid_c
                )
                st.download_button(
                    "⬇️  Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"kisan_drishti_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"PDF error: {e}")

    # JSON
    with col3:
        st.markdown("**🗂️ Summary (JSON)**")
        st.caption("Machine-readable analysis summary")
        summary = {
            "generated": datetime.now().isoformat(),
            "health_score": score,
            "total_cells": total,
            "cell_area_sqm": round(cell_a, 6),
            "grid": {"rows": grid_r, "cols": grid_c},
            "statistics": {a: stats[a] for a in ACTION_META},
            "healthy_cells": stats["HEALTHY"],
            "cost_estimate": {k: round(v, 2) for k, v in cost_results.items()},
            "field_notes": notes,
        }
        st.download_button(
            "⬇️  Download JSON",
            data=json.dumps(summary, indent=2).encode(),
            file_name=f"kisan_drishti_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.divider()

    # Field notes display
    if notes:
        st.markdown("<div class='section-header'>📝 Field Notes</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='kd-summary'>{notes}</div>",
                    unsafe_allow_html=True)

    # Full stats table
    st.markdown("<div class='section-header'>📊 Complete Statistics</div>",
                unsafe_allow_html=True)
    rows_s = []
    for a in ACTION_META:
        s = stats[a]
        rows_s.append({
            "Action":       f"{ACTION_META[a]['icon']} {a}",
            "Critical 🔴":  s["critical"],
            "Moderate 🟠":  s["moderate"],
            "Low 🟡":       s["low"],
            "Total Cells":  s["total"],
            "% of Field":   f"{s['total']/total*100:.2f}%",
            "Area (sqm)":   f"{s['total']*cell_a:.1f}",
        })
    st.dataframe(pd.DataFrame(rows_s), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
def main():
    inject_css()
    init_state()
    page = render_sidebar()

    if "Upload" in page:
        page_upload()
    else:
        page_results()


if __name__ == "__main__":
    main()
