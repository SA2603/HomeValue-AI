import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

from features import CITY_CENTERS, RAW_COLS  # noqa: F401  (needed so joblib can unpickle)

st.set_page_config(page_title="HomeValue.ai | Instant home valuation", page_icon="🏠",
                   layout="wide", initial_sidebar_state="collapsed")

BG, TEXT, MUTED = "#0A0F1F", "#E8ECF8", "#8D97B5"
TEAL, INDIGO, PINK = "#2DD4BF", "#818CF8", "#F472B6"
UNIT = 100_000  # the model predicts in units of $100,000

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=DM+Sans:wght@400;500;600&display=swap');

html, body, .stApp, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}
.stApp {{
  color: {TEXT};
  background:
    radial-gradient(900px 520px at 6% -8%, rgba(129,140,248,.38), transparent 62%),
    radial-gradient(800px 520px at 102% 2%, rgba(45,212,191,.24), transparent 58%),
    radial-gradient(700px 500px at 50% 112%, rgba(244,114,182,.14), transparent 60%),
    {BG};
  background-attachment: fixed;
}}
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display: none !important; }}
.block-container {{ padding: 1.4rem 2rem 3rem; max-width: 1220px; }}
.stApp p, .stApp label, .stApp span, .stApp li {{ color: {TEXT}; }}
h1, h2, h3 {{ font-family: 'Sora', sans-serif !important; letter-spacing: -0.025em; color: #fff; }}

/* ---------- top nav ---------- */
.nav {{ display:flex; justify-content:space-between; align-items:center; margin-bottom: 2.2rem; }}
.logo {{ display:flex; align-items:center; gap:.65rem; font-family:'Sora'; font-weight:700; font-size:1.2rem; color:#fff; }}
.logo-mark {{ width:36px; height:36px; border-radius:11px; display:grid; place-items:center;
  background: linear-gradient(135deg, {TEAL}, {INDIGO}); box-shadow: 0 6px 24px rgba(45,212,191,.35); }}
.pill {{ padding:.38rem .9rem; border-radius:999px; font-size:.82rem; font-weight:500; color:{TEXT};
  background: rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.12); }}
.pill b {{ color:{TEAL}; font-weight:600; }}

/* ---------- hero ---------- */
.hero h1 {{ font-size: clamp(2.3rem, 4.6vw, 3.7rem); line-height:1.04; font-weight:800; margin:0 0 1rem; }}
.grad {{ background: linear-gradient(90deg, {TEAL}, {INDIGO} 55%, {PINK}); -webkit-background-clip:text; background-clip:text; color:transparent; }}
.hero p {{ font-size:1.08rem; color:{MUTED} !important; max-width:50ch; margin:0 0 1.6rem; line-height:1.6; }}

/* ---------- glass cards ---------- */
.st-key-panel {{ background: rgba(255,255,255,.045); border:1px solid rgba(255,255,255,.11); border-radius:22px;
  padding:1.5rem 1.6rem 1.2rem; backdrop-filter: blur(16px); box-shadow: 0 20px 60px rgba(0,0,0,.35); }}
.panel-title {{ font-family:'Sora'; font-weight:700; font-size:1.05rem; color:#fff; margin-bottom:.6rem; }}
[data-testid="stWidgetLabel"] p {{ color:#C5CCE6 !important; font-size:.86rem; font-weight:500; }}
[data-testid="stSliderThumbValue"] {{ color:{TEAL} !important; font-weight:600; }}
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {{ color:{MUTED} !important; }}
div[data-baseweb="select"] > div {{ background: rgba(255,255,255,.07) !important; border:1px solid rgba(255,255,255,.14) !important; border-radius:12px !important; color:#fff !important; }}
div[data-baseweb="select"] * {{ color:#fff !important; }}
ul[role="listbox"], div[data-baseweb="popover"] > div {{ background:#141B33 !important; }}
li[role="option"] {{ color:{TEXT} !important; }}
li[role="option"]:hover {{ background: rgba(45,212,191,.16) !important; }}
input {{ color:#fff !important; }}
[data-testid="stNumberInput"] input {{ background: rgba(255,255,255,.07) !important; }}
[data-testid="stExpander"] {{ background: rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.09) !important; border-radius:14px; }}
[data-testid="stExpander"] summary p {{ color:{MUTED} !important; }}

.result {{ position:relative; overflow:hidden; border-radius:26px; padding:1.9rem 2rem 1.7rem; color:#fff;
  background: linear-gradient(145deg, rgba(45,212,191,.20), rgba(129,140,248,.22) 55%, rgba(244,114,182,.16));
  border:1px solid rgba(255,255,255,.18); box-shadow: 0 24px 70px rgba(99,102,241,.28); }}
.result::after {{ content:""; position:absolute; width:260px; height:260px; right:-80px; top:-100px; border-radius:50%;
  background: radial-gradient(circle, rgba(255,255,255,.18), transparent 65%); }}
.result .eyebrow {{ font-size:.78rem; letter-spacing:.14em; text-transform:uppercase; color:rgba(255,255,255,.72); font-weight:600; }}
.result .price {{ font-family:'Sora'; font-weight:800; font-size:clamp(2.6rem, 5.4vw, 4.3rem); letter-spacing:-0.035em; line-height:1.08; margin:.35rem 0 1.2rem; }}
.track {{ position:relative; height:10px; border-radius:99px; background: rgba(255,255,255,.14); }}
.track .fill {{ position:absolute; inset:0; border-radius:99px; background: linear-gradient(90deg, {TEAL}, {INDIGO}); }}
.track .dot {{ position:absolute; top:50%; width:22px; height:22px; border-radius:50%; background:#fff; border:5px solid {PINK};
  transform: translate(-50%,-50%); box-shadow:0 4px 14px rgba(0,0,0,.4); }}
.track-labels {{ display:flex; justify-content:space-between; margin-top:.8rem; font-size:.92rem; font-weight:600; }}
.track-labels small {{ display:block; font-weight:400; color:rgba(255,255,255,.65); font-size:.75rem; }}
.track-labels div:last-child {{ text-align:right; }}
.result .note {{ margin-top:1.1rem; font-size:.85rem; color:rgba(255,255,255,.72); }}

.tiles {{ display:grid; grid-template-columns: repeat(3, 1fr); gap:.9rem; margin-top:1rem; }}
.tile {{ background: rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.11); border-radius:16px; padding:.95rem 1.05rem; }}
.tile .k {{ font-size:.76rem; color:{MUTED}; }}
.tile .v {{ font-family:'Sora'; font-weight:700; font-size:1.25rem; margin-top:.2rem; color:#fff; }}
.tile .v.up {{ color:{TEAL}; }} .tile .v.down {{ color:{PINK}; }}

/* ---------- tabs & metrics ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap:.4rem; border-bottom:1px solid rgba(255,255,255,.1); }}
.stTabs [data-baseweb="tab"] {{ background:transparent; border-radius:10px 10px 0 0; padding:.6rem 1.1rem; color:{MUTED}; font-weight:500; }}
.stTabs [aria-selected="true"] {{ color:#fff !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: linear-gradient(90deg,{TEAL},{INDIGO}) !important; height:3px; }}
div[data-testid="stMetric"] {{ background: rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.11); border-radius:16px; padding:.9rem 1.1rem; }}
[data-testid="stMetricLabel"] p {{ color:{MUTED} !important; }}
[data-testid="stMetricValue"] {{ font-family:'Sora'; color:#fff; }}
.foot {{ color:{MUTED}; font-size:.82rem; margin-top:2.4rem; text-align:center; }}
:focus-visible {{ outline: 3px solid {PINK} !important; outline-offset: 2px; }}
@media (max-width: 760px) {{ .tiles {{ grid-template-columns: 1fr; }} .block-container {{ padding: 1rem; }} .pill {{ display:none; }} }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the saved model. If it is missing, or was saved by a different scikit-learn
    version than the one running this app, retrain automatically (about a minute)."""
    model_path, meta_path = Path("homevalue_model.joblib"), Path("model_meta.json")
    try:
        meta = json.loads(meta_path.read_text())
        if meta.get("sklearn_version") != sklearn.__version__:
            raise RuntimeError("scikit-learn version changed since the model was saved")
        return joblib.load(model_path), meta
    except Exception:
        with st.spinner("Preparing the valuation model for this computer (about a minute, first run only)..."):
            from train import train
            train(n_iter=8, verbose=False)
        return joblib.load(model_path), json.loads(meta_path.read_text())


model, meta = load_artifacts()


def money(usd, currency, rate):
    if currency == "USD":
        return f"${usd:,.0f}"
    inr = usd * rate
    return f"₹{inr / 1e7:,.2f} Cr" if inr >= 1e7 else f"₹{inr / 1e5:,.1f} Lakh"


def style_axes(ax):
    ax.set_facecolor("none")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color((1, 1, 1, .25))
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    ax.grid(color="white", alpha=.07)


def new_fig(w=5.2, h=3.2):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_alpha(0)
    style_axes(ax)
    return fig, ax


# ---------------------------------------------------------------- nav + hero
st.markdown(f"""
<div class="nav">
  <div class="logo"><span class="logo-mark">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#0A0F1F" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M3 11l9-8 9 8"/><path d="M5 10v10h14V10"/><path d="M10 20v-6h4v6"/></svg>
  </span>HomeValue<span class="grad">.ai</span></div>
  <div class="pill">Model accuracy <b>R² {meta['r2']:.2f}</b> on unseen homes</div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1.02], gap="large")

with left:
    st.markdown("""
    <div class="hero">
      <h1>Know what a home is <span class="grad">really worth.</span></h1>
      <p>Tell us about the home and its neighbourhood. Our machine-learning model returns an instant
      valuation with an honest price range, not one over-confident number.</p>
    </div>""", unsafe_allow_html=True)

    with st.container(key="panel"):
        st.markdown('<div class="panel-title">Describe the home</div>', unsafe_allow_html=True)
        a, b = st.columns([1.3, 1])
        area = a.selectbox("Nearest city", list(CITY_CENTERS) + ["Custom location"])
        currency = b.radio("Show price in", ["USD", "INR"], horizontal=True)
        income = st.slider("Neighbourhood median income (USD / year)", 10_000, 150_000, 55_000, 1_000)
        c, d = st.columns(2)
        age = c.slider("House age (years)", 1, 52, 25)
        rooms = d.slider("Average rooms per home", 2.0, 10.0, 5.5, 0.1)
        e, f = st.columns(2)
        bedrooms = e.slider("Average bedrooms per home", 0.8, 3.0, 1.1, 0.1)
        occ = f.slider("Average household size", 1.0, 6.0, 3.0, 0.1)
        pop = st.slider("People living in the block", 100, 10_000, 1_400, 50)
        with st.expander("Fine-tune location and currency rate"):
            lat0, lon0 = CITY_CENTERS.get(area, (36.5, -119.5))
            g, h = st.columns(2)
            lat = g.number_input("Latitude", 32.5, 42.0, float(lat0), 0.01, key=f"lat_{area}")
            lon = h.number_input("Longitude", -124.5, -114.0, float(lon0), 0.01, key=f"lon_{area}")
            rate = st.number_input("INR per USD", 60.0, 120.0, 88.0, 0.5, disabled=(currency == "USD"))

row = pd.DataFrame([[income / 10_000, age, rooms, bedrooms, pop, occ, lat, lon]], columns=RAW_COLS)
pred = float(model.predict(row)[0]) * UNIT
low, high = pred * meta["ratio_q10"], pred * meta["ratio_q90"]
pin = min(max(100 * (pred - low) / max(high - low, 1), 4), 96)
median_price = meta.get("median_price", 179_700)
diff = 100 * (pred / median_price - 1)
per_room = pred / rooms

with right:
    st.markdown(f"""
    <div class="result">
      <div class="eyebrow">Estimated market value</div>
      <div class="price">{money(pred, currency, rate)}</div>
      <div class="track"><div class="fill"></div><div class="dot" style="left:{pin:.1f}%"></div></div>
      <div class="track-labels">
        <div>{money(low, currency, rate)}<small>Low estimate</small></div>
        <div>{money(high, currency, rate)}<small>High estimate</small></div>
      </div>
      <div class="note">Likely range: 8 in 10 similar homes sold inside it during testing.</div>
    </div>
    <div class="tiles">
      <div class="tile"><div class="k">vs. typical home</div>
        <div class="v {'up' if diff >= 0 else 'down'}">{'+' if diff >= 0 else ''}{diff:.0f}%</div></div>
      <div class="tile"><div class="k">Value per room</div><div class="v">{money(per_room, currency, rate)}</div></div>
      <div class="tile"><div class="k">Location</div><div class="v">{area.replace(' location', '')}</div></div>
    </div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------- insights
st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["Price drivers", "Location", "About the model"])
scale = rate if currency == "INR" else 1


def sweep(col, values):
    frame = pd.concat([row] * len(values), ignore_index=True)
    frame[col] = values
    return model.predict(frame) * UNIT * scale


def curve(ax, x, y, cur_x, cur_y, xlabel):
    ax.plot(x, y, color=TEAL, lw=2.6)
    ax.fill_between(x, y, y.min() * 0.97, color=TEAL, alpha=.14)
    ax.scatter([cur_x], [cur_y], s=90, color=PINK, edgecolor="white", linewidth=1.6, zorder=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Estimate ({currency})")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}k"))


with tab1:
    st.caption("How the estimate moves as you change one thing and keep everything else the same. "
               "The pink dot is your current home.")
    c1, c2, c3 = st.columns(3)
    xs = np.arange(10_000, 150_001, 5_000)
    with c1:
        fig, ax = new_fig(4.2, 3.0); curve(ax, xs, sweep("MedInc", xs / 10_000), income, pred * scale, "Neighbourhood income (USD / year)")
        ax.set_title("Income", color="white", fontsize=11, fontweight="bold", loc="left"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with c2:
        ys = np.arange(1, 53)
        fig, ax = new_fig(4.2, 3.0); curve(ax, ys, sweep("HouseAge", ys.astype(float)), age, pred * scale, "House age (years)")
        ax.set_title("House age", color="white", fontsize=11, fontweight="bold", loc="left"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with c3:
        rs = np.round(np.arange(2.0, 10.01, 0.25), 2)
        fig, ax = new_fig(4.2, 3.0); curve(ax, rs, sweep("AveRooms", rs), rooms, pred * scale, "Average rooms")
        ax.set_title("Rooms", color="white", fontsize=11, fontweight="bold", loc="left"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

with tab2:
    fig, ax = new_fig(8.2, 4.6)
    for name, (la, lo) in CITY_CENTERS.items():
        ax.scatter(lo, la, s=60, color=INDIGO, zorder=3)
        ax.annotate(name, (lo, la), xytext=(8, 6), textcoords="offset points", fontsize=9, color=TEXT)
    ax.scatter(lon, lat, s=260, color=PINK, alpha=.25, zorder=4)
    ax.scatter(lon, lat, s=110, marker="D", color=PINK, edgecolor="white", linewidth=1.6, zorder=5)
    ax.set(xlim=(-124.8, -113.8), ylim=(32.0, 42.3), xlabel="Longitude", ylabel="Latitude")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.caption("The pink diamond is the location you selected. Homes closer to the big coastal cities are priced higher.")

with tab3:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model", meta["model_name"])
    m2.metric("R² on unseen data", f"{meta['r2']:.2f}")
    m3.metric("Typical error (MAE)", money(meta["mae"] * UNIT, currency, rate))
    m4.metric("Trained on", f"{meta['n_train']:,} blocks")
    st.caption(f"Data source: {meta['data_source']}. The model predicts for a neighbourhood block, "
               "using its average income, home age, rooms and location.")

st.markdown('<div class="foot">HomeValue.ai gives statistical guidance, not a formal appraisal. '
            'Always confirm with a licensed valuer before making financial decisions.</div>', unsafe_allow_html=True)
