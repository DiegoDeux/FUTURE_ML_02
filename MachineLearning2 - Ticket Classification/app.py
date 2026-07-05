"""
TicketIQ — IT Support Ticket Classification Dashboard

Pages:
  1. Overview        — KPIs, dataset summary, pipeline explanation
  2. Live Classifier — type a ticket and get instant prediction
  3. Analytics       — charts and data exploration
  4. Model Performance — confusion matrix, metrics, model comparison
  5. Word Clouds     — most common words per category
  6. Batch Processor — classify many tickets at once

"""

import re                    # text cleaning
import string                # punctuation removal
import warnings              # suppress warnings
import os                    # file paths

warnings.filterwarnings("ignore")

import joblib                # load saved models
import numpy as np           # number arrays
import pandas as pd          # data handling
import plotly.graph_objects as go          # interactive charts
from plotly.subplots import make_subplots  # side by side charts
import streamlit as st                     # dashboard framework
from wordcloud import WordCloud            # word cloud images
import matplotlib                          # for rendering word clouds
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
for r in ["stopwords", "wordnet", "punkt", "punkt_tab"]:
    nltk.download(r, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# Path to the models/ folder (sits next to app.py in the root)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


# ══════════════════════════════════════════════════════════════
# PAGE CONFIG — must be the very first Streamlit call
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="TicketIQ · IT Support Classifier",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════
# CUSTOM CSS — dark glassmorphism theme
# ══════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-main:   #0d1117;
    --bg-card:   #161b22;
    --bg-card2:  #1c2230;
    --blue:      #3b82f6;
    --purple:    #8b5cf6;
    --teal:      #14b8a6;
    --pink:      #ec4899;
    --amber:     #f59e0b;
    --green:     #10b981;
    --red:       #f87171;
    --text:      #e6edf3;
    --muted:     #8b949e;
    --border:    rgba(255,255,255,0.06);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-main) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem !important; max-width: 1400px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%) !important;
    border-right: 1px solid var(--border) !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-size: 1.6rem !important; font-weight: 700 !important; }

/* Reusable card */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}

/* Section headers */
.section-header {
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 1rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid var(--blue);
    display: inline-block;
}

/* Hero title gradient */
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.25rem;
}
.hero-sub {
    color: var(--muted);
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

/* Prediction result box */
.prediction-box {
    background: linear-gradient(135deg, #1c2230, #161b22);
    border: 1px solid var(--blue);
    border-radius: 16px;
    padding: 1.75rem;
    text-align: center;
    box-shadow: 0 0 20px rgba(59,130,246,0.3);
}

/* Text area */
.stTextArea textarea {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--blue), var(--purple)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.3) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(59,130,246,0.5) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 10px !important;
    padding: 0.3rem !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: var(--muted) !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--blue), var(--purple)) !important;
    color: white !important;
}

/* Selectbox */
[data-baseweb="select"] > div {
    background: var(--bg-card2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-main); }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# CONSTANTS — colours for each category
# ══════════════════════════════════════════════════════════════

CATEGORY_COLORS = {
    "Fileservice":       "#3b82f6",   # blue
    "Support general":   "#8b5cf6",   # purple
    "Software":          "#14b8a6",   # teal
    "O365":              "#f59e0b",   # amber
    "Active Directory":  "#ec4899",   # pink
    "Computer-Services": "#10b981",   # green
    "EOL":               "#f87171",   # red
}

# Base layout settings shared by all Plotly charts
PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",   # transparent background
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e6edf3", family="Inter"),
)


# ══════════════════════════════════════════════════════════════
# TEXT CLEANING
# Must be identical to train_models.py so predictions work correctly
# ══════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Clean raw ticket text the same way as during training."""
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", " ", text)              # remove [NAME] [TICKET ID]
    text = re.sub(r"http\S+|www\.\S+", " ", text)     # remove URLs
    text = re.sub(r"\S+@\S+", " ", text)              # remove emails
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in text.split()
        if tok not in STOP_WORDS and len(tok) > 2
    ]
    return " ".join(tokens)


# ══════════════════════════════════════════════════════════════
# LOAD MODELS AND DATA
# @st.cache_resource means these only load ONCE per session
# not on every page click — keeps the dashboard fast
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading ML models …")
def load_models():
    """Load the trained model, encoder, and results from disk."""
    model   = joblib.load(os.path.join(MODEL_DIR, "best_category_model.pkl"))
    encoder = joblib.load(os.path.join(MODEL_DIR, "category_encoder.pkl"))
    results = joblib.load(os.path.join(MODEL_DIR, "results.pkl"))
    return model, encoder, results


@st.cache_data(show_spinner="Loading dataset …")
def load_data():
    """Load the processed ticket dataset for analytics."""
    return pd.read_csv(os.path.join(MODEL_DIR, "processed_tickets.csv"))


def predict_ticket(text, model, encoder):
    """
    Run a single raw ticket through the model.
    Returns the predicted category and confidence scores.
    """
    cleaned  = clean_text(text)
    idx      = model.predict([cleaned])[0]
    category = encoder.inverse_transform([idx])[0]

    # Try to get probability scores (LinearSVC doesn't support this natively)
    try:
        proba = model.predict_proba([cleaned])[0]
    except AttributeError:
        # If no predict_proba, give 100% confidence to predicted class
        proba = np.zeros(len(encoder.classes_))
        proba[idx] = 1.0

    return category, proba


# ══════════════════════════════════════════════════════════════
# CHART FUNCTIONS
# Each function builds and returns one Plotly or Matplotlib figure
# ══════════════════════════════════════════════════════════════

def make_donut(data: dict, title: str) -> go.Figure:
    """Donut chart showing distribution of ticket categories."""
    labels = list(data.keys())
    values = list(data.values())
    colors = [CATEGORY_COLORS.get(l, "#60a5fa") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color="#0d1117", width=2)),
        textinfo="percent",
        textfont=dict(size=12, color="#e6edf3"),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e6edf3"), x=0.5),
        showlegend=True,
        legend=dict(font=dict(color="#8b949e", size=10), bgcolor="rgba(0,0,0,0)"),
        **PLOTLY_BASE,
        height=320,
        margin=dict(t=50, b=10, l=10, r=10),
        annotations=[dict(
            text=f"<b>{sum(values)}</b><br>tickets",
            x=0.5, y=0.5,
            font=dict(size=15, color="#e6edf3"),
            showarrow=False,
        )],
    )
    return fig


def make_bar_chart(series: pd.Series, title: str) -> go.Figure:
    """Horizontal bar chart for ticket counts per category."""
    labels = list(series.index)
    values = list(series.values)
    colors = [CATEGORY_COLORS.get(l, "#60a5fa") for l in labels]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker=dict(color=colors),
        text=[f"  {v}" for v in values],
        textposition="outside",
        textfont=dict(color="#8b949e", size=11),
        hovertemplate="<b>%{y}</b>: %{x} tickets<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e6edf3"), x=0),
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(tickfont=dict(size=11), gridcolor="rgba(255,255,255,0.06)"),
        **PLOTLY_BASE,
        height=300,
        margin=dict(t=40, b=10, l=10, r=60),
    )
    return fig


def make_confusion_matrix(cm, labels, title) -> go.Figure:
    """
    Heatmap showing where the model makes mistakes.
    Rows = actual category, Columns = predicted category.
    Numbers on the diagonal = correct predictions.
    Numbers off the diagonal = mistakes.
    """
    z    = np.array(cm)
    text = [[str(z[i][j]) for j in range(len(labels))] for i in range(len(labels))]

    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        colorscale=[
            [0,   "rgba(59,130,246,0.05)"],
            [0.5, "rgba(59,130,246,0.5)"],
            [1,   "#3b82f6"],
        ],
        showscale=False,
        hovertemplate="Actual: <b>%{y}</b><br>Predicted: <b>%{x}</b><br>Count: <b>%{z}</b><extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#e6edf3"), x=0.5),
        xaxis=dict(title="Predicted", tickfont=dict(size=9), tickangle=-30, gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="Actual",    tickfont=dict(size=9), autorange="reversed", gridcolor="rgba(0,0,0,0)"),
        **PLOTLY_BASE,
        height=400,
        margin=dict(t=50, b=100, l=130, r=20),
    )
    return fig


def make_model_comparison(results) -> go.Figure:
    """Grouped bar chart comparing all 4 models side by side."""
    names = [r["name"] for r in results]
    accs  = [r["accuracy"]    for r in results]
    f1s   = [r["f1_weighted"] for r in results]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Accuracy",
        x=names, y=accs,
        marker_color="#3b82f6",
        text=[f"{v:.3f}" for v in accs],
        textposition="outside",
        textfont=dict(color="#8b949e", size=11),
    ))
    fig.add_trace(go.Bar(
        name="F1 (weighted)",
        x=names, y=f1s,
        marker_color="#8b5cf6",
        text=[f"{v:.3f}" for v in f1s],
        textposition="outside",
        textfont=dict(color="#8b949e", size=11),
    ))
    fig.update_layout(
        barmode="group",
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", showline=False),
        yaxis=dict(range=[0, 1.12], gridcolor="rgba(255,255,255,0.06)", showline=False),
        **PLOTLY_BASE,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)", font=dict(color="#8b949e"),
        ),
        height=340,
        margin=dict(t=40, b=40, l=10, r=10),
    )
    return fig


def make_word_count_hist(df) -> go.Figure:
    """Overlapping histogram of word counts per category."""
    fig = go.Figure()
    for cat, color in CATEGORY_COLORS.items():
        sub = df[df["category"] == cat]["word_count"]
        if len(sub) == 0:
            continue
        fig.add_trace(go.Histogram(
            x=sub, name=cat,
            marker_color=color,
            opacity=0.65,
            nbinsx=25,
            hovertemplate=f"<b>{cat}</b><br>Words: %{{x}}<br>Count: %{{y}}<extra></extra>",
        ))
    fig.update_layout(
        barmode="overlay",
        xaxis=dict(title="Word Count", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Frequency",  gridcolor="rgba(255,255,255,0.06)"),
        **PLOTLY_BASE,
        legend=dict(font=dict(color="#8b949e", size=10), bgcolor="rgba(0,0,0,0)"),
        height=300,
        margin=dict(t=10, b=40, l=10, r=10),
    )
    return fig


def make_wordcloud(text_series, title) -> plt.Figure:
    """Word cloud image for a given set of ticket texts."""
    combined = " ".join(text_series.dropna().tolist())
    if not combined.strip():
        return None

    wc = WordCloud(
        width=700, height=300,
        background_color=None,
        mode="RGBA",
        colormap="cool",
        max_words=60,
        stopwords=STOP_WORDS,
        prefer_horizontal=0.8,
        collocations=False,
    ).generate(combined)

    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_alpha(0)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, color="#e6edf3", fontsize=12, fontweight="bold", pad=8)
    fig.tight_layout(pad=0.1)
    return fig


def show_confidence_bars(proba, labels):
    """
    Renders a visual confidence bar for each category.
    Shows how confident the model is for each possible class.
    """
    for label, prob in zip(labels, proba):
        color = CATEGORY_COLORS.get(label, "#60a5fa")
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem;">
            <div style="width:155px; font-size:0.8rem; color:#8b949e;">{label}</div>
            <div style="flex:1; background:rgba(255,255,255,0.05); border-radius:999px; height:8px;">
                <div style="width:{prob*100:.1f}%; background:{color}; border-radius:999px; height:8px;"></div>
            </div>
            <div style="width:45px; text-align:right; font-size:0.82rem; color:{color}; font-weight:600;">{prob:.1%}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 1.5rem;'>
        <div style='font-size:2.5rem; margin-bottom:0.3rem;'>🎫</div>
        <div style='font-size:1.2rem; font-weight:800;
                    background:linear-gradient(135deg,#60a5fa,#a78bfa);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            TicketIQ
        </div>
        <div style='font-size:0.75rem; color:#8b949e; margin-top:0.2rem;'>
            IT Support Classifier
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🏠  Overview",
        "🔍  Live Classifier",
        "📊  Analytics",
        "🧠  Model Performance",
        "☁️  Word Clouds",
        "📤  Batch Processor",
    ], label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem; color:#8b949e; line-height:1.9;'>
        <b style='color:#e6edf3;'>Dataset</b><br>
        Zenodo IT Support Tickets<br>
        2,229 real tickets · 7 categories<br><br>
        <b style='color:#e6edf3;'>Best Model</b><br>
        Linear SVM · 80.1% F1<br><br>
        <b style='color:#e6edf3;'>Pipeline</b><br>
        NLTK → TF-IDF → LinearSVC<br><br>
        <b style='color:#e6edf3;'>Internship</b><br>
        Future Interns · ML Task 2 · 2026
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LOAD EVERYTHING
# ══════════════════════════════════════════════════════════════

model, encoder, results = load_models()
df          = load_data()
labels      = results["category"]["labels"]
ds_stats    = results["dataset_stats"]
cat_results = results["category"]["results"]
best_name   = results["category"]["best_model_name"]
best_res    = next(r for r in cat_results if r["name"] == best_name)


# ══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════

if page == "🏠  Overview":
    st.markdown('<div class="hero-title">IT Support Ticket Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Automatic classification of real IT support tickets · 2026</div>', unsafe_allow_html=True)

    # ── KPI metrics row ───────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.metric("Total Tickets",    f"{ds_stats['total']:,}")
    with k2: st.metric("Training Set",     f"{ds_stats['train']:,}")
    with k3: st.metric("Test Set",         f"{ds_stats['test']:,}")
    with k4: st.metric("Categories",       len(labels))
    with k5: st.metric("Avg Word Count",   ds_stats["avg_word_count"])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Best model + dataset source cards ─────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="glass-card" style="border-color:rgba(59,130,246,0.4); box-shadow:0 0 20px rgba(59,130,246,0.15);">
            <div style="color:#8b949e; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;">Best Model</div>
            <div style="font-size:1.4rem; font-weight:700; color:#60a5fa;">{best_res['name']}</div>
            <div style="margin-top:0.75rem; display:flex; gap:1.5rem;">
                <div>
                    <span style="font-size:1.6rem; font-weight:800;">{best_res['accuracy']:.1%}</span><br>
                    <span style="color:#8b949e; font-size:0.78rem;">Accuracy</span>
                </div>
                <div>
                    <span style="font-size:1.6rem; font-weight:800;">{best_res['f1_weighted']:.1%}</span><br>
                    <span style="color:#8b949e; font-size:0.78rem;">F1 Score</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="glass-card" style="border-color:rgba(20,184,166,0.4); box-shadow:0 0 20px rgba(20,184,166,0.15);">
            <div style="color:#8b949e; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;">Dataset Source</div>
            <div style="font-size:1.1rem; font-weight:700; color:#2dd4bf;">Zenodo Open Repository</div>
            <div style="margin-top:0.5rem; color:#8b949e; font-size:0.85rem; line-height:1.7;">
                2,229 real human-classified IT support tickets<br>
                7 categories · Pre-split train/test sets<br>
                <span style="color:#2dd4bf;">zenodo.org/records/7648117</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Category donut + legend ────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    d1, d2 = st.columns([1.3, 0.7])
    with d1:
        st.plotly_chart(
            make_donut(ds_stats["category_dist"], "Ticket Distribution by Category"),
            use_container_width=True, key="overview_donut"
        )
    with d2:
        st.markdown('<div class="section-header">The 7 IT Categories</div>', unsafe_allow_html=True)
        descriptions = {
            "Fileservice":       "File share access and permissions",
            "Support general":   "General helpdesk requests",
            "Software":          "Software installation and issues",
            "O365":              "Microsoft Office 365 problems",
            "Active Directory":  "User accounts and AD management",
            "Computer-Services": "Hardware and device services",
            "EOL":               "End-of-life system migrations",
        }
        for cat, desc in descriptions.items():
            color = CATEGORY_COLORS.get(cat, "#60a5fa")
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.55rem;">
                <div style="width:10px; height:10px; border-radius:50%; background:{color}; flex-shrink:0;"></div>
                <div>
                    <span style="font-weight:600; color:#e6edf3; font-size:0.83rem;">{cat}</span>
                    <span style="color:#8b949e; font-size:0.78rem;"> — {desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Pipeline steps ─────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">How the Pipeline Works</div>', unsafe_allow_html=True)
    steps = [
        ("📥", "Load Data",    "4 CSV files from Zenodo — train/test split already done"),
        ("🧹", "Clean Text",   "Remove placeholders, stopwords, punctuation, lemmatize"),
        ("📐", "TF-IDF",       "Convert words → numbers using bigram TF-IDF (15k vocab)"),
        ("🤖", "Train",        "4 classifiers benchmarked — best picked by F1 score"),
        ("✅", "Predict",      "Linear SVM classifies any ticket into 1 of 7 categories"),
    ]
    cols = st.columns(5)
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; min-height:135px;">
                <div style="font-size:1.6rem; margin-bottom:0.4rem;">{icon}</div>
                <div style="font-weight:700; color:#e6edf3; font-size:0.88rem; margin-bottom:0.4rem;">{title}</div>
                <div style="font-size:0.76rem; color:#8b949e;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — LIVE CLASSIFIER
# ══════════════════════════════════════════════════════════════

elif page == "🔍  Live Classifier":
    st.markdown('<div class="hero-title">Live Ticket Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Type or paste any IT support ticket and get an instant category prediction.</div>', unsafe_allow_html=True)

    # Sample tickets the user can try
    SAMPLES = {
        "🗂️  File access request":
            "File Share Access - John Smith needs read access to the Marketing shared drive. Please grant permissions.",
        "💻  Software install":
            "User cannot install Adobe Acrobat on their laptop. Getting admin permission error when trying to install.",
        "📧  O365 issue":
            "Outlook is not syncing emails since this morning. User is on Office 365 Business. Already tried restarting.",
        "👤  Active Directory":
            "New employee Sarah Jones starting Monday. Please create her AD account and add her to the Finance group.",
        "🖥️  Computer service":
            "My laptop screen is flickering badly and two keyboard keys have stopped working. Need a replacement.",
        "⚠️  EOL migration":
            "Several workstations are still running Windows 7. Need to migrate to Windows 11 before the deadline.",
        "🆘  General support":
            "Cannot connect to the company VPN from home. Tried reinstalling the client but still getting timeout errors.",
    }

    sample_choice = st.selectbox(
        "Try a sample ticket",
        ["— type your own below —"] + list(SAMPLES.keys())
    )
    default_text = SAMPLES.get(sample_choice, "")

    ticket_text = st.text_area(
        "Ticket text",
        value=default_text,
        height=130,
        placeholder="Paste any IT support ticket here …",
        label_visibility="collapsed",
    )

    classify_btn = st.button("🔍  Classify Ticket")

    if classify_btn and ticket_text.strip():
        with st.spinner("Analysing ticket …"):
            category, proba = predict_ticket(ticket_text, model, encoder)

        st.markdown("<br>", unsafe_allow_html=True)
        color = CATEGORY_COLORS.get(category, "#60a5fa")

        # ── Prediction result ──────────────────────────────────
        st.markdown(f"""
        <div class="prediction-box" style="border-color:{color}; box-shadow:0 0 25px {color}33;">
            <div style="color:#8b949e; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.75rem;">
                Predicted Category
            </div>
            <div style="font-size:2rem; font-weight:800; color:{color}; margin-bottom:0.4rem;">
                {category}
            </div>
            <div style="color:#8b949e; font-size:0.9rem;">
                Model confidence: {max(proba):.1%}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Confidence breakdown ───────────────────────────────
        st.markdown('<div class="section-header">Confidence per Category</div>', unsafe_allow_html=True)
        show_confidence_bars(proba, labels)

        # ── Show cleaned text ──────────────────────────────────
        with st.expander("🔬  View preprocessed text (what the model actually sees)"):
            st.code(clean_text(ticket_text), language="text")

    elif classify_btn:
        st.warning("Please enter some ticket text first.")


# PAGE 3 — ANALYTICS

elif page == "📊  Analytics":
    st.markdown('<div class="hero-title">Dataset Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Exploratory analysis of the real Zenodo IT support ticket dataset.</div>', unsafe_allow_html=True)

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Total Tickets",   f"{len(df):,}")
    with k2: st.metric("Categories",      df["category"].nunique())
    with k3: st.metric("Training Set",    f"{ds_stats['train']:,}")
    with k4: st.metric("Test Set",        f"{ds_stats['test']:,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Bar chart + donut side by side
    a1, a2 = st.columns(2)
    with a1:
        st.plotly_chart(
            make_bar_chart(df["category"].value_counts(), "Tickets per Category"),
            use_container_width=True, key="analytics_bar"
        )
    with a2:
        st.plotly_chart(
            make_donut(df["category"].value_counts().to_dict(), "Category Share"),
            use_container_width=True, key="analytics_donut"
        )

    # Word count histogram
    st.markdown('<div class="section-header">Word Count Distribution by Category</div>', unsafe_allow_html=True)
    st.plotly_chart(make_word_count_hist(df), use_container_width=True, key="wc_hist")

    # Data preview table
    st.markdown('<div class="section-header">Sample Tickets</div>', unsafe_allow_html=True)
    n = st.slider("Rows to show", 5, 50, 10)
    st.dataframe(
        df[["id", "text", "category", "word_count"]].head(n),
        use_container_width=True,
        hide_index=True
    )


# ══════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════

elif page == "🧠  Model Performance":
    st.markdown('<div class="hero-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">4 classifiers compared on the Zenodo test set (657 real tickets).</div>', unsafe_allow_html=True)

    st.markdown(f"**Best model selected:** `{best_name}` — highest weighted F1 on the test set.")
    st.markdown("<br>", unsafe_allow_html=True)

    # Model comparison chart
    st.plotly_chart(
        make_model_comparison(cat_results),
        use_container_width=True, key="model_compare"
    )

    # Confusion matrix
    st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
    st.markdown("Rows = actual category · Columns = predicted · Diagonal = correct predictions")
    st.plotly_chart(
        make_confusion_matrix(
            best_res["confusion_matrix"],
            labels,
            f"Confusion Matrix — {best_name}"
        ),
        use_container_width=True, key="conf_matrix"
    )

    # Per-class metrics table
    st.markdown('<div class="section-header">Per-Class Metrics</div>', unsafe_allow_html=True)
    report = best_res["report"]
    rows = []
    for label in labels:
        m = report.get(label, {})
        rows.append({
            "Category":  label,
            "Precision": f"{m.get('precision', 0):.3f}",
            "Recall":    f"{m.get('recall',    0):.3f}",
            "F1 Score":  f"{m.get('f1-score',  0):.3f}",
            "Support":   int(m.get("support", 0)),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Summary table of all 4 models
    st.markdown('<div class="section-header">All Models Summary</div>', unsafe_allow_html=True)
    summary = []
    for r in cat_results:
        summary.append({
            "Model":         r["name"],
            "Accuracy":      f"{r['accuracy']:.3f}",
            "F1 (weighted)": f"{r['f1_weighted']:.3f}",
            "Selected":      "✅ Best" if r["name"] == best_name else "",
        })
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# PAGE 5 — WORD CLOUDS
# ══════════════════════════════════════════════════════════════

elif page == "☁️  Word Clouds":
    st.markdown('<div class="hero-title">Word Clouds by Category</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Most frequent terms in cleaned ticket text, broken down by IT category.</div>', unsafe_allow_html=True)

    # One tab per category plus an "All" tab
    tabs = st.tabs(["🌐 All"] + labels)

    with tabs[0]:
        fig = make_wordcloud(df["clean_text"], "All Tickets — Most Frequent Terms")
        if fig:
            st.pyplot(fig, use_container_width=True)

    for i, cat in enumerate(labels):
        with tabs[i + 1]:
            sub = df[df["category"] == cat]["clean_text"]
            fig = make_wordcloud(sub, f"{cat} — Most Frequent Terms")
            if fig:
                st.pyplot(fig, use_container_width=True)

            # Top 15 terms table underneath the word cloud
            from collections import Counter
            words = " ".join(sub.tolist()).split()
            top   = Counter(words).most_common(15)
            if top:
                st.dataframe(
                    pd.DataFrame(top, columns=["Term", "Count"]),
                    use_container_width=True,
                    hide_index=True
                )


# ══════════════════════════════════════════════════════════════
# PAGE 6 — BATCH PROCESSOR
# ══════════════════════════════════════════════════════════════

elif page == "📤  Batch Processor":
    st.markdown('<div class="hero-title">Batch Ticket Processor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Classify multiple tickets at once. Paste one per line or upload a CSV file.</div>', unsafe_allow_html=True)

    mode = st.radio("Input method", ["Paste text", "Upload CSV"], horizontal=True)

    if mode == "Paste text":
        bulk    = st.text_area("One ticket per line", height=200,
                               placeholder="Ticket 1 …\nTicket 2 …\nTicket 3 …")
        tickets = [t.strip() for t in bulk.strip().split("\n") if t.strip()]

    else:
        uploaded = st.file_uploader("Upload a CSV file with a 'text' column", type=["csv"])
        if uploaded:
            up_df = pd.read_csv(uploaded)
            if "text" in up_df.columns:
                tickets = up_df["text"].dropna().tolist()
                st.success(f"Loaded {len(tickets):,} tickets from CSV.")
            else:
                st.error("Your CSV must have a column called 'text'.")
                tickets = []
        else:
            tickets = []

    if st.button("⚡  Classify All Tickets", disabled=not tickets):
        with st.spinner(f"Classifying {len(tickets)} tickets …"):
            rows = []
            for i, txt in enumerate(tickets):
                cat, proba = predict_ticket(txt, model, encoder)
                rows.append({
                    "Ticket #":   i + 1,
                    "Text":       txt[:80] + ("…" if len(txt) > 80 else ""),
                    "Category":   cat,
                    "Confidence": f"{max(proba):.1%}",
                })

        result_df = pd.DataFrame(rows)
        st.success(f"✅ Done! Classified {len(result_df)} tickets.")
        st.markdown("<br>", unsafe_allow_html=True)

        # Summary metrics
        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Total Processed",    len(result_df))
        with m2: st.metric("Most Common Category", result_df["Category"].mode()[0])
        with m3: st.metric("Unique Categories",  result_df["Category"].nunique())

        st.markdown("<br>", unsafe_allow_html=True)

        # Donut of results
        st.plotly_chart(
            make_donut(result_df["Category"].value_counts().to_dict(), "Batch — Category Breakdown"),
            use_container_width=True, key="batch_donut"
        )

        # Results table
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        # Download button
        st.download_button(
            label="⬇️  Download Results as CSV",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="classified_tickets.csv",
            mime="text/csv",
        )
