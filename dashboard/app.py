"""
Telco ABSA dashboard  --  3 pages: Dashboard, Review Analyser, Performance.

The model behind every number: the SVM trained ONLY on my own hand labels (no auto-labeler).

First make the data (once):      python3 preprocessing/build_dashboard_data.py
Then run, from telco-absa:       streamlit run dashboard/app.py
Install once:                    pip install streamlit plotly wordcloud joblib
"""
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "dashboard"
sys.path.insert(0, str(ROOT / "preprocessing"))

try:
    from wordcloud import WordCloud
    HAVE_WORDCLOUD = True
except ImportError:
    HAVE_WORDCLOUD = False

st.set_page_config(page_title="Telco ABSA Dashboard", layout="wide", initial_sidebar_state="expanded")

INK, PAPER, LINE = "#111111", "#F2F2F0", "#DEDEDA"
NEG, NEU, POS = "#D0312D", "#C4C4C0", "#14877B"
SENT_COLOUR = {"negative": NEG, "neutral": NEU, "positive": POS}
SENT_TEXT = {"negative": "#FFFFFF", "neutral": INK, "positive": "#FFFFFF"}
CLASSES = ["negative", "neutral", "positive"]
ASPECTS = {"app_performance": "App performance", "service_reliability": "Service reliability",
           "customer_service": "Customer service", "none": "General (no aspect)"}
VERSION_LABEL = {"manglish": "Manglish", "malay": "Malay", "english": "English"}
VERSION_NOTE = {"manglish": "Cleaned and normalized text. The original wording is kept.",
                "english": "Normalized text, then the dictionary puts the meaning in English.",
                "malay": "Normalized text, then the dictionary puts the meaning in Malay."}
TEXT_KEY = {"manglish": "normalized", "english": "english", "malay": "malay"}
# brand colours: name colour, stripe under the name
BRAND = {"CelcomDigi": ("#0B57D0", "linear-gradient(90deg,#0B57D0 0 72%,#FFC72C 72% 100%)"),
         "Maxis": ("#00A04A", "#00A04A"),
         "U Mobile": ("#F26B0F", "#F26B0F")}
TELCO_ORDER = ["CelcomDigi", "Maxis", "U Mobile"]
EXAMPLES = ["Aplikasi slow gila tapi customer service helpful. Line selalu putus la",
            "Cant login, otp x masuk2. Network kt umah teruk but the app UI is nice"]
PAGES = ["Dashboard", "Review Analyser", "Performance"]
STOP = set("""a an the and or but is are was were be been am i me my we you your he she it they them this that these those to of in on at
for from with by as so if then than too very just not no yes can will would could should do does did done have has had get got
la lah je jer lah kan ke ni tu tak x yg yang dan atau di ke pada untuk dengan dari saya aku kita kami dia ada nak mahu boleh
sudah dah lagi pun juga lah kat kt sangat ok app apps application aplikasi celcom digi celcomdigi maxis umobile mobile u
also all any more much many one two use used using like even still really only out up down over back please""".split())

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=DM+Sans:wght@400;500;700&display=swap');
html, body, [class*="st-"], .stApp {font-family:'DM Sans',system-ui,sans-serif;}
.stApp {background:@PAPER@; color:@INK@;}
h1,h2,h3,h4 {font-family:'Bricolage Grotesque','DM Sans',sans-serif !important; color:@INK@ !important; letter-spacing:-0.02em;}
.block-container {padding-top:2.2rem; max-width:1280px;}
header[data-testid="stHeader"] {background:transparent;}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {background:#0C0C0C; border-right:1px solid #222;}
[data-testid="stSidebar"] .brand {font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:1.55rem; color:#fff; line-height:1.05; letter-spacing:-0.03em;}
[data-testid="stSidebar"] .brand small {display:block; font-family:'DM Sans',sans-serif; font-weight:400; font-size:.8rem; color:#9A9A96; letter-spacing:0; margin-top:6px;}
[data-testid="stSidebar"] .bars {display:flex; gap:4px; margin:14px 0 22px;}
[data-testid="stSidebar"] .bars i {height:4px; flex:1; border-radius:2px;}
[data-testid="stSidebar"] .note {color:#8C8C88; font-size:.78rem; line-height:1.45; margin-top:28px; border-top:1px solid #262626; padding-top:14px;}
[data-testid="stSidebar"] .stButton > button {width:100%; justify-content:flex-start; gap:.4rem; padding:.72rem 1rem; border-radius:12px;
    font-weight:700; font-size:1rem; transition:background .12s, color .12s;}
[data-testid="stSidebar"] button[kind="secondary"], [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {background:#171717; border:1px solid #2A2A2A; color:#E8E8E4;}
[data-testid="stSidebar"] button[kind="secondary"]:hover, [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {background:#242424; border-color:#444; color:#fff;}
[data-testid="stSidebar"] button[kind="primary"], [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {background:#FFFFFF; border:1px solid #FFFFFF; color:#111;}
[data-testid="stSidebar"] button p, [data-testid="stSidebar"] button span {color:inherit !important;}

/* ---------- page header ---------- */
.ptitle {font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:2.7rem; line-height:1.02; letter-spacing:-0.035em; margin:0 0 .35rem; color:@INK@;}
.psub {color:#5B5B57; font-size:1rem; max-width:62ch; line-height:1.5; margin-bottom:1.3rem;}

/* ---------- cards ---------- */
[class*="st-key-card"] {background:#FFFFFF; border:1px solid @LINE@; border-radius:18px; padding:1.25rem 1.3rem 1rem;}
.tname {font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:1.7rem; letter-spacing:-0.03em; line-height:1;}
.tbar {height:5px; border-radius:3px; margin:9px 0 14px; width:100%;}
.stat {display:inline-block; margin-right:30px;}
.stat b {display:block; font-family:'Bricolage Grotesque',sans-serif; font-size:2rem; font-weight:800; letter-spacing:-0.03em; line-height:1.05;}
.stat span {font-size:.82rem; color:#6B6B66;}
.legend {display:flex; gap:18px; font-size:.85rem; color:#44443F; margin:2px 0 12px;}
.legend i {display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:6px; vertical-align:-1px;}

/* ---------- pills ---------- */
.badge {display:inline-block; background:#111; color:#fff !important; border-radius:999px; padding:6px 22px; font-weight:700; margin-bottom:8px;}
.chip {display:inline-block; border-radius:999px; padding:3px 14px; font-weight:700; font-size:.9rem; color:@INK@ !important;}
.chip.pos, .chip.neg {color:#FFFFFF !important;}

/* ---------- widgets ---------- */
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:1.75rem !important; overflow:visible !important; text-overflow:clip !important;}
[data-testid="stMetric"] {background:#fff; border:1px solid @LINE@; border-radius:14px; padding:.7rem .9rem;}
div[data-testid="stForm"] {background:#fff; border:1px solid @LINE@; border-radius:18px; padding:1.1rem 1.2rem;}
div[data-testid="stForm"] button {background:#111; border:0; border-radius:12px; font-weight:700; padding:.65rem 1.2rem;}
div[data-testid="stForm"] button p {color:#fff !important;}
div[data-testid="stForm"] button:hover {background:#333;}
[data-baseweb="select"] > div, textarea {border-radius:12px !important;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:16px;}
:focus-visible {outline:2px solid #111 !important; outline-offset:2px;}
</style>
"""
st.markdown(CSS.replace("@PAPER@", PAPER).replace("@INK@", INK).replace("@LINE@", LINE), unsafe_allow_html=True)


def page_head(title, sub=""):
    st.markdown(f"<div class='ptitle'>{title}</div><div class='psub'>{sub}</div>", unsafe_allow_html=True)


_VER = tuple(int(x) for x in st.__version__.split(".")[:2])


def show(fig):
    """Draw a plotly chart at full column width, on both old and new Streamlit versions."""
    fig.update_layout(font=dict(family="DM Sans, sans-serif", color=INK), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    if _VER >= (1, 50):
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------ loading
@st.cache_data
def load_parts():
    df = pd.read_csv(DATA / "parts_dashboard.csv", dtype={"review_id": str})
    for c in ("text", "text_en", "telco", "source", "month", "aspect"):
        df[c] = df[c].fillna("").astype(str)
    return df


@st.cache_data
def load_perf():
    return json.loads((DATA / "performance.json").read_text())


@st.cache_resource
def load_models():
    return {v: joblib.load(DATA / f"svm_{v}.joblib") for v in VERSION_LABEL}


@st.cache_resource
def load_pipeline():
    import dictionary_mapper
    import segment_reviews as sr
    import show_pipeline as sp
    patterns = sr.compile_patterns(sr.filter_keywords(sr.load_keywords()))
    counts = Counter(joblib.load(DATA / "word_counts.joblib"))
    return sp, patterns, counts, dictionary_mapper.load_dictionary()


if not (DATA / "parts_dashboard.csv").exists():
    st.error("The dashboard data is not made yet. In the terminal, from the telco-absa folder, run:  "
             "python3 preprocessing/build_dashboard_data.py")
    st.stop()

perf = load_perf()
BEST = max(perf["versions"], key=lambda v: perf["versions"][v]["macro_f1"])
_neu = round(perf["versions"][BEST]["f1"]["neutral"])
CAUTION = (f"Neutral is the weakest class in my tests (F1 about {_neu} out of 100), so treat \"neutral\" with care. "
           f"The SVM learned from only {perf['n_labels']} hand labels.")

# ------------------------------------------------------------------ sidebar
if "page" not in st.session_state:
    st.session_state["page"] = "Dashboard"
with st.sidebar:
    st.markdown("<div class='brand'>Telco ABSA<small>Malaysian telco app reviews</small></div>"
                "<div class='bars'><i style='background:#0B57D0'></i><i style='background:#FFC72C'></i>"
                "<i style='background:#00A04A'></i><i style='background:#F26B0F'></i></div>", unsafe_allow_html=True)
    for name in PAGES:
        kw = {"width": "stretch"} if _VER >= (1, 50) else {"use_container_width": True}
        if st.button(name, key=f"nav_{name}", type="primary" if st.session_state["page"] == name else "secondary", **kw):
            st.session_state["page"] = name
            st.rerun()
    st.markdown(f"<div class='note'>Model: SVM trained only on my own {perf['n_labels']} hand labels. No auto-labeler.</div>",
                unsafe_allow_html=True)
page = st.session_state["page"]


def version_picker(key):
    names = list(VERSION_LABEL)
    return st.selectbox("Language version", names, index=names.index(BEST), key=key,
                        format_func=lambda v: VERSION_LABEL[v] + ("  (best in my tests)" if v == BEST else ""),
                        help="The SVM was trained once for each language version. The best one is chosen by Macro-F1.")


def stacked(df, group_col, order, labels, height=300, show_text=True, legend=True):
    """100% stacked horizontal bars: one row per group, split by predicted sentiment."""
    fig = go.Figure()
    counts = pd.crosstab(df[group_col], df["sent"]).reindex(index=order, columns=CLASSES, fill_value=0)
    totals = counts.sum(axis=1).replace(0, 1)
    for c in CLASSES:
        pct = (counts[c] / totals * 100).round(0)
        fig.add_bar(y=[labels[o] for o in order], x=pct, name=c.capitalize(), orientation="h", marker_color=SENT_COLOUR[c],
                    text=[f"{p:.0f}%" for p in pct] if show_text else None, textfont=dict(color=SENT_TEXT[c]),
                    customdata=counts[c], hovertemplate="%{y}: %{customdata} parts " + c + " (%{x}%)<extra></extra>")
    fig.update_layout(barmode="stack", height=height, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(range=[0, 100], ticksuffix="%", showgrid=False), yaxis=dict(autorange="reversed"),
                      showlegend=legend, legend=dict(orientation="h", y=-0.25))
    return fig


# ================================================================== PAGE 1
if page == "Dashboard":
    w = perf.get("window")
    sub = (f"{w.get('source', 'All sources')} reviews from {pd.Timestamp(w['from']):%d %B %Y} to {pd.Timestamp(w['to']):%d %B %Y}."
           if w else "How customers feel about each telco's app and network.")
    page_head("Service quality sentiment by telco", sub)
    df = load_parts()
    f1, f2, f3 = st.columns([1, 2, 1])
    with f1:
        sources = sorted(df["source"].unique())
        src = st.multiselect("Source", sources, default=sources)
    with f2:
        months = sorted(m for m in df["month"].unique() if m not in ("", "NaT", "nan"))
        m0, m1 = st.select_slider("Month range", options=months, value=(months[0], months[-1])) if len(months) > 1 else (months[0], months[0])
    with f3:
        ver = version_picker("ver_dash")

    d = df[df["source"].isin(src) & (df["month"] >= m0) & (df["month"] <= m1)].copy()
    d["sent"] = d[f"pred_{ver}"]
    d["aspect_one"] = d["aspect"].where(~d["aspect"].str.contains(r"\|"), "multi")
    one = d[d["aspect_one"].isin(ASPECTS)]
    st.caption(f"{len(d):,} parts shown, cut from the reviews of the selected source and months. "
               f"{len(d) - len(one):,} parts with two aspects count toward overall sentiment only, so they are left out of the aspect charts.")

    left, right = st.columns(2)
    with left.container(key="card_aspect"):
        st.subheader("Sentiment by aspect")
        show(stacked(one, "aspect_one", list(ASPECTS), ASPECTS, height=330))
        st.caption("Each bar is 100% of that aspect's parts. Hover to see the counts.")
    with right.container(key="card_words"):
        st.subheader("Most mentioned words")
        wsent = st.radio("In parts predicted as", CLASSES, index=0, horizontal=True, format_func=str.capitalize, key="wc_sent")
        words = Counter(w for t in d.loc[d["sent"] == wsent, "text_en"].str.lower()
                        for w in t.split() if len(w) > 2 and w not in STOP and not w.isdigit())
        top = dict(words.most_common(60))
        if not top:
            st.info("No words to show for this selection.")
        elif HAVE_WORDCLOUD:
            shades = {"negative": ["#D0312D", "#A52420", "#E5645F", "#7D1B18"], "neutral": ["#6E6E6A", "#8F8F8B", "#4A4A47", "#A9A9A5"],
                      "positive": ["#14877B", "#0E655C", "#3FAA9F", "#0A4A43"]}[wsent]
            wc = WordCloud(width=800, height=330, background_color="white", prefer_horizontal=1.0,
                           color_func=lambda *a, **k: shades[hash(a[0]) % len(shades)]).generate_from_frequencies(top)
            st.image(wc.to_array())
        else:
            st.bar_chart(pd.Series(top).head(20).sort_values())
        st.caption("Words are shown in their English-dictionary form. Size = how often the word appears.")

    st.subheader("By telco")
    st.markdown("<div class='legend'>" + "".join(f"<span><i style='background:{SENT_COLOUR[c]}'></i>{c.capitalize()}</span>" for c in CLASSES)
                + "<span style='color:#77776F'>Bars show each aspect's share of parts.</span></div>", unsafe_allow_html=True)
    cols = st.columns(3)
    for col, telco in zip(cols, [t for t in TELCO_ORDER if t in set(d["telco"])] + [t for t in sorted(set(d["telco"])) if t not in TELCO_ORDER]):
        with col.container(key="card_" + telco.lower().replace(" ", "")):
            colour, bar = BRAND.get(telco, (INK, INK))
            t = d[d["telco"] == telco]
            ot = one[one["telco"] == telco]
            neg = (t["sent"] == "negative").mean() * 100 if len(t) else 0
            st.markdown(f"<div class='tname' style='color:{colour}'>{telco}</div><div class='tbar' style='background:{bar}'></div>"
                        f"<div class='stat'><b>{len(t):,}</b><span>Parts</span></div>"
                        f"<div class='stat'><b style='color:{NEG}'>{neg:.0f}%</b><span>Negative</span></div>", unsafe_allow_html=True)
            show(stacked(ot, "aspect_one", list(ASPECTS), ASPECTS, height=250, show_text=False, legend=False))
    st.caption(CAUTION + " The hand-labelled parts used for training are included (a very small share of all parts).")

# ================================================================== PAGE 2
elif page == "Review Analyser":
    page_head("Review Analyser", "Type a telco review. The system cuts it into parts, finds the aspect of each part, and predicts the sentiment.")
    sp, patterns, counts, lookup = load_pipeline()
    models = load_models()

    def use_example():
        if st.session_state.get("ex"):
            st.session_state["review_text"] = st.session_state["ex"]

    top1, top2 = st.columns([3, 1])
    with top2:
        ver = version_picker("ver_an")
    with top1:
        st.selectbox("Or pick an example (invented, not from the data)", [""] + EXAMPLES, key="ex", on_change=use_example)
    with st.form("analyse"):
        a, b = st.columns([6, 1.3])
        text = a.text_area("Enter telco-related text:", key="review_text", height=110)
        b.markdown("<div style='height:2.2rem'></div>", unsafe_allow_html=True)
        b.form_submit_button("Analyse  ➜")

    if text.strip():
        parts = sp.explain(text, patterns, counts, lookup)
        for p in parts:
            p["sentiment"] = models[ver].predict([p[TEXT_KEY[ver]]])[0]
        left, right = st.columns(2)
        with left:
            st.markdown("<span class='badge'>Aspect identified</span>", unsafe_allow_html=True)
            with st.container(key="card_asp"):
                for i, p in enumerate(parts, 1):
                    asp = " + ".join(ASPECTS.get(a, a) for a in p["aspect"].split("|"))
                    st.markdown(f"**{i}.** {p['raw']}")
                    st.markdown(f"<span class='chip' style='background:#E8E8E4'>{asp}</span>", unsafe_allow_html=True)
                    if p["keywords"]:
                        st.caption("because: " + p["keywords"])
        with right:
            st.markdown("<span class='badge'>Aspect sentiment</span>", unsafe_allow_html=True)
            with st.container(key="card_sent"):
                for i, p in enumerate(parts, 1):
                    s = p["sentiment"]
                    st.markdown(f"**{i}.** {p['raw']}")
                    st.markdown(f"<span class='chip{' pos' if s == 'positive' else (' neg' if s == 'negative' else '')}' style='background:{SENT_COLOUR[s]}'>{s.capitalize()}</span>",
                                unsafe_allow_html=True)
                    if p["two_aspects"]:
                        st.caption("Two aspects in this part, so this is its overall sentiment.")
        st.caption(CAUTION)

        st.subheader("How the pipeline processed this review")
        st.caption("Order: 1 Segment (raw text) → 2 Aspect → 3 Clean → 4 Normalize → 5 Dictionary → SVM. "
                   "Segmenting comes first because cleaning removes the dots and commas that mark where to cut.")
        for i, p in enumerate(parts, 1):
            with st.container(key=f"card_part{i}"):
                st.markdown(f"**Part {i}** · cut because: {p['cut']}")
                st.table(pd.DataFrame([
                    ["Raw", p["raw"], ""],
                    ["Cleaned", p["cleaned"], ""],
                    ["Normalized", p["normalized"], p["norm_changes"]],
                    ["English dictionary", p["english"], p["en_changes"]],
                    ["Malay dictionary", p["malay"], p["ms_changes"]],
                ], columns=["Step", "Text", "What changed"]).set_index("Step"))
    else:
        st.info("Type a review above, or pick an example, then press Analyse.")

# ================================================================== PAGE 3
else:
    n = perf["n_labels"]
    cc = perf["class_counts"]
    page_head("Performance", f"The SVM is trained and tested only on my own {n} hand labels "
             f"({cc['negative']} negative, {cc['positive']} positive, {cc['neutral']} neutral). "
             f"Test: {perf['cv']} (each round trains on about 80% and tests on the other 20%). "
             f"Always saying \"negative\" would give {perf['baseline_accuracy']}% accuracy.")
    cols = st.columns(3)
    for col, v in zip(cols, ["manglish", "malay", "english"]):
        r = perf["versions"][v]
        with col.container(key=f"card_perf_{v}"):
            st.markdown(f"<span class='badge'>{VERSION_LABEL[v]}</span>" + ("  ★ best" if v == BEST else ""), unsafe_allow_html=True)
            st.caption(VERSION_NOTE[v])
            m1, m2 = st.columns(2)
            m1.metric("Accuracy", f"{r['accuracy']}%", help="% right: same label as me.")
            m2.metric("Macro-F1", f"{r['macro_f1']}", help="Fair score: the average of the F1-score of negative, neutral and positive.")
            st.caption(f"Moves by about ±{r['accuracy_sd']} (accuracy) and ±{r['macro_f1_sd']} (Macro-F1) from round to round.")
            fig = go.Figure(go.Bar(y=[c.capitalize() for c in CLASSES], x=[r["f1"][c] for c in CLASSES], orientation="h",
                                   marker_color=[SENT_COLOUR[c] for c in CLASSES], text=[f"{r['f1'][c]}" for c in CLASSES],
                                   textposition="outside"))
            fig.update_layout(title="F1-score per class", height=210, margin=dict(l=0, r=50, t=40, b=0), plot_bgcolor="rgba(0,0,0,0)",
                              xaxis=dict(range=[0, 125], showgrid=False, tickvals=[0, 50, 100]), yaxis=dict(autorange="reversed"))
            show(fig)
            cm = r["confusion"]
            lab = [c.capitalize() for c in CLASSES]
            hm = go.Figure(go.Heatmap(z=cm, x=lab, y=lab, colorscale=[[0, "#FFFFFF"], [1, "#8A8A86"]], showscale=False,
                                      text=[[f"{x:.0f}" for x in row] for row in cm], texttemplate="%{text}"))
            hm.update_layout(title="Confusion matrix", height=290, margin=dict(l=0, r=0, t=40, b=0),
                             xaxis=dict(title="Predicted"), yaxis=dict(title="True (my label)", autorange="reversed"))
            show(hm)
    with st.container(key="card_howto"):
        st.markdown("**How to read**")
        st.markdown("- **Accuracy** = out of all parts, how many got the same label as me.\n"
                    "- **F1-score** (per class) = a mix of precision (when it says this class, is it right?) and recall "
                    "(does it find the real ones?). It is high only when both are high.\n"
                    "- **Macro-F1** = the average of the three F1-scores, so the small neutral class counts as much as negative.\n"
                    "- **Confusion matrix** = rows are my label, columns are the SVM's answer. The numbers add up to about "
                    f"{n} parts.")
    st.caption(CAUTION + " Differences of 1 to 3 points between versions are within the round-to-round spread.")