"""
Group Project 3 - Multi-Agent Healthcare Decision Support
Streamlit dashboard (PDF Task 5). Every tab has its own local filters.
Works in both Streamlit themes (Settings -> Theme -> Light / Dark).

Run:  streamlit run app.py
Data: the CSV files produced by Steps 3-7, in ./data
"""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ------------------------------------------------------------------ colour tokens
# One categorical palette that passes the colour-blind, lightness and contrast checks
# on BOTH the light (#ffffff) and dark (#0f1420) surfaces, so charts read the same in either theme.
BLUE, ORANGE, TEAL, VIOLET = "#3b6ff5", "#e8622c", "#0e9f8e", "#9b5de5"
RED = "#e5484d"          # sepsis / critical - only ever shown next to blue, teal or violet
AMBER = "#f5a524"        # NEWS2 "medium" band only
NEUTRAL = "#8a8f98"      # reference lines and secondary text; readable on both themes
CLEAR = "rgba(0,0,0,0)"

st.set_page_config(page_title="Sepsis Decision Support", page_icon=":material/monitor_heart:", layout="wide")

st.markdown(f"""
<style>
  .block-container {{padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1520px;}}
  /* hero header: translucent colour washes work on both light and dark backgrounds */
  .hero {{position: relative; overflow: hidden; border-radius: 18px; padding: 22px 26px 20px 26px; margin-bottom: 10px;
          border: 1px solid rgba(128,128,128,.22);
          background: linear-gradient(120deg, rgba(59,111,245,.20) 0%, rgba(155,93,229,.16) 45%, rgba(14,159,142,.14) 100%);}}
  .hero h1 {{margin: 0 0 6px 0; font-size: 1.65rem; line-height: 1.25; font-weight: 750; letter-spacing: -.01em;}}
  .hero p {{margin: 0; opacity: .78; font-size: .93rem;}}
  .hero .chips {{margin-top: 12px; display: flex; flex-wrap: wrap; gap: 8px;}}
  .chip {{display: inline-block; padding: 4px 11px; border-radius: 999px; font-size: .78rem; font-weight: 600;
          border: 1px solid rgba(128,128,128,.28); background: rgba(128,128,128,.10);}}
  .dot {{display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: middle;}}
  /* KPI tiles */
  .tile {{border-radius: 14px; padding: 13px 15px 12px 15px; height: 100%; min-height: 96px;
          border: 1px solid rgba(128,128,128,.22); background: rgba(128,128,128,.07);
          box-shadow: inset 4px 0 0 0 var(--accent);}}
  .tile .label {{opacity: .66; font-size: .70rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 600;}}
  .tile .value {{font-size: 1.45rem; font-weight: 750; line-height: 1.25; margin-top: 3px; word-break: break-word;}}
  .tile .note {{opacity: .72; font-size: .78rem; margin-top: 2px;}}
  .tile .up {{color: {TEAL}; font-weight: 700;}} .tile .down {{color: {RED}; font-weight: 700;}}
  /* section titles and notes */
  .sec {{font-weight: 700; font-size: 1.02rem; margin: 18px 0 2px 0;}}
  .sub {{opacity: .66; font-size: .82rem; margin: 0 0 4px 0;}}
  .src {{opacity: .6; font-size: .75rem; border-top: 1px dashed rgba(128,128,128,.35); padding-top: 6px; margin: -6px 0 6px 0;}}
  /* summary card */
  .card {{border-radius: 14px; padding: 16px 18px; border: 1px solid rgba(128,128,128,.22);
          background: linear-gradient(135deg, rgba(155,93,229,.12), rgba(59,111,245,.08)); line-height: 1.6;}}
  .card b {{font-weight: 700;}}
  .agent {{display: inline-block; margin: 2px 6px 2px 0; padding: 2px 9px; border-radius: 8px; font-size: .8rem;
           background: rgba(128,128,128,.12); border: 1px solid rgba(128,128,128,.22);}}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
@st.cache_data(show_spinner=False)
def load(name):
    return pd.read_csv(os.path.join(DATA_DIR, name))


@st.cache_data(show_spinner=False)
def load_main():
    d = load("sepsis_multiagent_dataset.csv").sort_values(["patient_id", "ICULOS"])
    d["age_group"] = pd.cut(d["Age"], [0, 45, 65, 80, 120], labels=["<45", "45-64", "65-79", "80+"])
    return d


def tile(label, value, note="", accent=BLUE):
    st.markdown(f'<div class="tile" style="--accent:{accent}"><div class="label">{label}</div>'
                f'<div class="value">{value}</div><div class="note">{note}</div></div>', unsafe_allow_html=True)


def section(title, sub=""):
    st.markdown(f'<div class="sec">{title}</div>' + (f'<div class="sub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


def source(text):
    st.markdown(f'<div class="src">{text}</div>', unsafe_allow_html=True)


def style(fig, height=320, legend=True, xtitle="", ytitle="", bottom=10):
    """Theme-neutral layout: transparent background, the Streamlit theme supplies fonts and grid colours.
    The legend sits in its own strip above the plot, so it never covers data or titles."""
    fig.update_layout(height=height, paper_bgcolor=CLEAR, plot_bgcolor=CLEAR,
                      margin=dict(l=10, r=18, t=44 if legend else 14, b=bottom),
                      showlegend=legend, hoverlabel=dict(font_size=12),
                      legend=dict(orientation="h", x=0, xanchor="left", y=1.0, yanchor="bottom",
                                  bgcolor=CLEAR, font=dict(size=12), title_text=""))
    fig.update_xaxes(title_text=xtitle, zeroline=False, showgrid=False, automargin=True)
    fig.update_yaxes(title_text=ytitle, zeroline=False, automargin=True)
    return fig


def show(fig, key):
    st.plotly_chart(fig, key=key, width="stretch", config={"displaylogo": False, "displayModeBar": False})


def headroom(fig, values, axis="x", factor=1.25):
    lo, hi = float(min(0, min(values))), float(max(0, max(values)))
    span = (hi - lo) or 1.0
    rng = [lo - (span * (factor - 1) if lo < 0 else 0), hi + (span * (factor - 1) if hi > 0 else 0)]
    (fig.update_xaxes if axis == "x" else fig.update_yaxes)(range=rng)
    return fig


def gauge(value, threshold, title, color=VIOLET, height=210, fmt=".3f", axis_max=None):
    axis_max = axis_max or max(threshold * 1.25, value * 1.25, 0.05)
    f = go.Figure(go.Indicator(
        mode="gauge", value=value, domain=dict(x=[0.08, 0.92], y=[0.0, 0.92]),
        title=dict(text=title, font=dict(size=13)),
        gauge=dict(axis=dict(range=[0, axis_max], tickformat=".2f", nticks=4, tickfont=dict(size=10)),
                   bar=dict(color=color, thickness=.32), bgcolor=CLEAR, borderwidth=0,
                   steps=[dict(range=[0, min(threshold, axis_max)], color="rgba(14,159,142,.18)"),
                          dict(range=[min(threshold, axis_max), axis_max], color="rgba(229,72,77,.16)")],
                   threshold=dict(line=dict(color=RED, width=3), thickness=.85, value=min(threshold, axis_max)))))
    f.add_annotation(x=.5, y=.06, xref="paper", yref="paper", showarrow=False,
                     text=f"<b>{value:{fmt}}</b>", font=dict(size=24, color=color))
    f.update_layout(height=height, margin=dict(l=20, r=20, t=42, b=4), paper_bgcolor=CLEAR)
    return f


def donut(labels, values, colors, center, height=260):
    values = np.asarray(values, dtype=float)
    total = values.sum() or 1.0
    names = [f"{l} · {v / total * 100:.1f}%" for l, v in zip(labels, values)]
    f = go.Figure(go.Pie(labels=names, values=values, hole=.64, sort=False, direction="clockwise", textinfo="none",
                         marker=dict(colors=colors, line=dict(color="rgba(128,128,128,.25)", width=2)),
                         domain=dict(x=[0.2, 0.8], y=[0.3, 1.0]),
                         hovertemplate="%{label}<br>%{value:,.0f}<extra></extra>"))
    f.add_annotation(text=center, showarrow=False, font=dict(size=14), x=.5, y=.65, xref="paper", yref="paper")
    f.update_layout(height=height, margin=dict(l=6, r=6, t=6, b=6), paper_bgcolor=CLEAR,
                    legend=dict(orientation="h", x=.5, xanchor="center", y=0.22, yanchor="top", bgcolor=CLEAR, font=dict(size=12)))
    return f


def play_buttons(fig, frame_ms=120):
    """Play / pause controls under the plot, kept clear of the legend and the data."""
    fig.update_layout(updatemenus=[dict(
        type="buttons", direction="left", x=0, xanchor="left", y=-0.2, yanchor="top", pad=dict(t=0, r=6),
        showactive=False, bgcolor="rgba(128,128,128,.14)", bordercolor="rgba(128,128,128,.35)", font=dict(size=12),
        buttons=[dict(label="Play", method="animate",
                      args=[None, dict(frame=dict(duration=frame_ms, redraw=False), fromcurrent=False,
                                       transition=dict(duration=0), mode="immediate")]),
                 dict(label="Pause", method="animate",
                      args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])])
    return fig


df = load_main()
n_pat = df["patient_id"].nunique()
sep_rate = df.groupby("patient_id")["sepsis_patient"].first().mean()

st.markdown(f"""<div class="hero">
  <h1>Multi-Agent Healthcare Decision Support &middot; Sepsis in the ICU</h1>
  <p>Six agents &mdash; doctor, nurse, AI tool, pharmacist, lab technician and administrator &mdash; reason under uncertainty with
     MLE, a POMDP, bandits and Bayesian inference. Each tab has its own filters.</p>
  <div class="chips">
    <span class="chip"><span class="dot" style="background:{BLUE}"></span>{n_pat:,} ICU patients &middot; {len(df):,} patient-hours</span>
    <span class="chip"><span class="dot" style="background:{RED}"></span>{sep_rate*100:.2f}% developed sepsis</span>
    <span class="chip"><span class="dot" style="background:{TEAL}"></span>Real data: PhysioNet/CinC 2019 (CC BY 4.0)</span>
    <span class="chip"><span class="dot" style="background:{VIOLET}"></span>Operations layer from cited sources</span>
  </div>
</div>""", unsafe_allow_html=True)

TABS = [":material/person: Patient Overview", ":material/timeline: POMDP Belief Tracker",
        ":material/casino: Bandit Decision Explorer", ":material/hub: Bayesian Network",
        ":material/functions: MLE Parameter Insights", ":material/forum: Agent Collaboration Log",
        ":material/monitoring: Outcome & KPI Analysis"]
t1, t2, t3, t4, t5, t6, t7 = st.tabs(TABS)

# ============================================================ 1. PATIENT OVERVIEW
with t1:
    c1, c2, c3 = st.columns([1.6, 1.3, 1.8], vertical_alignment="bottom")
    with c1:
        only_sepsis = st.toggle("Only patients who developed sepsis", value=True, key="po_sep")
        pool = df[df["sepsis_patient"] == 1] if only_sepsis else df
        pid = st.selectbox("Patient", sorted(pool["patient_id"].unique()), key="po_pid")
    full = df[df["patient_id"] == pid]
    lo_h, hi_h = int(full["ICULOS"].min()), int(full["ICULOS"].max())
    with c2:
        hrs = st.slider("Hours shown", lo_h, hi_h, (lo_h, hi_h), key="po_hrs") if hi_h > lo_h else (lo_h, hi_h)
    with c3:
        shown = st.multiselect("Vital signs", ["HR", "Resp", "Temp", "O2Sat", "SBP", "MAP"],
                               default=["HR", "Resp", "Temp", "O2Sat"], key="po_vitals")
    g = full[(full["ICULOS"] >= hrs[0]) & (full["ICULOS"] <= hrs[1])]
    r0 = g.iloc[0]
    sep = int(r0["sepsis_patient"])
    onset = float(r0["sepsis_onset_hour"]) if sep else None

    k = st.columns(6)
    with k[0]: tile("Age / sex", f"{r0['Age']:.0f} / {'M' if r0['Gender'] == 1 else 'F'}", f"hospital set {r0['hospital_set']}")
    with k[1]: tile("Hours shown", f"{len(g)}", f"of {hi_h} in the record", TEAL)
    with k[2]: tile("Sepsis", "Yes" if sep else "No", f"onset at hour {onset:.0f}" if sep else "never labelled", RED if sep else TEAL)
    with k[3]: tile("NEWS2 peak", f"{g['news2_score'].max():.0f}", f"median {g['news2_score'].median():.0f}", ORANGE)
    with k[4]: tile("Lab panels", f"{int(g['panels_ordered'].sum())}",
                    f"{g['lab_tat_max_min'].mean():.0f} min average wait" if g["panels_ordered"].sum() else "none ordered", VIOLET)
    with k[5]: tile("ICU cost", f"${g['cum_cost_usd'].max():,.0f}", "cumulative at $179.17 per hour", BLUE)

    section("Care timeline", "Every event in this patient's stay, hour by hour")
    ev = [("NEWS2 ≥ 5", g.loc[g["news2_score"] >= 5, "ICULOS"], ORANGE, "diamond"),
          ("Lab panel", g.loc[g["panels_ordered"] > 0, "ICULOS"], VIOLET, "square"),
          ("Antibiotics", g.loc[g["on_antibiotics"] == 1, "ICULOS"], TEAL, "circle"),
          ("Sepsis label", g.loc[g["SepsisLabel"] == 1, "ICULOS"], RED, "circle")]
    f = go.Figure()
    for name, xs, col, sym in ev:
        f.add_trace(go.Scatter(x=xs, y=[name] * len(xs), mode="markers", name=name,
                               marker=dict(color=col, size=11, symbol=sym, line=dict(color="rgba(255,255,255,.7)", width=1.5)),
                               hovertemplate=f"{name} · hour %{{x}}<extra></extra>"))
    f.update_yaxes(categoryorder="array", categoryarray=[e[0] for e in ev][::-1], showgrid=False)
    f.update_xaxes(range=[hrs[0] - .5, hrs[1] + .5], showgrid=True, gridcolor="rgba(128,128,128,.15)")
    show(style(f, height=200, legend=False, xtitle="ICU hour"), "po_timeline")

    left, right = st.columns([1.55, 1], gap="large")
    with left:
        section("Vital signs", "One panel per sign because the scales differ · dots = measured that hour, line = last value carried forward")
        for i, v in enumerate(shown):
            f = go.Figure()
            f.add_trace(go.Scatter(x=g["ICULOS"], y=g[v], mode="lines", line=dict(color=BLUE, width=2.2, shape="spline", smoothing=.4),
                                   name=v,
                                   hovertemplate=f"hour %{{x}} · {v} %{{y:.1f}}<extra></extra>"))
            meas = g[g[f"{v}_measured"] == 1]
            f.add_trace(go.Scatter(x=meas["ICULOS"], y=meas[v], mode="markers", name="measured",
                                   marker=dict(color=BLUE, size=6.5, line=dict(color="rgba(255,255,255,.8)", width=1.5)),
                                   hovertemplate="measured this hour<extra></extra>"))
            ymin, ymax = np.nanmin(g[v]), np.nanmax(g[v])
            pad = (ymax - ymin) * .25 if ymax > ymin else 1
            f.update_yaxes(range=[ymin - pad, ymax + pad])
            if onset is not None and hrs[0] <= onset <= hrs[1]:
                f.add_vline(x=onset, line=dict(color=RED, width=2, dash="dot"))
            show(style(f, height=160, legend=False, ytitle=v, xtitle="ICU hour" if i == len(shown) - 1 else ""), f"po_v{v}")
        if onset is not None:
            source(f"Red dotted line = sepsis onset (hour {onset:.0f}). Data: PhysioNet 2019.")
    with right:
        section("NEWS2 early-warning score", "Royal College of Physicians bands · 5 of 7 parameters available")
        f = go.Figure()
        top = max(8, float(np.nanmax(g["news2_score"])) + 2)
        f.add_hrect(y0=0, y1=4.5, fillcolor=TEAL, opacity=.10, line_width=0,
                    annotation_text="low", annotation_position="inside top left", annotation_font=dict(color=TEAL, size=11))
        f.add_hrect(y0=4.5, y1=6.5, fillcolor=AMBER, opacity=.14, line_width=0,
                    annotation_text="medium", annotation_position="inside top left", annotation_font=dict(color=AMBER, size=11))
        f.add_hrect(y0=6.5, y1=top, fillcolor=RED, opacity=.10, line_width=0,
                    annotation_text="high", annotation_position="inside top left", annotation_font=dict(color=RED, size=11))
        f.add_trace(go.Scatter(x=g["ICULOS"], y=g["news2_score"], mode="lines+markers", name="NEWS2",
                               line=dict(color=VIOLET, width=2.2), marker=dict(size=5),
                               hovertemplate="hour %{x} · NEWS2 %{y:.0f}<extra></extra>"))
        f.update_yaxes(range=[0, top])
        show(style(f, height=260, legend=False, ytitle="NEWS2", xtitle="ICU hour"), "po_news2")

        section("Tests ordered and time to result", "Turnaround from published stat-test times (Fei et al., 2015)")
        f = go.Figure()
        vals = []
        for name, col, colr in [("CBC", "cbc_ordered", BLUE), ("Chemistry", "chem_ordered", ORANGE), ("Coagulation", "coag_ordered", TEAL)]:
            sel = g[g[col] == 1]
            vals += list(sel[col.replace("ordered", "tat_min")])
            f.add_trace(go.Bar(x=sel["ICULOS"], y=sel[col.replace("ordered", "tat_min")], name=name,
                               marker=dict(color=colr, cornerradius=4), hovertemplate=f"{name} · hour %{{x}} · %{{y:.0f}} min<extra></extra>"))
        f.update_layout(barmode="group", bargap=.25)
        if vals:
            f.add_hline(y=60, line=dict(color=NEUTRAL, width=1.5, dash="dot"))
        show(style(f, height=260, ytitle="minutes", xtitle="ICU hour"), "po_labs")
        source("Dotted line = 60-minute target (Hawkins, 2007).")

# ======================================================= 2. POMDP BELIEF TRACKER
with t2:
    tr = load("pomdp_belief_traces.csv")
    pol = load("pomdp_policy.csv")
    par = load("pomdp_parameters.csv").set_index("parameter")["value"]
    thr = {int(h): float(gg.loc[gg["best_action"] == "treat", "belief_p_sepsis"].min()) for h, gg in pol.groupby("hours_left")}

    c1, c2, c3 = st.columns([1.4, 1.3, 1.7], vertical_alignment="bottom")
    with c1:
        grp = st.segmented_control("Patients", ["Sepsis", "All"], default="Sepsis", key="pb_grp") or "Sepsis"
        pool = tr[tr["sepsis"] == 1] if grp == "Sepsis" else tr
    with c2:
        pid2 = st.selectbox("Patient", sorted(pool["patient_id"].unique()), key="pb_pid")
    with c3:
        hleft = st.select_slider("Treat threshold when this many hours are left", options=sorted(thr.keys()), value=24, key="pb_hl")
    b = tr[tr["patient_id"] == pid2].sort_values("hour")
    tt = b["treated_at"].iloc[0]

    kc = st.columns([1, 1, 1, 1.25])
    with kc[0]: tile("Start belief", f"{par['P_INIT']:.3f}", "MLE: sepsis at admission", BLUE)
    with kc[1]: tile("Peak belief", f"{b['belief_p_sepsis'].max():.3f}", f"at hour {int(b.loc[b['belief_p_sepsis'].idxmax(), 'hour'])}", ORANGE)
    with kc[2]: tile("POMDP decision", "Antibiotics" if pd.notna(tt) else "Keep monitoring",
                     f"at hour {tt:.0f}" if pd.notna(tt) else "belief stayed under the threshold", VIOLET)
    with kc[3]:
        show(gauge(float(b["belief_p_sepsis"].iloc[-1]), thr[hleft], f"Final belief vs threshold ({hleft} h left)", height=170), "pb_gauge")

    left, right = st.columns([1.5, 1], gap="large")
    with left:
        section("Belief that this patient has sepsis", "Updated by Bayes' rule every hour · press Play to watch it evolve")
        x, y = b["hour"].to_numpy(), b["belief_p_sepsis"].to_numpy()
        top = max(float(y.max()) * 1.5, 0.02)
        f = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers", line=dict(color=BLUE, width=2.6),
                                 marker=dict(size=5), fill="tozeroy", fillcolor="rgba(59,111,245,.12)", name="belief",
                                 hovertemplate="hour %{x} · belief %{y:.3f}<extra></extra>"))
        f.frames = [go.Frame(data=[go.Scatter(x=x[:i], y=y[:i])], name=str(i)) for i in range(1, len(x) + 1)]
        f.update_xaxes(range=[x.min() - .5, x.max() + .5])
        f.update_yaxes(range=[0, top])
        if thr[hleft] <= top:
            f.add_hline(y=thr[hleft], line=dict(color=RED, width=2, dash="dash"))
        if pd.notna(tt):
            f.add_vline(x=float(tt), line=dict(color=VIOLET, width=2, dash="dot"))
        play_buttons(style(f, height=330, legend=False, ytitle="P(sepsis)", xtitle="ICU hour", bottom=58), frame_ms=90)
        show(f, "pb_trace")
        note = (f"Red dashed line = treat threshold {thr[hleft]:.3f}." if thr[hleft] <= top
                else f"The treat threshold ({thr[hleft]:.3f}) is far above this patient's belief, so the policy keeps monitoring.")
        source(note + (f" Violet dotted line = antibiotics at hour {tt:.0f}." if pd.notna(tt) else "")
               + " Transition 0.0017/hour and test accuracies are MLE estimates (Step 4).")
    with right:
        section("When does the policy say treat?", "Expected value of the best action · dotted line = switch point")
        f = go.Figure()
        cols = {72: BLUE, 24: ORANGE, 1: TEAL}
        for h in sorted(thr.keys(), reverse=True):
            sub = pol[pol["hours_left"] == h]
            f.add_trace(go.Scatter(x=sub["belief_p_sepsis"], y=sub["value"], mode="lines", name=f"{h} h left",
                                   line=dict(color=cols.get(h, VIOLET), width=2.4),
                                   hovertemplate=f"{h} h left · belief %{{x:.2f}} · value %{{y:.2f}}<extra></extra>"))
            f.add_vline(x=thr[h], line=dict(color=cols.get(h, VIOLET), width=1.5, dash="dot"))
        show(style(f, height=330, ytitle="expected value", xtitle="belief P(sepsis)"), "pb_policy")
        source("Fewer hours left → the policy treats at a lower belief, because waiting can no longer pay off.")

    section("Where every tracked patient ended up", f"{tr['patient_id'].nunique()} patients followed in Step 5 (the traces file holds the first 200)")
    fin = tr.sort_values("hour").groupby("patient_id").tail(1)
    f = go.Figure()
    for lab, val, colr in [("no sepsis", 0, BLUE), ("sepsis", 1, RED)]:
        s = fin[fin["sepsis"] == val]
        f.add_trace(go.Histogram(x=s["belief_p_sepsis"], name=lab, nbinsx=40, opacity=.85,
                                 marker=dict(color=colr, cornerradius=3),
                                 hovertemplate=f"{lab} · belief %{{x}} · %{{y}} patients<extra></extra>"))
    f.update_layout(barmode="overlay")
    show(style(f, height=250, ytitle="patients", xtitle="final belief P(sepsis)"), "pb_hist")

# =================================================== 3. BANDIT DECISION EXPLORER
with t3:
    bt = load("bandit_treatment.csv")
    bp = load("bandit_policy.csv")
    sh = load("bandit_arm_shares.csv")
    ALG = {"Thompson sampling": ("thompson_mean_regret", BLUE), "UCB1": ("ucb1_mean_regret", ORANGE),
           "Random choice": ("random_choice_regret", NEUTRAL)}

    c1, c2 = st.columns([1.3, 2], vertical_alignment="bottom")
    with c1:
        upto = st.slider("Patients seen by the bandit", 100, int(bt["patient"].max()), int(bt["patient"].max()), step=100, key="bd_upto")
    with c2:
        algos = st.pills("Algorithms", list(ALG), selection_mode="multi", default=list(ALG), key="bd_alg") or list(ALG)
    b = bt[bt["patient"] <= upto]

    k = st.columns(4)
    with k[0]: tile("Arm A · antibiotics ≤ 1 h", "83.4% survive", "MLE on 1,526 sepsis patients", BLUE)
    with k[1]: tile("Arm B · antibiotics > 1 h", "80.6% survive", "MLE on 1,406 sepsis patients", ORANGE)
    rnd = b["random_choice_regret"].iloc[-1]
    for i, name in enumerate(["Thompson sampling", "UCB1"]):
        v = b[ALG[name][0]].iloc[-1]
        with k[2 + i]:
            tile(f"{name} regret", f"{v:.1f}", f'<span class="up">▼ {(1 - v / rnd) * 100:.0f}%</span> vs random after {upto:,} patients', ALG[name][1])

    left, right = st.columns([1.55, 1], gap="large")
    with left:
        section("Learning which timing saves more lives", "Cumulative regret (lower is better) · press Play to replay the learning")
        step = 50
        idx = list(range(step - 1, len(b), step)) or [len(b) - 1]
        f = go.Figure()
        for name in algos:
            col, colr = ALG[name]
            f.add_trace(go.Scatter(x=b["patient"], y=b[col], mode="lines", name=name,
                                   line=dict(color=colr, width=2.6, dash="dash" if name == "Random choice" else "solid"),
                                   hovertemplate=f"{name} · patient %{{x}} · regret %{{y:.1f}}<extra></extra>"))
        f.frames = [go.Frame(data=[go.Scatter(x=b["patient"].iloc[:i + 1], y=b[ALG[n][0]].iloc[:i + 1]) for n in algos],
                             name=str(i)) for i in idx]
        f.update_xaxes(range=[0, upto * 1.02])
        f.update_yaxes(range=[0, float(b[[ALG[n][0] for n in algos]].max().max()) * 1.1])
        play_buttons(style(f, height=340, ytitle="cumulative regret", xtitle="patients treated", bottom=58), frame_ms=60)
        show(f, "bd_regret")
        source("Regret = survival lost against always choosing the better timing. Mean of 30 runs.")
    with right:
        section("Which ward policy does the bandit settle on?", "Share of patients given each rule")
        BN = {"thompson": "Thompson", "thompson_contextual": "Thompson, per patient group"}
        who = st.segmented_control("Bandit", list(BN), format_func=BN.get, default="thompson", key="bd_who") or "thompson"
        s = sh[sh["bandit"] == who].groupby("arm", as_index=False)["times_chosen"].sum()
        order = ["treat at once", "NEWS2 >= 5", "SIRS >= 2", "never treat"]
        s = s.set_index("arm").reindex(order).fillna(0).reset_index()
        nice = [a.replace(">=", "≥") for a in s["arm"]]
        show(donut(nice, s["times_chosen"], [RED, ORANGE, BLUE, TEAL], f"{int(s['times_chosen'].sum()):,}<br>patients"), "bd_donut")
        source("One example run at W_ABX = 1.0. The rules' rewards are close, so the mix varies from run to run.")

    section("Choosing among ward policies", "Regret under two views of how costly unnecessary antibiotics are")
    c1, c2 = st.columns([1, 3], vertical_alignment="center")
    with c1:
        w = st.radio("Cost of unnecessary antibiotics (W_ABX)", sorted(bp["w_abx"].unique()),
                     format_func=lambda v: f"{v} · {'cheap' if v < 1 else 'costly'}", key="bd_w")
        st.caption("Costly: every patient group has the same best rule, so learning per group only adds exploration.")
    with c2:
        s = bp[bp["w_abx"] == w]
        f = go.Figure()
        for name, col, colr in [("Thompson", "thompson", BLUE), ("UCB1", "ucb1", ORANGE), ("Thompson, per patient group", "thompson_contextual", VIOLET)]:
            f.add_trace(go.Scatter(x=s["patient"], y=s[col], mode="lines", name=name, line=dict(color=colr, width=2.4),
                                   hovertemplate=f"{name} · patient %{{x}} · regret %{{y:.0f}}<extra></extra>"))
        show(style(f, height=300, ytitle="cumulative regret", xtitle="patients"), "bd_pol")

# ======================================================= 4. BAYESIAN NETWORK
with t4:
    cpt = load("bayes_network.csv")
    fus = load("bayes_fusion.csv")
    LAB = {"temp_abn": "Temperature > 38 or < 36 °C", "hr_abn": "Heart rate > 90",
           "rr_abn": "Respiratory rate > 20", "wbc_abn": "WBC > 12 or < 4"}
    SHORT = {"temp_abn": "Temp", "hr_abn": "HR", "rr_abn": "RR", "wbc_abn": "WBC"}

    section("Try it: tick what the patient shows and watch the probability move")
    c = st.columns([1.3, 1, 1, 1, 1], vertical_alignment="bottom")
    with c[0]:
        prior = st.slider("Prior P(sepsis) this hour", 0.005, 0.30, 0.018, 0.001, key="bn_prior")
    picks = {}
    for i, s in enumerate(cpt["sign"]):
        with c[i + 1]:
            picks[s] = st.checkbox(LAB[s], value=(s in ("hr_abn", "rr_abn")), key=f"bn_{s}")

    lo = np.log(prior / (1 - prior)); run = lo; steps = []
    for _, r in cpt.iterrows():
        lr = r["LR if present"] if picks[r["sign"]] else r["LR if absent"]
        run += np.log(lr)
        steps.append((SHORT[r["sign"]] + (" ✓" if picks[r["sign"]] else " ✗"), np.log(lr)))
    post = 1 / (1 + np.exp(-run))

    k = st.columns([1, 1, 1, 1.3])
    with k[0]: tile("Prior", f"{prior:.3f}", "before any sign is read", NEUTRAL)
    with k[1]:
        ch = post / prior
        tile("Posterior", f"{post:.3f}", f'<span class="{"down" if ch > 1 else "up"}">{"▲" if ch > 1 else "▼"} {ch:.2f}×</span> the prior', VIOLET)
    with k[2]: tile("Signs present", f"{sum(picks.values())} of 4", "SIRS criteria (Bone et al., 1992)", ORANGE)
    with k[3]: show(gauge(post, 0.143, "Posterior vs treat-or-never threshold (0.143)", height=170), "bn_gauge")

    left, mid, right = st.columns([1.05, 1.25, 1.1], gap="large")
    with left:
        section("The network", "Sepsis is the parent; the four signs are its children")
        f = go.Figure()
        kids = list(cpt["sign"])
        xs = np.linspace(0.1, 0.9, len(kids))
        for xk, s in zip(xs, kids):
            f.add_shape(type="line", x0=.5, y0=.78, x1=xk, y1=.26, layer="below", line=dict(color="rgba(128,128,128,.55)", width=2))
        f.add_trace(go.Scatter(x=[.5], y=[.82], mode="markers+text", text=[f"Sepsis<br>{post:.3f}"], textposition="middle center",
                               marker=dict(size=74, color=VIOLET, line=dict(color="rgba(255,255,255,.8)", width=2)),
                               textfont=dict(color="white", size=12), hoverinfo="skip"))
        f.add_trace(go.Scatter(x=xs, y=[.22] * len(kids), mode="markers+text",
                               text=[SHORT[s] + ("<br>present" if picks[s] else "<br>absent") for s in kids],
                               textposition="middle center", textfont=dict(color="white", size=11),
                               marker=dict(size=64, color=[RED if picks[s] else TEAL for s in kids],
                                           line=dict(color="rgba(255,255,255,.8)", width=2)),
                               customdata=[LAB[s] for s in kids], hovertemplate="%{customdata}<extra></extra>"))
        f.update_xaxes(visible=False, range=[-.05, 1.05]); f.update_yaxes(visible=False, range=[0, 1.05])
        show(style(f, height=300, legend=False), "bn_net")
        source("Red = sign present, teal = absent. Tables estimated from 55,735 real patient-hours.")
    with mid:
        section("How each sign moves the log-odds", "✓ present · ✗ absent")
        f = go.Figure(go.Waterfall(
            orientation="v", measure=["absolute"] + ["relative"] * len(steps) + ["total"],
            x=["prior"] + [s[0] for s in steps] + ["posterior"], y=[lo] + [s[1] for s in steps] + [0],
            connector=dict(line=dict(color="rgba(128,128,128,.4)")),
            increasing=dict(marker=dict(color=RED)), decreasing=dict(marker=dict(color=TEAL)),
            totals=dict(marker=dict(color=VIOLET)),
            text=[f"{lo:.2f}"] + [f"{s[1]:+.2f}" for s in steps] + [f"{run:.2f}"], textposition="outside",
            hovertemplate="%{x}<br>%{text}<extra></extra>"))
        f.update_yaxes(range=[min(lo, run) - 1.2, max(0.4, max(lo, run) + .8)])
        show(style(f, height=300, legend=False, ytitle="log-odds of sepsis"), "bn_wf")
    with right:
        section("Likelihood ratios", "Above 1 raises the odds, below 1 lowers them")
        f = go.Figure()
        f.add_trace(go.Bar(y=[SHORT[s] for s in cpt["sign"]], x=cpt["LR if present"], orientation="h", name="present",
                           marker=dict(color=RED, cornerradius=4), hovertemplate="%{y} present: ×%{x:.2f}<extra></extra>"))
        f.add_trace(go.Bar(y=[SHORT[s] for s in cpt["sign"]], x=cpt["LR if absent"], orientation="h", name="absent",
                           marker=dict(color=TEAL, cornerradius=4), hovertemplate="%{y} absent: ×%{x:.2f}<extra></extra>"))
        f.add_vline(x=1, line=dict(color=NEUTRAL, width=1.5, dash="dot"))
        f.update_layout(barmode="group", bargap=.3)
        show(style(f, height=300, xtitle="multiplies the odds by"), "bn_lr")

    section("Fusing the nurse, doctor and AI opinions", "Held-out 453 patients · the fused opinion ranks best but overstates the risk")
    c1, c2 = st.columns([1.5, 1], gap="large")
    with c1:
        metric = st.segmented_control("Measure", ["Ranking (area under ROC)", "Accuracy of the number (Brier)"],
                                      default="Ranking (area under ROC)", key="bn_metric") or "Ranking (area under ROC)"
        col = "area under ROC" if metric.startswith("Ranking") else "Brier score"
        s = fus.sort_values(col, ascending=(col != "area under ROC"))
        f = go.Figure(go.Bar(y=s["opinion"], x=s[col], orientation="h",
                             marker=dict(color=[VIOLET if "fused" in o else BLUE for o in s["opinion"]], cornerradius=5),
                             text=[f"{v:.3f}" if col == "area under ROC" else f"{v:.5f}" for v in s[col]], textposition="outside",
                             hovertemplate="%{y}<br>%{x:.4f}<extra></extra>"))
        if col == "area under ROC":
            f.add_vline(x=.5, line=dict(color=NEUTRAL, width=1.5, dash="dot"))
        headroom(f, list(s[col]), "x", 1.18)
        show(style(f, height=250, legend=False, xtitle=col + ("  (higher is better)" if col == "area under ROC" else "  (lower is better)")), "bn_fus")
    with c2:
        mr = fus.set_index("opinion")["mean predicted risk"]
        actual = float(fus["actual rate"].iloc[0])
        f = go.Figure(go.Bar(x=[o.split(" (")[0] for o in mr.index], y=mr.values * 100,
                             marker=dict(color=[VIOLET if "fused" in o else BLUE for o in mr.index], cornerradius=5),
                             text=[f"{v*100:.2f}%" for v in mr.values], textposition="outside",
                             hovertemplate="%{x}<br>mean predicted risk %{y:.2f}%<extra></extra>"))
        f.add_hline(y=actual * 100, line=dict(color=RED, width=2, dash="dash"))
        headroom(f, list(mr.values * 100), "y", 1.3)
        show(style(f, height=250, legend=False, ytitle="mean predicted risk (%)"), "bn_cal")
        source(f"Red dashed line = the real rate ({actual*100:.2f}%). Pooling assumes the three opinions are independent; they share the same vitals.")

# ==================================================== 5. MLE PARAMETER INSIGHTS
with t5:
    mle = load("mle_parameters.csv")
    NICE = {"progression": "Disease progression", "test_accuracy": "Test accuracy",
            "treatment_success": "Treatment success", "satisfaction_curve": "Satisfaction curve"}
    c1, c2 = st.columns([2.3, 1.5], vertical_alignment="bottom")
    with c1:
        item = st.segmented_control("Estimate group", list(NICE), format_func=NICE.get, default="progression", key="mle_item") or "progression"
    sub = mle[mle["item"] == item].copy()
    with c2:
        params = st.multiselect("Parameters", sorted(sub["parameter"].unique()), default=sorted(sub["parameter"].unique()),
                                key=f"mle_par_{item}")
    sub = sub[sub["parameter"].isin(params)] if params else sub
    sub["label"] = (sub["group"].str.replace("news2_band=", "NEWS2 ").str.replace(">=", "≥")
                    + " · " + sub["parameter"].str.replace("_", " "))

    k = st.columns(4)
    with k[0]: tile("Estimates shown", f"{len(sub)}", NICE[item], BLUE)
    with k[1]: tile("Widest interval", f"{(sub['ci_high'] - sub['ci_low']).max():.3f}", "the least certain estimate", ORANGE)
    with k[2]: tile("Smallest sample", f"{int(sub['n'].min()):,}", "observations behind one estimate", TEAL)
    with k[3]: tile("Method", "Max. likelihood", "95% intervals: patient bootstrap or Wald", VIOLET)

    if item == "test_accuracy":
        left, right = st.columns([1.3, 1], gap="large")
    else:
        left, right = st.container(), None
    with left:
        section("Each estimate with its 95% confidence interval")
        f = go.Figure(go.Scatter(
            x=sub["mle"], y=sub["label"], mode="markers", marker=dict(color=VIOLET, size=12, line=dict(color="rgba(255,255,255,.8)", width=2)),
            error_x=dict(type="data", symmetric=False, array=sub["ci_high"] - sub["mle"], arrayminus=sub["mle"] - sub["ci_low"],
                         color=VIOLET, thickness=2.2, width=7),
            customdata=np.stack([sub["ci_low"], sub["ci_high"], sub["n"]], axis=-1),
            hovertemplate="%{y}<br>MLE %{x:.4f}<br>95% CI %{customdata[0]:.4f} – %{customdata[1]:.4f}<br>n = %{customdata[2]:,}<extra></extra>"))
        f.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,.12)")
        show(style(f, height=110 + 40 * len(sub), legend=False, xtitle="estimated value"), "mle_dot")
    if right is not None:
        with right:
            section("The tests in ROC space", "Up and to the left is better · the diagonal is a coin flip")
            se = sub[sub["parameter"] == "sensitivity"].set_index("group")["mle"]
            sp = sub[sub["parameter"] == "specificity"].set_index("group")["mle"]
            both = se.index.intersection(sp.index)
            f = go.Figure()
            f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=NEUTRAL, dash="dot", width=1.5), hoverinfo="skip"))
            SHORTT = {"SIRS >= 2": "SIRS ≥ 2", "HR > 90": "HR > 90", "WBC > 12 or < 4": "WBC", "RR > 20": "RR > 20",
                      "Temp > 38 or < 36": "Temp", "NEWS2 >= 5": "NEWS2 ≥ 5"}
            POS = {"SIRS >= 2": "top center", "HR > 90": "middle left", "WBC > 12 or < 4": "middle right",
                   "RR > 20": "bottom right", "Temp > 38 or < 36": "top left", "NEWS2 >= 5": "bottom right"}
            f.add_trace(go.Scatter(x=1 - sp[both], y=se[both], mode="markers+text", text=[SHORTT.get(t, t) for t in both],
                                   textposition=[POS.get(t, "top center") for t in both], customdata=list(both),
                                   textfont=dict(size=11), marker=dict(size=13, color=[VIOLET if "SIRS >= 2" in t else BLUE for t in both],
                                                                   line=dict(color="rgba(255,255,255,.8)", width=2)),
                                   hovertemplate="%{customdata}<br>false-positive rate %{x:.3f}<br>sensitivity %{y:.3f}<extra></extra>"))
            f.update_xaxes(range=[0, .55]); f.update_yaxes(range=[0, .7])
            show(style(f, height=330, legend=False, xtitle="1 − specificity", ytitle="sensitivity"), "mle_roc")
    source("Estimated in Step 4 from the 2,000-patient dataset; hourly quantities resample whole patients for the intervals.")
    with st.expander("See the numbers"):
        st.dataframe(sub[["parameter", "group", "mle", "ci_low", "ci_high", "n"]]
                     .rename(columns={"mle": "MLE", "ci_low": "CI low", "ci_high": "CI high", "n": "observations"}),
                     width="stretch", hide_index=True)

# ==================================================== 6. AGENT COLLABORATION LOG
with t6:
    log = load("collaboration_log.csv")
    c1, c2, c3 = st.columns([1.2, 1.6, 1.4], vertical_alignment="bottom")
    with c1:
        only_conf = st.toggle("Only hours where the team disagrees", value=False, key="cl_conf")
    with c2:
        acts = st.pills("Suggested action", sorted(log["action"].unique()), selection_mode="multi",
                        default=sorted(log["action"].unique()), key="cl_act") or sorted(log["action"].unique())
    view = log[log["action"].isin(acts)]
    if only_conf:
        view = view[view["conflict"]]
    with c3:
        pid3 = st.selectbox("Patient", sorted(view["patient_id"].unique()) if len(view) else ["—"], key="cl_pid")

    k = st.columns(5)
    with k[0]: tile("Hours in view", f"{len(view):,}", f"of {len(log):,} logged", BLUE)
    with k[1]: tile("Team disagrees", f"{log['conflict'].mean() * 100:.1f}%", "of all logged hours", ORANGE)
    with k[2]: tile("Antibiotics suggested", f"{(log['action'] == 'start antibiotics').mean() * 100:.1f}%", "of all logged hours", VIOLET)
    sep_h = log[log["sepsis_label"] == 1]
    with k[3]: tile("…in true sepsis hours", f"{(sep_h['action'] == 'start antibiotics').mean() * 100:.1f}%", f"{len(sep_h)} such hours", RED)
    with k[4]: tile("Patients logged", f"{log['patient_id'].nunique()}", "first 200 in the dataset", TEAL)

    p = view[view["patient_id"] == pid3].sort_values("hour")
    left, right = st.columns([1.6, 1], gap="large")
    with left:
        section(f"Patient {pid3}: what each agent thought, hour by hour", "Dotted = single agent · solid = pooled posterior that decides")
        f = go.Figure()
        for name, col, colr in [("Nurse (NEWS2)", "nurse_p", BLUE), ("Doctor (SIRS)", "doctor_p", ORANGE),
                                ("AI tool", "ai_p", TEAL), ("Fused", "fused_p", VIOLET)]:
            f.add_trace(go.Scatter(x=p["hour"], y=p[col], mode="lines", name=name,
                                   line=dict(color=colr, width=3.2 if name == "Fused" else 1.8, dash="solid" if name == "Fused" else "dot"),
                                   hovertemplate=f"{name} · hour %{{x}} · %{{y:.3f}}<extra></extra>"))
        note = ""
        if len(p):
            top = max(float(p[["nurse_p", "doctor_p", "ai_p", "fused_p"]].to_numpy().max()) * 1.4, .02)
            if p["threshold"].min() <= top:
                f.add_trace(go.Scatter(x=p["hour"], y=p["threshold"], mode="lines", name="treat threshold",
                                       line=dict(color=RED, width=1.8, dash="dash"), hovertemplate="threshold %{y:.3f}<extra></extra>"))
            else:
                note = f"The treat threshold ({p['threshold'].min():.3f}–{p['threshold'].max():.3f}) is above this chart. "
            f.update_yaxes(range=[0, top])
        show(style(f, height=330, ytitle="P(sepsis)", xtitle="ICU hour"), "cl_lines")
        source(note + "Each agent reports only what its role covers (Step 2 agent table).")
    with right:
        section("Suggested actions in view")
        s = view["action"].value_counts().reindex(["keep monitoring", "order lab panel", "start antibiotics"]).fillna(0)
        show(donut(s.index, s.values, [BLUE, ORANGE, VIOLET], f"{int(s.sum()):,}<br>hours", height=250), "cl_donut")
        agree = int((~view["conflict"]).sum()); dis = int(view["conflict"].sum())
        f = go.Figure()
        f.add_trace(go.Bar(y=["team"], x=[agree], orientation="h", name="agrees", marker=dict(color=TEAL),
                           hovertemplate="agrees · %{x:,} hours<extra></extra>"))
        f.add_trace(go.Bar(y=["team"], x=[dis], orientation="h", name="disagrees", marker=dict(color=ORANGE),
                           hovertemplate="disagrees · %{x:,} hours<extra></extra>"))
        f.update_layout(barmode="stack"); f.update_yaxes(visible=False)
        show(style(f, height=110), "cl_agree")
        source(f"Hours where the team agrees ({agree:,}) or disagrees ({dis:,}).")

    section("Written summary for one hour", "Template writer from Step 7 · the treating clinician always decides")
    if len(p):
        c1, c2 = st.columns([2.2, 1], gap="large", vertical_alignment="center")
        with c1:
            hsel = st.select_slider("Hour", options=list(p["hour"]), value=int(p["hour"].iloc[-1]), key="cl_hour")
            r = p[p["hour"] == hsel].iloc[0]
            conflict_txt = (f"The team <b>disagrees</b> — nurse: {r['nurse_says']}, doctor: {r['doctor_says']}, AI: {r['ai_says']}. "
                            "Pooling the three probabilities settles it." if r["conflict"] else "The team <b>agrees</b>.")
            st.markdown(f"""<div class="card">
              <b>Patient {r['patient_id']} · ICU hour {int(r['hour'])}</b><br>
              <span class="agent">Nurse {r['nurse_p']:.3f}</span><span class="agent">Doctor {r['doctor_p']:.3f}</span>
              <span class="agent">AI tool {r['ai_p']:.3f}</span><span class="agent">Lab panels {int(r['labs_this_hour'])}</span>
              <span class="agent">Antibiotics {'running' if r['on_antibiotics'] else 'not running'}</span>
              <span class="agent">Beds free {int(r['beds_available'])}/22</span><span class="agent">Cost ${r['cost_so_far']:,.0f}</span><br>
              Combined probability of sepsis <b>{r['fused_p']:.3f}</b> against a treat threshold of {r['threshold']:.3f}. {conflict_txt}<br>
              Suggested next step: <b>{r['action']}</b>. This is a suggestion for the treating clinician, who decides.
            </div>""", unsafe_allow_html=True)
        with c2:
            show(gauge(float(r["fused_p"]), float(r["threshold"]), "Fused probability vs threshold", height=200), "cl_gauge")

# ===================================================== 7. OUTCOME & KPI ANALYSIS
with t7:
    ev = load("pomdp_evaluation.csv")
    sens = load("pomdp_sensitivity.csv")
    c1, c2, c3 = st.columns([1.1, 1.6, 1.3], vertical_alignment="bottom")
    with c1:
        sets = st.pills("Hospital set", ["A", "B"], selection_mode="multi", default=["A", "B"], key="kpi_set") or ["A", "B"]
    with c2:
        groups = st.pills("Age group", ["<45", "45-64", "65-79", "80+"], selection_mode="multi",
                          default=["<45", "45-64", "65-79", "80+"], key="kpi_age") or ["<45", "45-64", "65-79", "80+"]
    with c3:
        who = st.segmented_control("Patients", ["All", "Sepsis", "No sepsis"], default="All", key="kpi_who") or "All"

    d = df[df["hospital_set"].isin(sets) & df["age_group"].astype(str).isin(groups)]
    if who == "Sepsis":
        d = d[d["sepsis_patient"] == 1]
    elif who == "No sepsis":
        d = d[d["sepsis_patient"] == 0]
    pts = d.groupby("patient_id").agg(sepsis=("sepsis_patient", "first"), cost=("cum_cost_usd", "max"), hours=("ICULOS", "max"),
                                      sat=("satisfied_top_box", "first"), wait=("mean_lab_wait_min", "first"), occ=("occupancy_pct", "mean"))

    def delta(v, ref, higher_good=True, unit="pts", dec=1):
        dv = v - ref
        good = (dv >= 0) == higher_good
        return f'<span class="{"up" if good else "down"}">{"▲" if dv >= 0 else "▼"} {abs(dv):.{dec}f} {unit}</span>'

    k = st.columns(5)
    with k[0]: tile("Patients", f"{len(pts):,}", "in the current filter", BLUE)
    with k[1]: tile("Sepsis rate", f"{pts['sepsis'].mean() * 100:.2f}%", f"{delta(pts['sepsis'].mean()*100, 7.27, False, dec=2)} vs 7.27% (all PhysioNet)", RED)
    with k[2]: tile("Mean ICU cost", f"${pts['cost'].mean():,.0f}", f"mean stay {pts['hours'].mean():.0f} hours", VIOLET)
    with k[3]: tile("Top-box satisfaction", f"{pts['sat'].mean() * 100:.1f}%", f"{delta(pts['sat'].mean()*100, 72)} vs US 72% (HCAHPS)", TEAL)
    with k[4]: tile("Mean ICU occupancy", f"{pts['occ'].mean():.1f}%", f"vs US 68.2% (Wunsch, 2013)", ORANGE)

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        section("Which policy does best?", "Mean utility per patient: +1 correct treatment, −1 unnecessary antibiotics, −5 missed sepsis")
        PRETTY = {"pomdp": "POMDP policy", "never_treat": "Never treat", "news2_rule": "Treat when NEWS2 ≥ 5", "sirs_rule": "Treat when SIRS ≥ 2"}
        e = ev.copy(); e["policy"] = e["policy"].map(PRETTY).fillna(e["policy"]); e = e.sort_values("mean utility")
        f = go.Figure(go.Bar(y=e["policy"], x=e["mean utility"], orientation="h",
                             marker=dict(color=[VIOLET if p_ == "POMDP policy" else BLUE for p_ in e["policy"]], cornerradius=5),
                             text=[f"{v:.3f}" for v in e["mean utility"]], textposition="outside",
                             hovertemplate="%{y}<br>utility %{x:.3f}<extra></extra>"))
        headroom(f, list(e["mean utility"]), "x", 1.2)
        show(style(f, height=250, legend=False, xtitle="mean utility (closer to 0 is better)"), "kpi_pol")

        section("How that answer depends on the one weight with no published value", "Cost of unnecessary antibiotics (W_ABX) against a missed case (5)")
        f = go.Figure()
        for name, col, colr in [("POMDP", "POMDP utility", VIOLET), ("NEWS2 rule", "NEWS2 rule", BLUE),
                                ("SIRS rule", "SIRS rule", ORANGE), ("Never treat", "never treat", TEAL)]:
            f.add_trace(go.Scatter(x=sens["W_ABX"], y=sens[col], mode="lines+markers", name=name, line=dict(color=colr, width=2.4),
                                   marker=dict(size=9, line=dict(color="rgba(255,255,255,.8)", width=1.5)),
                                   hovertemplate=f"{name} · W_ABX %{{x}} · utility %{{y:.3f}}<extra></extra>"))
        show(style(f, height=300, ytitle="mean utility", xtitle="W_ABX"), "kpi_sens")
        source("Cheap antibiotics: acting early wins. Costly: the POMDP beats both rules, and doing nothing is hard to beat.")
    with right:
        section("Patient mix", "Share of patients in the current filter")
        counts = pts["sepsis"].value_counts().reindex([0, 1]).fillna(0)
        show(donut(["no sepsis", "sepsis"], counts.values, [BLUE, RED], f"{len(pts):,}<br>patients", height=230), "kpi_mix")

        section("ICU cost per patient", "Sepsis records stop about 3 hours after onset, so their stays look shorter")
        f = go.Figure()
        for lab, val, colr in [("no sepsis", 0, BLUE), ("sepsis", 1, RED)]:
            s = pts[pts["sepsis"] == val]
            if len(s):
                f.add_trace(go.Histogram(x=s["cost"], name=lab, nbinsx=40, opacity=.85, marker=dict(color=colr, cornerradius=3),
                                         hovertemplate=f"{lab} · $%{{x}} · %{{y}} patients<extra></extra>"))
        f.update_layout(barmode="overlay")
        show(style(f, height=240, ytitle="patients", xtitle="ICU cost ($)"), "kpi_cost")

        section("Satisfaction against lab waiting time", "Dotted line = US average 72% (HCAHPS)")
        w = pts.dropna(subset=["wait"]).copy()
        w["band"] = pd.cut(w["wait"], [0, 30, 40, 50, 60, 200], labels=["<30", "30–40", "40–50", "50–60", "60+"])
        s = w.groupby("band", observed=True).agg(rate=("sat", "mean"), n=("sat", "size")).reset_index()
        f = go.Figure(go.Bar(x=s["band"].astype(str), y=s["rate"] * 100, marker=dict(color=TEAL, cornerradius=5),
                             text=[f"{v*100:.0f}%" for v in s["rate"]], textposition="outside", customdata=s["n"],
                             hovertemplate="wait %{x} min<br>%{y:.1f}% top box<br>n = %{customdata}<extra></extra>"))
        f.add_hline(y=72, line=dict(color=NEUTRAL, width=1.5, dash="dot"))
        f.update_yaxes(range=[0, 108])
        show(style(f, height=250, legend=False, ytitle="% top score", xtitle="mean lab wait (minutes)"), "kpi_sat")
        source("Curve shape from Bleustein et al. (2014); level set to the HCAHPS national average.")
