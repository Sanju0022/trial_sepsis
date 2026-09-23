"""
Group Project 3 - Multi-Agent Healthcare Decision Support
Streamlit dashboard (PDF Task 5). Every tab has its own local filters.

Run:  streamlit run app.py
Data: the CSV files produced by Steps 3-7, in ./data
"""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ---------------------------------------------------------------- design tokens
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # categorical slots 1-4
CRIT, VIOLET = "#e34948", "#4a3aa7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8c8b85"
SURFACE, GRID = "#fcfcfb", "#e7e6e1"

st.set_page_config(page_title="Multi-Agent Healthcare Decision Support",
                   page_icon="+", layout="wide")

st.markdown(f"""
<style>
  .block-container {{padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px;}}
  h1, h2, h3 {{color: {INK}; letter-spacing: -0.01em;}}
  .hdr {{background: linear-gradient(90deg, #eaf2fd 0%, #f6faff 55%, {SURFACE} 100%);
         border: 1px solid {GRID}; border-radius: 14px; padding: 18px 22px; margin-bottom: 14px;}}
  .hdr h1 {{margin: 0 0 4px 0; font-size: 1.55rem;}}
  .hdr p {{margin: 0; color: {INK2}; font-size: 0.92rem;}}
  .tile {{background: {SURFACE}; border: 1px solid {GRID}; border-left: 4px solid var(--accent, {S1});
          border-radius: 12px; padding: 12px 14px; height: 100%;}}
  .tile .label {{color: {MUTED}; font-size: 0.72rem; text-transform: uppercase; letter-spacing: .07em;}}
  .tile .value {{color: {INK}; font-size: 1.5rem; font-weight: 700; line-height: 1.25; margin-top: 2px;}}
  .tile .note {{color: {INK2}; font-size: 0.78rem;}}
  .src {{color: {MUTED}; font-size: 0.76rem; border-top: 1px dashed {GRID}; padding-top: 6px; margin-top: 2px;}}
  div[data-testid="stTabs"] button p {{font-size: 0.95rem; font-weight: 600;}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data loading
@st.cache_data(show_spinner=False)
def load(name):
    return pd.read_csv(os.path.join(DATA_DIR, name))


@st.cache_data(show_spinner=False)
def load_main():
    d = load("sepsis_multiagent_dataset.csv").sort_values(["patient_id", "ICULOS"])
    d["sirs_count"] = (((d["Temp"] > 38) | (d["Temp"] < 36)).astype(int) + (d["HR"] > 90).astype(int)
                       + (d["Resp"] > 20).astype(int) + ((d["WBC"] > 12) | (d["WBC"] < 4)).astype(int))
    d["age_group"] = pd.cut(d["Age"], [0, 45, 65, 80, 120], labels=["<45", "45-64", "65-79", "80+"])
    return d


def tile(label, value, note="", accent=S1):
    st.markdown(f"""<div class="tile" style="--accent:{accent}">
      <div class="label">{label}</div><div class="value">{value}</div>
      <div class="note">{note}</div></div>""", unsafe_allow_html=True)


def style(fig, height=340, legend=True, ytitle="", xtitle=""):
    fig.update_layout(
        height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        margin=dict(l=8, r=12, t=34, b=8), font=dict(color=INK2, size=12.5),
        title=dict(font=dict(color=INK, size=14.5), x=0, xanchor="left"),
        hovermode="x unified" if fig.data and fig.data[0].type in ("scatter",) else "closest",
        showlegend=legend,
        legend=dict(orientation="h", y=1.14, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11.5)),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRID, ticks="outside",
                     tickcolor=GRID, title=dict(text=xtitle, font=dict(size=11.5)))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)",
                     title=dict(text=ytitle, font=dict(size=11.5)))
    return fig


def pad_x(fig, values, factor=1.22):
    fig.update_xaxes(range=[0, float(max(values)) * factor])
    return fig


def pad_y(fig, values, factor=1.22):
    fig.update_yaxes(range=[0, float(max(values)) * factor])
    return fig


def source_note(text):
    st.markdown(f'<div class="src">{text}</div>', unsafe_allow_html=True)


df = load_main()

st.markdown("""<div class="hdr">
  <h1>Multi-Agent Healthcare Decision Support &mdash; Sepsis in the ICU</h1>
  <p>Real patient data: PhysioNet/CinC Challenge 2019 (CC BY 4.0) &middot; hospital operations layer built from cited sources &middot;
     POMDP, bandits, Bayesian inference and MLE estimated in Steps 4&ndash;7. Each tab has its own filters.</p>
</div>""", unsafe_allow_html=True)

TABS = ["Patient Overview", "POMDP Belief Tracker", "Bandit Decision Explorer", "Bayesian Network",
        "MLE Parameter Insights", "Agent Collaboration Log", "Outcome & KPI Analysis"]
t1, t2, t3, t4, t5, t6, t7 = st.tabs(TABS)

# ============================================================ 1. PATIENT OVERVIEW
with t1:
    c1, c2, c3 = st.columns([2, 1.3, 1.6])
    with c1:
        only_sepsis = st.checkbox("Only patients who developed sepsis", value=True, key="po_sep")
        pool = df[df["sepsis_patient"] == 1] if only_sepsis else df
        pid = st.selectbox("Patient", sorted(pool["patient_id"].unique()), key="po_pid")
    g = df[df["patient_id"] == pid]
    with c2:
        hrs = st.slider("Hours shown", int(g["ICULOS"].min()), int(g["ICULOS"].max()),
                        (int(g["ICULOS"].min()), int(g["ICULOS"].max())), key="po_hrs") \
            if g["ICULOS"].max() > g["ICULOS"].min() else (int(g["ICULOS"].min()), int(g["ICULOS"].max()))
    with c3:
        shown = st.multiselect("Vital signs", ["HR", "Resp", "Temp", "O2Sat", "SBP", "MAP"],
                               default=["HR", "Resp", "Temp", "O2Sat"], key="po_vitals")
    g = g[(g["ICULOS"] >= hrs[0]) & (g["ICULOS"] <= hrs[1])]
    row0 = g.iloc[0]

    k = st.columns(6)
    with k[0]: tile("Age / sex", f"{row0['Age']:.0f} / {'M' if row0['Gender'] == 1 else 'F'}", f"hospital set {row0['hospital_set']}")
    with k[1]: tile("Hours shown", f"{len(g)}", f"of {int(df[df.patient_id == pid]['ICULOS'].max())} in record", S3)
    with k[2]:
        sep = int(row0["sepsis_patient"])
        tile("Sepsis", "Yes" if sep else "No",
             f"onset hour {row0['sepsis_onset_hour']:.0f}" if sep else "never labelled", CRIT if sep else S3)
    with k[3]: tile("NEWS2 (max)", f"{g['news2_score'].max():.0f}", f"median {g['news2_score'].median():.0f}", S2)
    with k[4]: tile("Lab panels", f"{int(g['panels_ordered'].sum())}", f"{g['lab_tat_max_min'].mean():.0f} min mean wait" if g['panels_ordered'].sum() else "none ordered", S4)
    with k[5]: tile("ICU cost", f"${g['cum_cost_usd'].max():,.0f}", "cumulative, $179.17/hour", VIOLET)

    st.markdown("")
    left, right = st.columns([1.55, 1])
    with left:
        st.markdown("**Vital signs, hour by hour** — one panel per sign, because the scales differ")
        for i, v in enumerate(shown):
            f = go.Figure()
            f.add_trace(go.Scatter(x=g["ICULOS"], y=g[v], mode="lines", line=dict(color=S1, width=2),
                                   name=v, connectgaps=False, hovertemplate=f"hour %{{x}} · {v} %{{y:.1f}}<extra></extra>"))
            meas = g[g[f"{v}_measured"] == 1]
            f.add_trace(go.Scatter(x=meas["ICULOS"], y=meas[v], mode="markers", name="measured",
                                   marker=dict(color=S1, size=7, line=dict(color=SURFACE, width=2)),
                                   hovertemplate="measured this hour<extra></extra>"))
            if int(row0["sepsis_patient"]) == 1:
                f.add_vline(x=float(row0["sepsis_onset_hour"]), line=dict(color=CRIT, width=2, dash="dot"),
                            annotation_text="sepsis onset", annotation_position="top left",
                            annotation_font_color=CRIT, annotation_font_size=11)
            st.plotly_chart(style(f, height=180, legend=(i == 0), ytitle=v, xtitle="ICU hour" if i == len(shown) - 1 else ""),
                            use_container_width=True, key=f"po_v{v}")
        source_note("Line = value carried forward from the last measurement · dot = actually measured that hour (PhysioNet 2019).")
    with right:
        st.markdown("**NEWS2 early-warning score**")
        f = go.Figure()
        f.add_hrect(y0=0, y1=4.5, fillcolor=S3, opacity=.07, line_width=0)
        f.add_hrect(y0=4.5, y1=6.5, fillcolor=S4, opacity=.10, line_width=0)
        f.add_hrect(y0=6.5, y1=20, fillcolor=CRIT, opacity=.08, line_width=0)
        f.add_trace(go.Scatter(x=g["ICULOS"], y=g["news2_score"], mode="lines", line=dict(color=INK, width=2),
                               name="NEWS2", hovertemplate="hour %{x} · NEWS2 %{y:.0f}<extra></extra>"))
        f.add_annotation(x=g["ICULOS"].max(), y=7.6, text="high (≥7)", showarrow=False, font=dict(color=CRIT, size=11), xanchor="right")
        f.add_annotation(x=g["ICULOS"].max(), y=5.5, text="medium (5–6)", showarrow=False, font=dict(color="#a06c00", size=11), xanchor="right")
        st.plotly_chart(style(f, height=250, legend=False, ytitle="NEWS2", xtitle="ICU hour"),
                        use_container_width=True, key="po_news2")
        source_note("Bands: Royal College of Physicians NEWS2 (2017). 5 of 7 parameters available.")

        st.markdown("**Tests ordered and turnaround**")
        f = go.Figure()
        for name, col, colr in [("CBC", "cbc_ordered", S1), ("Chemistry", "chem_ordered", S2), ("Coagulation", "coag_ordered", S3)]:
            sel = g[g[col] == 1]
            f.add_trace(go.Bar(x=sel["ICULOS"], y=sel[col.replace("ordered", "tat_min")], name=name,
                               marker=dict(color=colr, line=dict(color=SURFACE, width=2)),
                               hovertemplate=f"{name} · hour %{{x}} · %{{y:.0f}} min<extra></extra>"))
        f.update_layout(barmode="group", bargap=.25)
        st.plotly_chart(style(f, height=230, ytitle="minutes to result", xtitle="ICU hour"),
                        use_container_width=True, key="po_labs")
        source_note("Turnaround drawn from published stat-test times (Fei et al., Biochemia Medica 2015).")

# ======================================================= 2. POMDP BELIEF TRACKER
with t2:
    tr = load("pomdp_belief_traces.csv")
    pol = load("pomdp_policy.csv")
    par = load("pomdp_parameters.csv").set_index("parameter")["value"]
    thr = {int(h): float(gg.loc[gg["best_action"] == "treat", "belief_p_sepsis"].min()) for h, gg in pol.groupby("hours_left")}

    c1, c2, c3 = st.columns([1.5, 1.3, 1.6])
    with c1:
        grp = st.radio("Patients", ["Sepsis patients", "All patients"], horizontal=True, key="pb_grp")
        pool = tr[tr["sepsis"] == 1] if grp == "Sepsis patients" else tr
    with c2:
        pid2 = st.selectbox("Patient", sorted(pool["patient_id"].unique()), key="pb_pid")
    with c3:
        hleft = st.select_slider("Treat threshold for hours left", options=sorted(thr.keys()), value=24, key="pb_hl")

    b = tr[tr["patient_id"] == pid2]
    k = st.columns(5)
    with k[0]: tile("Start belief", f"{par['P_INIT']:.3f}", "MLE: sepsis at admission", S1)
    with k[1]: tile("Peak belief", f"{b['belief_p_sepsis'].max():.3f}", f"hour {int(b.loc[b['belief_p_sepsis'].idxmax(), 'hour'])}", S2)
    with k[2]: tile("Final belief", f"{b['belief_p_sepsis'].iloc[-1]:.3f}", f"{len(b)} hours followed", S3)
    with k[3]: tile("Treat threshold", f"{thr[hleft]:.3f}", f"with {hleft} h left in stay", CRIT)
    with k[4]:
        tt = b["treated_at"].iloc[0]
        tile("POMDP decision", "Treated" if pd.notna(tt) else "Kept monitoring",
             f"at hour {tt:.0f}" if pd.notna(tt) else "belief stayed below threshold", VIOLET)

    left, right = st.columns([1.5, 1])
    with left:
        st.markdown("**Belief that this patient has sepsis, updated by Bayes' rule each hour**")
        f = go.Figure()
        f.add_trace(go.Scatter(x=b["hour"], y=b["belief_p_sepsis"], mode="lines", name="belief P(sepsis)",
                               line=dict(color=S1, width=2.5), fill="tozeroy", fillcolor="rgba(42,120,214,0.10)",
                               hovertemplate="hour %{x} · belief %{y:.3f}<extra></extra>"))
        top = max(b["belief_p_sepsis"].max() * 1.45, 0.02)
        if thr[hleft] <= top:
            f.add_hline(y=thr[hleft], line=dict(color=CRIT, width=2, dash="dash"))
            f.add_annotation(x=b["hour"].max(), y=thr[hleft], text=f"treat threshold {thr[hleft]:.3f}",
                             showarrow=False, yshift=10, xanchor="right", font=dict(color=CRIT, size=11))
        else:
            f.add_annotation(x=b["hour"].min(), y=top, xanchor="left", showarrow=False, font=dict(color=CRIT, size=11),
                             text=f"treat threshold {thr[hleft]:.3f} — above this chart, so the policy keeps monitoring")
        f.update_yaxes(range=[0, top])
        if pd.notna(b["treated_at"].iloc[0]):
            f.add_vline(x=float(b["treated_at"].iloc[0]), line=dict(color=VIOLET, width=2, dash="dot"),
                        annotation_text="antibiotics", annotation_font_color=VIOLET, annotation_font_size=11)
        st.plotly_chart(style(f, height=330, legend=False, ytitle="P(sepsis)", xtitle="ICU hour"),
                        use_container_width=True, key="pb_trace")
        source_note("Transition 0.0017/hour and test accuracies are MLE estimates from Step 4; belief update is Bayes' rule.")
    with right:
        st.markdown("**When does the policy say treat?**")
        f = go.Figure()
        for i, h in enumerate(sorted(thr.keys())):
            sub = pol[pol["hours_left"] == h]
            f.add_trace(go.Scatter(x=sub["belief_p_sepsis"], y=sub["value"], mode="lines", name=f"{h} h left",
                                   line=dict(color=[S1, S2, S3][i % 3], width=2),
                                   hovertemplate=f"{h} h left · belief %{{x:.2f}} · value %{{y:.2f}}<extra></extra>"))
            f.add_vline(x=thr[h], line=dict(color=[S1, S2, S3][i % 3], width=1.5, dash="dot"))
        st.plotly_chart(style(f, height=330, ytitle="expected value", xtitle="belief P(sepsis)"),
                        use_container_width=True, key="pb_policy")
        source_note("Dotted line = the belief where the best action switches from monitoring to antibiotics.")

    st.markdown("**Where every tracked patient ended up**")
    fin = tr.sort_values("hour").groupby("patient_id").tail(1)
    f = go.Figure()
    for lab, val, colr in [("no sepsis", 0, S1), ("sepsis", 1, CRIT)]:
        s = fin[fin["sepsis"] == val]
        f.add_trace(go.Histogram(x=s["belief_p_sepsis"], name=lab, marker=dict(color=colr, line=dict(color=SURFACE, width=2)),
                                 opacity=.85, nbinsx=40, hovertemplate=f"{lab} · belief %{{x}} · %{{y}} patients<extra></extra>"))
    f.update_layout(barmode="overlay")
    st.plotly_chart(style(f, height=260, ytitle="patients", xtitle="final belief P(sepsis)"),
                    use_container_width=True, key="pb_hist")
    source_note(f"{fin['patient_id'].nunique()} patients tracked in Step 5 (the traces file holds the first 200 patients).")

# =================================================== 3. BANDIT DECISION EXPLORER
with t3:
    bt = load("bandit_treatment.csv")
    bp = load("bandit_policy.csv")
    sh = load("bandit_arm_shares.csv")

    c1, c2 = st.columns([1.4, 2])
    with c1:
        upto = st.slider("Patients seen by the bandit", 100, int(bt["patient"].max()), int(bt["patient"].max()), step=100, key="bd_upto")
    with c2:
        algos = st.multiselect("Algorithms", ["Thompson sampling", "UCB1", "Random choice"],
                               default=["Thompson sampling", "UCB1", "Random choice"], key="bd_alg")
    b = bt[bt["patient"] <= upto]
    cols = {"Thompson sampling": ("thompson_mean_regret", S1), "UCB1": ("ucb1_mean_regret", S2), "Random choice": ("random_choice_regret", MUTED)}

    k = st.columns(4)
    with k[0]: tile("Arm A", "antibiotics ≤ 1 h", "survival 83.4% (MLE, 1,526 patients)", S1)
    with k[1]: tile("Arm B", "antibiotics > 1 h", "survival 80.6% (MLE, 1,406 patients)", S2)
    for i, name in enumerate(["Thompson sampling", "UCB1"]):
        with k[2 + i]:
            v = b[cols[name][0]].iloc[-1]
            tile(f"{name} regret", f"{v:.1f}", f"after {upto:,} patients; random = {b['random_choice_regret'].iloc[-1]:.1f}", cols[name][1])

    left, right = st.columns([1.5, 1])
    with left:
        st.markdown("**Learning which timing is better — cumulative regret (lower is better)**")
        f = go.Figure()
        for name in algos:
            col, colr = cols[name]
            f.add_trace(go.Scatter(x=b["patient"], y=b[col], mode="lines", name=name, line=dict(color=colr, width=2),
                                   hovertemplate=f"{name} · patient %{{x}} · regret %{{y:.1f}}<extra></extra>"))
        st.plotly_chart(style(f, height=330, ytitle="cumulative regret", xtitle="patients treated"),
                        use_container_width=True, key="bd_regret")
        source_note("Regret = survival lost against always choosing the better timing. Mean of 30 runs.")
    with right:
        st.markdown("**Which ward policy does the bandit settle on?**")
        BNAME = {"thompson": "Thompson sampling", "thompson_contextual": "Thompson, per patient group"}
        who = st.radio("Bandit", sorted(sh["bandit"].unique()), horizontal=True,
                       format_func=lambda v: BNAME.get(v, v), key="bd_who")
        s = sh[sh["bandit"] == who].groupby("arm", as_index=False)["times_chosen"].sum().sort_values("times_chosen")
        f = go.Figure(go.Bar(x=s["times_chosen"], y=s["arm"], orientation="h",
                             marker=dict(color=S3, line=dict(color=SURFACE, width=2)),
                             text=s["times_chosen"], textposition="outside", textfont=dict(color=INK2),
                             hovertemplate="%{y} · chosen %{x} times<extra></extra>"))
        st.plotly_chart(pad_x(style(f, height=280, legend=False, xtitle="times chosen"), s["times_chosen"]),
                        use_container_width=True, key="bd_shares")
        source_note("Arms are complete ward rules; the bandit only sees the reward of the arm it picks.")

    st.markdown("**Choosing among ward policies — regret under two views of how costly unnecessary antibiotics are**")
    c1, c2 = st.columns([1.2, 3])
    with c1:
        w = st.radio("Cost of unnecessary antibiotics (W_ABX)", sorted(bp["w_abx"].unique()),
                     format_func=lambda v: f"{v} — {'cheap' if v < 1 else 'costly'}", key="bd_w")
    with c2:
        s = bp[bp["w_abx"] == w]
        f = go.Figure()
        for name, col, colr in [("Thompson", "thompson", S1), ("UCB1", "ucb1", S2), ("Thompson, per patient group", "thompson_contextual", S3)]:
            f.add_trace(go.Scatter(x=s["patient"], y=s[col], mode="lines", name=name, line=dict(color=colr, width=2),
                                   hovertemplate=f"{name} · patient %{{x}} · regret %{{y:.0f}}<extra></extra>"))
        st.plotly_chart(style(f, height=300, ytitle="cumulative regret", xtitle="patients"),
                        use_container_width=True, key="bd_pol")
    source_note("With costly antibiotics every patient group has the same best arm, so the contextual bandit only pays the extra exploration.")

# ======================================================= 4. BAYESIAN NETWORK
with t4:
    cpt = load("bayes_network.csv")
    fus = load("bayes_fusion.csv")
    LABELS = {"temp_abn": "Temperature > 38 or < 36 °C", "hr_abn": "Heart rate > 90",
              "rr_abn": "Respiratory rate > 20", "wbc_abn": "WBC > 12 or < 4"}

    st.markdown("**Try it: pick what the patient shows and watch the posterior move**")
    c = st.columns([1.1, 1, 1, 1, 1])
    with c[0]:
        prior = st.slider("Prior P(sepsis) this hour", 0.005, 0.30, 0.018, 0.001, key="bn_prior")
    picks = {}
    for i, s in enumerate(cpt["sign"]):
        with c[i + 1]:
            picks[s] = st.checkbox(LABELS[s], value=(s in ("hr_abn", "rr_abn")), key=f"bn_{s}")

    lo = np.log(prior / (1 - prior))
    steps, running = [], lo
    for _, r in cpt.iterrows():
        lr = r["LR if present"] if picks[r["sign"]] else r["LR if absent"]
        running += np.log(lr)
        short = {"temp_abn": "Temp", "hr_abn": "HR", "rr_abn": "RR", "wbc_abn": "WBC"}[r["sign"]]
        steps.append((short + (" present" if picks[r["sign"]] else " absent"), np.log(lr), running))
    post = 1 / (1 + np.exp(-running))

    k = st.columns(4)
    with k[0]: tile("Prior", f"{prior:.3f}", "before any sign is read", MUTED)
    with k[1]: tile("Posterior", f"{post:.3f}", f"{post / prior:.1f}× the prior", CRIT if post > 0.1 else S1)
    with k[2]: tile("Signs present", f"{sum(picks.values())} of 4", "SIRS criteria (Bone et al., 1992)", S2)
    with k[3]: tile("Strongest sign", "Temperature", "likelihood ratio 1.95 when present", S4)

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("**How each sign moves the log-odds**")
        f = go.Figure(go.Waterfall(
            orientation="v", measure=["absolute"] + ["relative"] * len(steps) + ["total"],
            x=["prior"] + [s[0] for s in steps] + ["posterior"],
            y=[lo] + [s[1] for s in steps] + [0],
            connector=dict(line=dict(color=GRID)),
            increasing=dict(marker=dict(color=CRIT)), decreasing=dict(marker=dict(color=S3)),
            totals=dict(marker=dict(color=S1)),
            hovertemplate="%{x}<br>log-odds %{y:+.3f}<extra></extra>"))
        st.plotly_chart(style(f, height=330, legend=False, ytitle="log-odds of sepsis"),
                        use_container_width=True, key="bn_wf")
        source_note("Conditional probability tables estimated from 55,735 real patient-hours (Step 6).")
    with right:
        st.markdown("**Likelihood ratios in the network**")
        f = go.Figure()
        f.add_trace(go.Bar(y=[LABELS[s] for s in cpt["sign"]], x=cpt["LR if present"], orientation="h", name="sign present",
                           marker=dict(color=CRIT, line=dict(color=SURFACE, width=2)),
                           hovertemplate="%{y}<br>present: ×%{x:.2f}<extra></extra>"))
        f.add_trace(go.Bar(y=[LABELS[s] for s in cpt["sign"]], x=cpt["LR if absent"], orientation="h", name="sign absent",
                           marker=dict(color=S3, line=dict(color=SURFACE, width=2)),
                           hovertemplate="%{y}<br>absent: ×%{x:.2f}<extra></extra>"))
        f.add_vline(x=1, line=dict(color=MUTED, width=1.5, dash="dot"))
        f.update_layout(barmode="group", bargap=.3)
        st.plotly_chart(style(f, height=330, xtitle="multiplies the odds by"),
                        use_container_width=True, key="bn_lr")
        source_note("Above 1 raises the odds of sepsis, below 1 lowers them.")

    st.markdown("**Fusing the nurse, doctor and AI opinions** — held-out 453 patients")
    c1, c2 = st.columns([1.4, 1])
    with c1:
        metric = st.radio("Measure", ["Ranking ability (area under ROC)", "Accuracy of the number (Brier, lower is better)"],
                          horizontal=True, key="bn_metric")
        col = "area under ROC" if metric.startswith("Ranking") else "Brier score"
        s = fus.sort_values(col, ascending=(col != "area under ROC"))
        f = go.Figure(go.Bar(y=s["opinion"], x=s[col], orientation="h",
                             marker=dict(color=[S1 if "fused" not in o else CRIT for o in s["opinion"]],
                                         line=dict(color=SURFACE, width=2)),
                             text=[f"{v:.3f}" for v in s[col]], textposition="outside", textfont=dict(color=INK2),
                             hovertemplate="%{y}<br>%{x:.4f}<extra></extra>"))
        if col == "area under ROC":
            f.add_vline(x=0.5, line=dict(color=MUTED, width=1.5, dash="dot"))
        st.plotly_chart(style(f, height=270, legend=False, xtitle=col), use_container_width=True, key="bn_fus")
    with c2:
        st.markdown("")
        tile("Best ranking", "fused opinion", "area under ROC 0.700 vs 0.687 for the AI alone", CRIT)
        st.markdown("")
        tile("But over-confident", "predicts 2.8%", "when the real rate is 1.4% — the three opinions read the same vitals", S2)
    source_note("Fusion pools log-odds, which assumes the three opinions are independent given the true state. They are not, and the Brier score shows it.")

# ==================================================== 5. MLE PARAMETER INSIGHTS
with t5:
    mle = load("mle_parameters.csv")
    NICE = {"progression": "Disease progression", "test_accuracy": "Test accuracy",
            "treatment_success": "Treatment success", "satisfaction_curve": "Satisfaction curve"}
    c1, c2 = st.columns([1.6, 2])
    with c1:
        item = st.selectbox("Estimate group", list(NICE), format_func=lambda k: NICE[k], key="mle_item")
    sub = mle[mle["item"] == item].copy()
    with c2:
        params = st.multiselect("Parameters", sorted(sub["parameter"].unique()),
                                default=sorted(sub["parameter"].unique()), key="mle_par")
    sub = sub[sub["parameter"].isin(params)]
    sub["label"] = sub["group"] + "  ·  " + sub["parameter"].str.replace("_", " ")

    k = st.columns(4)
    with k[0]: tile("Estimates shown", f"{len(sub)}", NICE[item], S1)
    with k[1]: tile("Widest interval", f"{(sub['ci_high'] - sub['ci_low']).max():.3f}", "least certain estimate", S2)
    with k[2]: tile("Smallest sample", f"{int(sub['n'].min()):,}", "observations behind an estimate", S4)
    with k[3]: tile("Method", "Maximum likelihood", "95% intervals: patient-level bootstrap or Wald", VIOLET)

    st.markdown("**Estimate with its 95% confidence interval**")
    f = go.Figure()
    f.add_trace(go.Scatter(
        x=sub["mle"], y=sub["label"], mode="markers", name="MLE",
        marker=dict(color=S1, size=11, line=dict(color=SURFACE, width=2)),
        error_x=dict(type="data", symmetric=False, array=sub["ci_high"] - sub["mle"],
                     arrayminus=sub["mle"] - sub["ci_low"], color=S1, thickness=2, width=6),
        customdata=np.stack([sub["ci_low"], sub["ci_high"], sub["n"]], axis=-1),
        hovertemplate="%{y}<br>MLE %{x:.4f}<br>95% CI %{customdata[0]:.4f} – %{customdata[1]:.4f}<br>n = %{customdata[2]:,}<extra></extra>"))
    st.plotly_chart(style(f, height=120 + 42 * len(sub), legend=False, xtitle="estimated value"),
                    use_container_width=True, key="mle_dot")
    source_note("Estimated in Step 4 from the 2,000-patient dataset; intervals for hourly quantities resample whole patients.")

    with st.expander("See the numbers"):
        st.dataframe(sub[["parameter", "group", "mle", "ci_low", "ci_high", "n"]]
                     .rename(columns={"mle": "MLE", "ci_low": "CI low", "ci_high": "CI high", "n": "observations"}),
                     use_container_width=True, hide_index=True)

# ==================================================== 6. AGENT COLLABORATION LOG
with t6:
    log = load("collaboration_log.csv")
    c1, c2, c3 = st.columns([1.3, 1.3, 1.6])
    with c1:
        only_conf = st.checkbox("Only hours where the team disagrees", value=False, key="cl_conf")
    with c2:
        acts = st.multiselect("Suggested action", sorted(log["action"].unique()),
                              default=sorted(log["action"].unique()), key="cl_act")
    view = log[log["action"].isin(acts)]
    if only_conf:
        view = view[view["conflict"]]
    with c3:
        pid3 = st.selectbox("Patient", sorted(view["patient_id"].unique()) if len(view) else ["—"], key="cl_pid")

    k = st.columns(5)
    with k[0]: tile("Hours in view", f"{len(view):,}", f"of {len(log):,} logged", S1)
    with k[1]: tile("Disagreement", f"{log['conflict'].mean() * 100:.1f}%", "of all logged hours", S2)
    with k[2]: tile("Antibiotics suggested", f"{(log['action'] == 'start antibiotics').mean() * 100:.1f}%", "of all logged hours", CRIT)
    with k[3]:
        sep_h = log[log["sepsis_label"] == 1]
        tile("...in true sepsis hours", f"{(sep_h['action'] == 'start antibiotics').mean() * 100:.1f}%", f"{len(sep_h)} such hours", S3)
    with k[4]: tile("Patients logged", f"{log['patient_id'].nunique()}", "first 200 of the dataset", VIOLET)

    left, right = st.columns([1.5, 1])
    with left:
        st.markdown(f"**Patient {pid3} — what each agent thought, hour by hour**")
        p = view[view["patient_id"] == pid3].sort_values("hour")
        f = go.Figure()
        for name, col, colr in [("Nurse (NEWS2)", "nurse_p", S1), ("Doctor (SIRS)", "doctor_p", S2),
                                ("AI tool", "ai_p", S3), ("Fused", "fused_p", CRIT)]:
            f.add_trace(go.Scatter(x=p["hour"], y=p[col], mode="lines", name=name,
                                   line=dict(color=colr, width=3 if name == "Fused" else 2,
                                             dash="solid" if name == "Fused" else "dot"),
                                   hovertemplate=f"{name} · hour %{{x}} · P(sepsis) %{{y:.3f}}<extra></extra>"))
        if len(p):
            top = max(p[["nurse_p", "doctor_p", "ai_p", "fused_p"]].to_numpy().max() * 1.5, 0.02)
            if p["threshold"].min() <= top:
                f.add_trace(go.Scatter(x=p["hour"], y=p["threshold"], mode="lines", name="treat threshold",
                                       line=dict(color=MUTED, width=1.5, dash="dash"),
                                       hovertemplate="threshold %{y:.3f}<extra></extra>"))
            else:
                f.add_annotation(x=p["hour"].min(), y=top, xanchor="left", showarrow=False,
                                 font=dict(color=MUTED, size=11),
                                 text=f"treat threshold {p['threshold'].min():.3f}–{p['threshold'].max():.3f} — above this chart")
            f.update_yaxes(range=[0, top])
        st.plotly_chart(style(f, height=320, ytitle="P(sepsis)", xtitle="ICU hour"),
                        use_container_width=True, key="cl_lines")
        source_note("Each agent reports only what its role covers; the fused line is the pooled posterior that decides.")
    with right:
        st.markdown("**Suggested actions in view**")
        s = view["action"].value_counts().reset_index()
        s.columns = ["action", "hours"]
        f = go.Figure(go.Bar(y=s["action"], x=s["hours"], orientation="h",
                             marker=dict(color=[CRIT if a == "start antibiotics" else S2 if a == "order lab panel" else S1 for a in s["action"]],
                                         line=dict(color=SURFACE, width=2)),
                             text=s["hours"], textposition="outside", textfont=dict(color=INK2),
                             hovertemplate="%{y} · %{x} hours<extra></extra>"))
        st.plotly_chart(pad_x(style(f, height=200, legend=False, xtitle="hours"), s["hours"]),
                        use_container_width=True, key="cl_acts")

        st.markdown("**Agreement**")
        agree = pd.DataFrame({"state": ["team agrees", "team disagrees"],
                              "hours": [(~view["conflict"]).sum(), view["conflict"].sum()]})
        f = go.Figure(go.Bar(x=agree["state"], y=agree["hours"],
                             marker=dict(color=[S3, S2], line=dict(color=SURFACE, width=2)),
                             text=agree["hours"], textposition="outside", textfont=dict(color=INK2),
                             hovertemplate="%{x} · %{y} hours<extra></extra>"))
        st.plotly_chart(pad_y(style(f, height=200, legend=False, ytitle="hours"), agree["hours"]),
                        use_container_width=True, key="cl_agree")

    st.markdown("**Written summary for one hour**")
    if len(p):
        hsel = st.select_slider("Hour", options=list(p["hour"]), value=int(p["hour"].iloc[-1]), key="cl_hour")
        r = p[p["hour"] == hsel].iloc[0]
        conflict_txt = ("The team disagrees — nurse says {}, doctor says {}, AI says {}. Pooling the three "
                        "probabilities settles it.".format(r["nurse_says"], r["doctor_says"], r["ai_says"])
                        if r["conflict"] else "The team agrees.")
        st.info(f"**Patient {r['patient_id']}, ICU hour {int(r['hour'])}.** Nurse {r['nurse_p']:.3f} · doctor "
                f"{r['doctor_p']:.3f} · AI {r['ai_p']:.3f}. Combined probability of sepsis **{r['fused_p']:.3f}** "
                f"against a treat threshold of {r['threshold']:.3f}. {conflict_txt} "
                f"Lab panels this hour: {int(r['labs_this_hour'])}. Antibiotics running: "
                f"{'yes' if r['on_antibiotics'] else 'no'}. Beds free: {int(r['beds_available'])} of 22. "
                f"Cost so far ${r['cost_so_far']:,.0f}. **Suggested next step: {r['action']}.** "
                f"This is a suggestion for the treating clinician, who decides.")

# ===================================================== 7. OUTCOME & KPI ANALYSIS
with t7:
    ev = load("pomdp_evaluation.csv")
    sens = load("pomdp_sensitivity.csv")

    c1, c2, c3 = st.columns([1.3, 1.3, 1.6])
    with c1:
        sets = st.multiselect("Hospital set", ["A", "B"], default=["A", "B"], key="kpi_set")
    with c2:
        groups = st.multiselect("Age group", ["<45", "45-64", "65-79", "80+"],
                                default=["<45", "45-64", "65-79", "80+"], key="kpi_age")
    with c3:
        who = st.radio("Patients", ["All", "Sepsis only", "No sepsis"], horizontal=True, key="kpi_who")

    d = df[df["hospital_set"].isin(sets) & df["age_group"].astype(str).isin(groups)]
    if who == "Sepsis only":
        d = d[d["sepsis_patient"] == 1]
    elif who == "No sepsis":
        d = d[d["sepsis_patient"] == 0]
    pts = d.groupby("patient_id").agg(sepsis=("sepsis_patient", "first"), cost=("cum_cost_usd", "max"),
                                      hours=("ICULOS", "max"), sat=("satisfied_top_box", "first"),
                                      wait=("mean_lab_wait_min", "first"), occ=("occupancy_pct", "mean"))
    k = st.columns(5)
    with k[0]: tile("Patients", f"{len(pts):,}", "in the current filter", S1)
    with k[1]: tile("Sepsis rate", f"{pts['sepsis'].mean() * 100:.2f}%", "real PhysioNet rate is 7.27%", CRIT)
    with k[2]: tile("Mean ICU cost", f"${pts['cost'].mean():,.0f}", f"mean stay {pts['hours'].mean():.0f} hours", VIOLET)
    with k[3]: tile("Top-box satisfaction", f"{pts['sat'].mean() * 100:.1f}%", "US average is 72% (HCAHPS)", S3)
    with k[4]: tile("Mean ICU occupancy", f"{pts['occ'].mean():.1f}%", "US average is 68.2% (Wunsch 2013)", S2)

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("**Which policy does best? (utility: +1 correct treatment, −1 unnecessary antibiotics, −5 missed sepsis)**")
        PRETTY = {"pomdp": "POMDP policy", "never_treat": "never treat",
                  "news2_rule": "treat when NEWS2 ≥ 5", "sirs_rule": "treat when SIRS ≥ 2"}
        e = ev.copy()
        e["policy"] = e["policy"].map(PRETTY).fillna(e["policy"])
        e = e.sort_values("mean utility")
        f = go.Figure(go.Bar(y=e["policy"], x=e["mean utility"], orientation="h",
                             marker=dict(color=[CRIT if p == "POMDP policy" else S1 for p in e["policy"]],
                                         line=dict(color=SURFACE, width=2)),
                             text=[f"{v:.3f}" for v in e["mean utility"]], textposition="outside", textfont=dict(color=INK2),
                             hovertemplate="%{y}<br>utility %{x:.3f}<extra></extra>"))
        st.plotly_chart(style(f, height=250, legend=False, xtitle="mean utility per patient (higher is better)"),
                        use_container_width=True, key="kpi_pol")
        source_note("From Step 5, on all 2,000 patients. Treating everyone or nobody are included as reference points.")

        st.markdown("**How that answer depends on the one weight with no published value**")
        f = go.Figure()
        for name, col, colr in [("POMDP", "POMDP utility", CRIT), ("NEWS2 rule", "NEWS2 rule", S1),
                                ("SIRS rule", "SIRS rule", S2), ("Never treat", "never treat", S3)]:
            f.add_trace(go.Scatter(x=sens["W_ABX"], y=sens[col], mode="lines+markers", name=name,
                                   line=dict(color=colr, width=2), marker=dict(size=8, line=dict(color=SURFACE, width=2)),
                                   hovertemplate=f"{name} · W_ABX %{{x}} · utility %{{y:.3f}}<extra></extra>"))
        st.plotly_chart(style(f, height=300, ytitle="mean utility", xtitle="cost of unnecessary antibiotics (W_ABX)"),
                        use_container_width=True, key="kpi_sens")
        source_note("When antibiotics are treated as cheap, acting early wins; when they are costly, the POMDP wins and doing nothing is hard to beat.")
    with right:
        st.markdown("**ICU cost per patient**")
        f = go.Figure()
        for lab, val, colr in [("no sepsis", 0, S1), ("sepsis", 1, CRIT)]:
            s = pts[pts["sepsis"] == val]
            if len(s):
                f.add_trace(go.Histogram(x=s["cost"], name=lab, nbinsx=40, opacity=.85,
                                         marker=dict(color=colr, line=dict(color=SURFACE, width=2)),
                                         hovertemplate=f"{lab} · $%{{x}} · %{{y}} patients<extra></extra>"))
        f.update_layout(barmode="overlay")
        st.plotly_chart(style(f, height=270, ytitle="patients", xtitle="ICU cost ($)"),
                        use_container_width=True, key="kpi_cost")
        source_note("Sepsis records stop about 3 hours after onset, so their stays look shorter.")

        st.markdown("**Satisfaction against lab waiting time**")
        w = pts.dropna(subset=["wait"]).copy()
        w["band"] = pd.cut(w["wait"], [0, 30, 40, 50, 60, 200], labels=["<30", "30–40", "40–50", "50–60", "60+"])
        s = w.groupby("band", observed=True).agg(rate=("sat", "mean"), n=("sat", "size")).reset_index()
        f = go.Figure(go.Bar(x=s["band"].astype(str), y=s["rate"] * 100, marker=dict(color=S3, line=dict(color=SURFACE, width=2)),
                             text=[f"{v * 100:.0f}%" for v in s["rate"]], textposition="outside", textfont=dict(color=INK2),
                             customdata=s["n"], hovertemplate="wait %{x} min<br>%{y:.1f}% top box<br>n = %{customdata}<extra></extra>"))
        f.add_hline(y=72, line=dict(color=MUTED, width=1.5, dash="dot"))
        f.add_annotation(x=len(s) - 1, y=76, xanchor="right", text="US average 72%", showarrow=False,
                         font=dict(color=MUTED, size=11))
        st.plotly_chart(pad_y(style(f, height=270, legend=False, ytitle="% giving the top score",
                                    xtitle="mean lab wait (minutes)"), [100]),
                        use_container_width=True, key="kpi_sat")
        source_note("Curve shape from Bleustein et al. (2014); level set to the HCAHPS national average.")
