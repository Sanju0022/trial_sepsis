# Multi-Agent Healthcare Decision Support — Streamlit dashboard

Group Project 3 (PDF Task 5). Seven tabs, each with its own local filters — there is no global sidebar filter.

## Run it

```bash
pip install streamlit plotly pandas numpy
streamlit run app.py
```

Open http://localhost:8501. In Google Colab, run the same command with a tunnel (for example `localtunnel` or `ngrok`).

## Folder

```
app.py                  the dashboard
.streamlit/config.toml  theme
data/                   the CSV files produced by Steps 3–7
```

## Tabs and where their data comes from

| Tab | Data file | Local filters |
|---|---|---|
| Patient Overview | sepsis_multiagent_dataset.csv (2,000 patients) | sepsis-only toggle, patient, hour range, which vital signs |
| POMDP Belief Tracker | pomdp_belief_traces.csv (200 patients), pomdp_policy.csv, pomdp_parameters.csv | patient group, patient, hours left for the threshold |
| Bandit Decision Explorer | bandit_treatment.csv, bandit_policy.csv, bandit_arm_shares.csv | patients seen, algorithms, bandit type, cost of unnecessary antibiotics |
| Bayesian Network | bayes_network.csv, bayes_fusion.csv | prior slider, the four SIRS signs, which measure to compare |
| MLE Parameter Insights | mle_parameters.csv | estimate group, parameters |
| Agent Collaboration Log | collaboration_log.csv (200 patients) | disagreement only, suggested action, patient, hour |
| Outcome & KPI Analysis | sepsis_multiagent_dataset.csv, pomdp_evaluation.csv, pomdp_sensitivity.csv | hospital set, age group, patient type |

## Notes on the data behind the charts

- Belief traces and the collaboration log cover the first 200 patients, so those two tabs say so on screen.
- The POMDP policy file holds three points in the stay (72, 24 and 1 hours left), so the threshold slider offers exactly those.
- Sepsis records in PhysioNet stop about 3 hours after onset, so sepsis stays look shorter and cheaper in the cost chart.
- Vital-sign panels are drawn one per sign rather than on one axis, because the scales differ.
- Where a treat threshold sits far above the plotted probabilities, it is written on the chart instead of being drawn, so the curve stays readable.
