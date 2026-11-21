"""Experiment 3: descriptive outputs of the frozen pipeline on DeSmoke-LAP. No anatomical references exist,
so nothing here is accuracy. python src/desmoke_analysis.py <desmoke_results.csv> <frozen_policy.json> <outdir>"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
csv, pol, out = sys.argv[1], json.load(open(sys.argv[2])), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
d = pd.read_csv(csv); q = pol["quality_estimator"]; FEAT = ["conf_u", "ent_u", "area_u", "mconf_u", "disagree_u", "sdc_u"]
ent_fill = d.ent_u.max()
def quality(df, enh):
    X = df[FEAT].copy(); X["conf_u_missing"] = X.conf_u.isna().astype(float); X["conf_u"] = X.conf_u.fillna(0)
    X["ent_u"] = X.ent_u.fillna(0.0); X["is_enh"] = 1.0 if enh else 0.0
    Z = (X[q["cols"]].values - np.array(q["mu"])) / np.array(q["sd"]); return np.c_[np.ones(len(Z)), Z] @ np.array(q["w"])
rows = []
key = ["condition", "video", "frame"]; raw = d[d.enhancer == "none"].set_index(key)
for e in ["clahe", "dcp", "desmoke_lap"]:
    b = d[d.enhancer == e].set_index(key); j = raw.join(b, lsuffix="_r", rsuffix="_e", how="inner").reset_index()
    if j.empty: continue
    r = j[[c + "_r" for c in FEAT]].copy(); r.columns = FEAT; en = j[[c + "_e" for c in FEAT]].copy(); en.columns = FEAT
    qr, qe = quality(r, False), quality(en, True); pick = qe > qr; qbest = np.where(pick, qe, qr)
    j["enhancer"] = e; j["selector_picks_enhanced"] = pick; j["quality"] = qbest; j["abstain"] = qbest < pol["tau_80pct_coverage_dev"]
    j["uterus_pred_raw"] = j.area_u_r > 0.005; j["uterus_pred_enh"] = j.area_u_e > 0.005
    rows.append(j)
J = pd.concat(rows)
S = J.groupby(["condition", "enhancer"]).agg(n=("frame", "size"), conf_raw=("mconf_u_r", "mean"), conf_enh=("mconf_u_e", "mean"),
    area_raw=("area_u_r", "mean"), area_enh=("area_u_e", "mean"), disagreement=("disagree_u_e", "mean"),
    presence_raw=("uterus_pred_raw", "mean"), presence_enh=("uterus_pred_enh", "mean"),
    selector_picks_enh=("selector_picks_enhanced", "mean"), abstain_rate=("abstain", "mean")).round(4)
S.to_csv(out / "table_desmoke_descriptive.csv")
# paired-by-video summary (n = 10 procedures) for the learned desmoker on the 50-frame test clips
t = J[(J.condition == "Input") & (J.enhancer == "desmoke_lap")].groupby("video").agg(
    conf_raw=("mconf_u_r", "mean"), conf_enh=("mconf_u_e", "mean"), disagreement=("disagree_u_e", "mean"), abstain=("abstain", "mean"))
t.to_csv(out / "table_desmoke_lap_by_video.csv")
fig, ax = plt.subplots(figsize=(5.5, 3.6)); x = np.arange(len(t)); ax.bar(x - .2, t.conf_raw, .4, label="input"); ax.bar(x + .2, t.conf_enh, .4, label="DeSmoke-LAP output")
ax.set_xticks(x); ax.set_xticklabels(t.index, rotation=60); ax.set_ylabel("mean confidence (uterus branch)"); ax.legend(); plt.tight_layout()
plt.savefig(out / "fig_desmoke_confidence.png", dpi=200); plt.close()
print(S.to_string())
