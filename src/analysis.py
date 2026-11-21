"""Selector, controls, selective prediction and clustered statistics.
python src/analysis.py <per_case_results.csv> <outdir> [enhancer=dcp]"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
csv, out = sys.argv[1], Path(sys.argv[2]); ENH = sys.argv[3] if len(sys.argv) > 3 else "dcp"
out.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(2027)
d = pd.concat([pd.read_csv(f) for f in csv.split(",")], ignore_index=True)
if "rep" not in d: d["rep"] = 0
d["clip_i"] = d["clip"].astype(int)
role = lambda c: "dev_train" if c <= 170 else "dev_val" if c <= 227 else "final_test"
d["role"] = d.clip_i.map(role)
FEAT = ["conf_u", "ent_u", "area_u", "mconf_u", "disagree_u", "sdc_u"]

def pair(df, enh):
    k = ["rep", "image", "clip_i", "role", "family", "severity", "has_uterus_ref"]
    a = df[df.enhancer == "none"].set_index(k); b = df[df.enhancer == enh].set_index(k)
    return a.join(b, lsuffix="_r", rsuffix="_e").reset_index()
P = pair(d, ENH)

def feats(p, s):
    X = p[[f"{f}_{s}" for f in FEAT]].copy(); X.columns = FEAT
    X["conf_u_missing"] = X.conf_u.isna().astype(float)
    X["conf_u"] = X.conf_u.fillna(0); X["ent_u"] = X.ent_u.fillna(0.0)
    X["is_enh"] = 1.0 if s == "e" else 0.0
    return X

def ridge_fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0).replace(0, 1); Z = ((X - mu) / sd).values
    Zb = np.c_[np.ones(len(Z)), Z]; R = lam * np.eye(Zb.shape[1]); R[0, 0] = 0
    w = np.linalg.solve(Zb.T @ Zb + R, Zb.T @ y.values)
    f = lambda Xn: np.c_[np.ones(len(Xn)), ((Xn - mu) / sd).values] @ w
    f.params = dict(cols=list(X.columns), mu=mu.tolist(), sd=sd.tolist(), w=w.tolist())
    return f

def stack(p):
    r = feats(p, "r"); e = feats(p, "e")
    return pd.concat([r, e]).reset_index(drop=True), np.r_[p.dice_u_r.values, p.dice_u_e.values]

# selector: develop on original + family A only; uterus-positive images; simulator params never used
pos = P[P.has_uterus_ref == 1].copy()
dev = pos[(pos.role != "final_test") & (pos.family.isin(["original", "A"]))]
tr, va = dev[dev.role == "dev_train"], dev[dev.role == "dev_val"]
Xtr, ytr = stack(tr); best = None
for lam in [0.01, 0.1, 1, 10, 100, 1000]:
    f = ridge_fit(Xtr, pd.Series(ytr), lam); Xv, yv = stack(va); n = len(va)
    pr = f(Xv); sel = pr[n:] > pr[:n]; got = np.where(sel, va.dice_u_e, va.dice_u_r).mean()
    if best is None or got > best[0]: best = (got, lam)
LAM = best[1]; Xd, yd = stack(dev); model = ridge_fit(Xd, pd.Series(yd), LAM)

def score(p):
    X, _ = stack(p); s = model(X); n = len(p); return s[:n], s[n:]

def rule_choose(p, name, thr=None):
    if name == "always_raw": return np.zeros(len(p), bool)
    if name == "always_enh": return np.ones(len(p), bool)
    if name == "conf_only": return (p.conf_u_e.fillna(0).values > p.conf_u_r.fillna(0).values)   # foreground confidence
    if name == "conf_global": return (p.mconf_u_e.values > p.mconf_u_r.values)                  # whole-image mean confidence
    if name == "sdc": return (p.sdc_u_e.values > p.sdc_u_r.values)                               # soft-Dice-style expected-Dice confidence
    if name == "disagree": return p.disagree_u_e.values < thr
    if name == "selector": sr, se = score(p); return se > sr
    if name == "oracle": return p.dice_u_e.values > p.dice_u_r.values

def dice_of(p, ch): return np.where(ch, p.dice_u_e.values, p.dice_u_r.values)

thr = max(np.linspace(0.0, 0.6, 31), key=lambda t: dice_of(dev, rule_choose(dev, "disagree", t)).mean())
RULES = ["always_raw", "always_enh", "conf_only", "conf_global", "sdc", "disagree", "selector", "oracle"]
choose = lambda p, r: rule_choose(p, r, thr)
simple = ["always_raw", "always_enh", "conf_only", "conf_global", "sdc", "disagree"]
STRONG = max(simple, key=lambda r: dice_of(dev, choose(dev, r)).mean())

def clip_means(p, vals):
    t = pd.DataFrame({"image": p.image.values, "clip": p.clip_i.values, "v": vals})
    return t.groupby(["clip", "image"]).v.mean().groupby("clip").mean()

def boot(cm, B=4000):
    idx = rng.integers(0, len(cm), (B, len(cm))); s = cm.values[idx].mean(1)
    return float(cm.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)), s

def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); m = len(ps); run = 0
    for rank, i in enumerate(o): run = max(run, (m - rank) * ps[i]); adj[i] = min(1, run)
    return adj

final = pos[pos.role == "final_test"]
res = {"enhancer": ENH, "lambda": LAM, "disagree_thr": float(thr), "strongest_simple_control": STRONG}
REG = (("original", final[final.family == "original"]), ("A_in_regime", final[final.family == "A"]),
       ("B_heldout", final[final.family == "B"]))
rows = []
for reg, sub in REG:
    if len(sub) == 0: continue
    for r in RULES:
        m, lo, hi, _ = boot(clip_means(sub, dice_of(sub, choose(sub, r))))
        rows.append(dict(regime=reg, rule=r, dice=m, lo=lo, hi=hi, n_images=sub.image.nunique(), n_clips=sub.clip_i.nunique()))
T1 = pd.DataFrame(rows); T1.to_csv(out / "table_main_comparison.csv", index=False)

con = []
for reg, sub in REG[1:]:
    if len(sub) == 0: continue
    for ctrl in simple:
        diff = clip_means(sub, dice_of(sub, choose(sub, "selector")) - dice_of(sub, choose(sub, ctrl)))
        m, lo, hi, s = boot(diff); p = float(2 * min((s <= 0).mean(), (s >= 0).mean()))
        con.append(dict(regime=reg, contrast=f"selector - {ctrl}", primary=(reg == "B_heldout" and ctrl == STRONG), diff=m, lo=lo, hi=hi, p_boot=p))
C = pd.DataFrame(con)
if len(C): C["p_holm"] = holm(C.p_boot.values)
C.to_csv(out / "table_contrasts.csv", index=False)

eo = []
for (fam, sv), s in final.groupby(["family", "severity"]):
    dd = s.dice_u_e.values - s.dice_u_r.values; m, lo, hi, _ = boot(clip_means(s, dd))
    eo.append(dict(family=fam, severity=sv, dice_raw=s.dice_u_r.mean(), dice_enh=s.dice_u_e.mean(), delta=m, lo=lo, hi=hi,
                   large_degradation_rate=float((dd < -0.05).mean()), selector_dice=dice_of(s, choose(s, "selector")).mean()))
pd.DataFrame(eo).to_csv(out / "table_enhancement_outcomes.csv", index=False)

ins = []
for (fam, sv), s in d[d.role == "final_test"].groupby(["family", "severity"]):
    for e, g in s.groupby("enhancer"): ins.append(dict(family=fam, severity=sv, enhancer=e, dice_i=g.dice_i.mean(), n=len(g)))
pd.DataFrame(ins).to_csv(out / "table_instrument_secondary.csv", index=False)

def rc(p, conf, ch):
    dv = dice_of(p, ch); o = np.argsort(-conf); n = np.arange(1, len(o) + 1)
    return n / len(o), np.cumsum(dv[o]) / n

sel_rows = []; fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for k, (reg, sub) in enumerate(REG[1:]):
    if len(sub) == 0: continue
    ch_sel = choose(sub, "selector"); sr, se = score(sub); qsel = np.where(ch_sel, se, sr)
    ch_sdc = choose(sub, "sdc"); qsdc = np.where(ch_sdc, sub.sdc_u_e, sub.sdc_u_r)
    ch_conf = choose(sub, "conf_only"); qconf = np.where(ch_conf, sub.conf_u_e.fillna(0), sub.conf_u_r.fillna(0))
    dsr, dse = score(dev)
    for name, q, ch, dq in (("selector_quality", qsel, ch_sel, np.where(choose(dev, "selector"), dse, dsr)),
                            ("confidence_only", qconf, ch_conf, np.where(choose(dev, "conf_only"), dev.conf_u_e.fillna(0), dev.conf_u_r.fillna(0))),
                            ("soft_dice_confidence", qsdc, ch_sdc, np.where(choose(dev, "sdc"), dev.sdc_u_e, dev.sdc_u_r))):
        cov, curve = rc(sub, q, ch); ax[k].plot(cov, curve, label=name)
        tau = np.quantile(dq, 0.2); keep = q >= tau; o = np.argsort(-q); top = o[: int(round(0.8 * len(o)))]
        sel_rows.append(dict(regime=reg, score=name, dice_all=dice_of(sub, ch).mean(), dice_at_80pct_ranked=dice_of(sub, ch)[top].mean(),
                             aurc=float((1 - curve).mean()), dev_fixed_threshold_coverage=float(keep.mean()),
                             dice_at_dev_fixed_threshold=float(dice_of(sub, ch)[keep].mean()) if keep.any() else np.nan))
    ax[k].set_title(reg); ax[k].set_xlabel("coverage"); ax[k].set_ylabel("Dice (retained)"); ax[k].legend()
plt.tight_layout(); plt.savefig(out / "fig_risk_coverage.png", dpi=200); plt.close()
pd.DataFrame(sel_rows).to_csv(out / "table_selective.csv", index=False)

E = pd.DataFrame(eo); order = ["none", "low", "mid", "high"]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for fam, a in zip(["A", "B"], ax):
    s = final[final.family.isin(["original", fam])].copy(); s["sev"] = s.severity.where(s.family != "original", "none")
    g = s.groupby("sev")[["dice_u_r", "dice_u_e"]].mean().reindex(order)
    a.plot(order, g.dice_u_r, "o-", label="unenhanced"); a.plot(order, g.dice_u_e, "s-", label=ENH)
    s["c"] = dice_of(s, choose(s, "selector")); a.plot(order, s.groupby("sev").c.mean().reindex(order), "^--", label="selector")
    a.set_title(f"family {fam}"); a.set_xlabel("severity"); a.set_ylabel("uterus Dice"); a.legend()
plt.tight_layout(); plt.savefig(out / "fig_degradation_recovery.png", dpi=200); plt.close()
fig, a = plt.subplots(figsize=(5.5, 3.6))
if len(E): a.bar([f"{f}-{s}" for f, s in zip(E.family, E.severity)], E.delta, yerr=[E.delta - E.lo, E.hi - E.delta], capsize=2)
a.axhline(0, c="k", lw=.5); a.set_ylabel(f"Dice change ({ENH} - raw)"); plt.xticks(rotation=45); plt.tight_layout()
plt.savefig(out / "fig_paired_effect.png", dpi=200); plt.close()
res["n_dev_images"] = int(dev.image.nunique()); res["n_final_images_uterus_pos"] = int(final.image.nunique())
_sr, _se = score(dev); res["tau_80pct_coverage_dev"] = float(np.quantile(np.where(choose(dev, "selector"), _se, _sr), 0.2))
res["selector_picks_enh_rate"] = {r: float(choose(sub, "selector").mean()) for r, sub in REG if len(sub)}
res["quality_estimator"] = model.params
json.dump(res, open(out / "frozen_policy.json", "w"), indent=1)
print(json.dumps(res, indent=1)); print(T1.round(3).to_string()); print(C.round(3).to_string())

# ---- quantities used by the formal analysis in the paper (regret split, score rank agreement) ----
th = {}
for reg, sub in REG:
    if len(sub) == 0: continue
    dl = sub.dice_u_e.values - sub.dice_u_r.values; pk = choose(sub, "selector")
    cm = lambda v: float(clip_means(sub, v).mean())
    row = dict(HD=cm(np.maximum(dl, 0)), captured_help=cm(np.maximum(dl, 0) * pk), missed_help=cm(np.maximum(dl, 0) * ~pk),
               harmful_pick=cm(np.maximum(-dl, 0) * pk), avoided_harm=cm(np.maximum(-dl, 0) * ~pk), gain=cm(dl * pk),
               RD=cm(np.maximum(dl, 0) - dl * pk), pick_rate=float(pk.mean()))
    if reg != "original":
        from scipy.stats import spearmanr
        sr, se = score(sub); q = np.where(pk, se, sr)
        qc = np.where(choose(sub, "conf_only"), sub.conf_u_e.fillna(0), sub.conf_u_r.fillna(0)); qs = np.where(choose(sub, "sdc"), sub.sdc_u_e, sub.sdc_u_r)
        row["spearman_q_vs_fgconf"] = float(spearmanr(q, qc)[0]); row["spearman_q_vs_sdc"] = float(spearmanr(q, qs)[0])
        row["spearman_q_vs_dice"] = float(spearmanr(q, dice_of(sub, pk))[0]); row["spearman_fgconf_vs_dice"] = float(spearmanr(qc, dice_of(sub, choose(sub, "conf_only")))[0])
    th[reg] = row
json.dump(th, open(out / "theory_quantities.json", "w"), indent=1); print(json.dumps(th, indent=1))
