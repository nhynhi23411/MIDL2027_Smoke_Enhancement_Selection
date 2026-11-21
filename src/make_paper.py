"""Builds paper/midl-fullpaper.tex and paper/refs.bib from the result tables (all numbers come from outputs/)."""
import json, pandas as pd, numpy as np
from pathlib import Path
O = Path("outputs"); P = Path("paper")
R = {e: dict(pol=json.load(open(O / d / "frozen_policy.json")), main=pd.read_csv(O / d / "table_main_comparison.csv"),
             con=pd.read_csv(O / d / "table_contrasts.csv"), enh=pd.read_csv(O / d / "table_enhancement_outcomes.csv"),
             sel=pd.read_csv(O / d / "table_selective.csv")) for e, d in (("dcp", "analysis"), ("clahe", "analysis_clahe"))}
ins = pd.read_csv(O / "analysis/table_instrument_secondary.csv"); DS = pd.read_csv(O / "analysis_desmoke/table_desmoke_descriptive.csv")
DV = pd.read_csv(O / "analysis_desmoke/table_desmoke_lap_by_video.csv"); aud = pd.read_csv(O / "image_mask_audit.csv")
f3 = lambda x: f"{x:.3f}"; ci = lambda m, lo, hi: f"{m:.3f} [{lo:.3f}, {hi:.3f}]"
def m(e, reg, rule):
    r = R[e]["main"]; r = r[(r.regime == reg) & (r.rule == rule)].iloc[0]; return r.dice, r.lo, r.hi
def c(e, reg, ctrl):
    r = R[e]["con"]; return r[(r.regime == reg) & (r.contrast == f"selector - {ctrl}")].iloc[0]
NM = {"always_raw": "Always unenhanced", "always_enh": "Always enhanced", "conf_only": "Confidence only (foreground)", "conf_global": "Confidence only (whole image)",
      "disagree": "Disagreement rule", "sdc": "Confidence only (soft-Dice-style)", "selector": "Learned selector (ours)", "oracle": "Oracle (upper bound)"}
def table_main(e):
    rows = []
    for rule in NM:
        rows.append(f"{NM[rule]} & " + " & ".join(ci(*m(e, reg, rule)) for reg in ("original", "A_in_regime", "B_heldout")) + r" \\")
    return "\n".join(rows)
def table_con(e):
    out = []
    for reg, tag in (("A_in_regime", "A"), ("B_heldout", "B")):
        for ctrl in ("always_raw", "conf_only", "sdc", "disagree"):
            r = c(e, reg, ctrl); star = r"$^{\dagger}$" if r.primary else ""
            out.append(f"{tag} & {NM[ctrl]}{star} & {ci(r['diff'], r.lo, r.hi)} & {r.p_holm:.3f} \\\\")
    return "\n".join(out)
def table_enh(e):
    t = R[e]["enh"]; o = []
    for _, r in t.iterrows():
        o.append(f"{r.family} & {r.severity} & {f3(r.dice_raw)} & {f3(r.dice_enh)} & {ci(r.delta, r.lo, r.hi)} & {100*r.large_degradation_rate:.0f}\\% \\\\")
    return "\n".join(o)
def table_sel(e):
    t = R[e]["sel"]; o = []
    for _, r in t.iterrows():
        nm = {"selector_quality": "selector quality", "confidence_only": "fg confidence", "soft_dice_confidence": "soft-Dice conf."}[r.score]
        o.append(f"{r.regime.split('_')[0]} & {nm} & {f3(r.dice_at_80pct_ranked)} & {f3(r.aurc)} & {100*r.dev_fixed_threshold_coverage:.0f}\\% & {f3(r.dice_at_dev_fixed_threshold)} \\\\")
    return "\n".join(o)
pv = ins.pivot_table(index=["family", "severity"], columns="enhancer", values="dice_i")
def table_ins():
    o = []
    for (fam, sv), r in pv.iterrows(): o.append(f"{fam if fam != 'original' else 'orig.'} & {sv} & {f3(r['none'])} & {f3(r['clahe'])} & {f3(r['dcp'])} \\\\")
    return "\n".join(o)
def table_ds():
    nm = {"Input": "Test clips (input)", "hazy": "Hazy frames", "clear": "Clear frames"}; en = {"clahe": "CLAHE", "dcp": "DCP", "desmoke_lap": "DeSmoke-LAP output"}
    o = []
    for _, r in DS.iterrows():
        o.append(f"{nm[r.condition]} & {en[r.enhancer]} & {int(r.n)} & {r.conf_raw:.3f} / {r.conf_enh:.3f} & {100*r.presence_raw:.0f}\\% / {100*r.presence_enh:.0f}\\% & {r.disagreement:.3f} & {100*r.selector_picks_enh:.0f}\\% & {100*r.abstain_rate:.0f}\\% \\\\")
    return "\n".join(o)
d_ = R["dcp"]; STRONG_D = R["dcp"]["pol"]["strongest_simple_control"]; STRONG_C = R["clahe"]["pol"]["strongest_simple_control"]
pB = c("dcp", "B_heldout", STRONG_D); rawB = c("dcp", "B_heldout", "always_raw"); rawBc = c("clahe", "B_heldout", "always_raw"); pBc = c("clahe", "B_heldout", STRONG_C)
rawA = c("dcp", "A_in_regime", "always_raw"); rawAc = c("clahe", "A_in_regime", "always_raw")
orig_raw, orig_dcp, orig_cl = m("dcp", "original", "always_raw")[0], m("dcp", "original", "always_enh")[0], m("clahe", "original", "always_enh")[0]
eo = R["dcp"]["enh"]; eoc = R["clahe"]["enh"]; ldr = lambda t: (t.large_degradation_rate.min() * 100, t.large_degradation_rate.max() * 100)
n_u = int((aud.uterus_px > 0).sum()); pick = d_["pol"]["selector_picks_enh_rate"]
pcs = pd.concat([pd.read_csv(O / f"per_case_v2_rep{k}.csv") for k in range(3)]); pcs["clip_i"] = pcs["clip"].astype(int)
pcs = pcs[(pcs.clip_i > 227) & (pcs.has_uterus_ref == 1)]
def repstat(fam, enh):
    v = []
    for k in range(3):
        t = pcs[(pcs.rep == k) & (pcs.family == fam) & (pcs.enhancer == enh)]
        v.append(t.groupby(["clip_i", "image"]).dice_u.mean().groupby("clip_i").mean().mean())
    return min(v), max(v)
rB0, rB1 = repstat("B", "none"); eB0, eB1 = repstat("B", "dcp"); rA0, rA1 = repstat("A", "none"); eA0, eA1 = repstat("A", "dcp")
_key = ["rep", "image", "family", "severity"]
def hh(enh):
    raw = pcs[pcs.enhancer == "none"].set_index(_key).dice_u; en = pcs[pcs.enhancer == enh].set_index(_key).dice_u
    g = (en - raw).dropna().reset_index(); g["reg"] = np.where(g.family == "original", "original", g.family)
    return {r: (100 * (t.dice_u > .05).mean(), 100 * (t.dice_u < -.05).mean()) for r, t in g.groupby("reg")}
HH = {"dcp": hh("dcp"), "clahe": hh("clahe")}
def table_hh():
    o = []
    for e, nm_ in (("dcp", "DCP"), ("clahe", "CLAHE")):
        for tag, reg in (("original", "original"), ("A", "A_in_regime"), ("B", "B_heldout")):
            raw_, orc_, sel_, enh_ = (m(e, reg, k)[0] for k in ("always_raw", "oracle", "selector", "always_enh"))
            h_, x_ = HH[e][tag]
            o.append(f"{nm_} & {tag} & {h_:.1f}\\% & {x_:.1f}\\% & {enh_ - raw_:+.3f} & {orc_ - raw_:.3f} & {sel_ - raw_:+.3f} & {orc_ - sel_:.3f} \\\\")
        if e == "dcp": o.append("\\midrule")
    return "\n".join(o)
def table_pol():
    o = []
    for e, nm_ in (("dcp", "DCP"), ("clahe", "CLAHE")):
        pl = R[e]["pol"]
        o.append(f"{nm_} & {pl['lambda']} & {pl['disagree_thr']:.2f} & {NM[pl['strongest_simple_control']]} & {pl['tau_80pct_coverage_dev']:.3f} & {pl['n_dev_images']} & {100*pl['selector_picks_enh_rate']['B_heldout']:.0f}\\% \\\\")
    return "\n".join(o)
HHS = dict(hbh=HH["dcp"]["B"][0], hbx=HH["dcp"]["B"][1], hoh=HH["dcp"]["original"][0], hox=HH["dcp"]["original"][1], hah=HH["dcp"]["A"][0], hax=HH["dcp"]["A"][1])
TH = json.load(open(O / "analysis/theory_quantities.json"))["B_heldout"]; DG = json.load(open(O / "dcp_gain.json"))
THS = dict(thHD=TH["HD"], thCap=TH["captured_help"], thMiss=TH["missed_help"], thHarm=TH["harmful_pick"], thAvoid=TH["avoided_harm"], thGain=TH["gain"], thRD=TH["RD"],
           rhoF=TH["spearman_q_vs_fgconf"], rhoS=TH["spearman_q_vs_sdc"], rhoQD=TH["spearman_q_vs_dice"], rhoCD=TH["spearman_fgconf_vs_dice"],
           dgM=DG["median_m"], dgMed=DG["median_gain"], dgMean=DG["mean_gain"], dgF15=100 * DG["frac_gain_gt_1p5"], dgF2=100 * DG["frac_gain_gt_2"], dgN=DG["n_frames"],
           thFrac=100 * TH["captured_help"] / TH["HD"], thAvoidNet=TH["avoided_harm"] - TH["missed_help"])
ds_l = DS[(DS.condition == "Input") & (DS.enhancer == "desmoke_lap")].iloc[0]
tex = r"""\documentclass{midl}
\jmlrvolume{-- Under Review}
\jmlryear{2027}
\jmlrworkshop{Full Paper -- MIDL 2027 submission}
\editors{Under Review for MIDL 2027}
\usepackage{booktabs}
\title[When does clearing the smoke hurt?]{When Does Clearing the Smoke Hurt? Enhancement, Selection and Abstention for Laparoscopic Segmentation under Controlled Smoke}
\midlauthor{\Name{Nhi-Yen Nguyen-Huynh$^{1}$} \Email{nhinhy23411@st.uel.edu.vn}\\
\addr $^{1}$ Vietnam National University, Ho Chi Minh City -- University of Economics and Law\\
\Name{Nguyen Quoc Khanh Le$^{2}$} \Email{khanhlee@tmu.edu.tw}\\
\addr $^{2}$ Taipei Medical University}
\begin{document}
\maketitle
\begin{abstract}
Smoke from energy devices is usually treated as a preprocessing problem: clear the frame, then segment it. The cleared frame can repair a prediction or overturn a correct one, and mean Dice cannot tell the two apart. We count both events with a paired design in which every enhanced prediction is compared with the unenhanced prediction of the same frozen segmenter on the same frame, using existing AutoLaparo masks, two geometry-preserving smoke simulators (one reserved for final testing), two enhancers (CLAHE and dark-channel-prior dehazing, DCP) and clip-clustered bootstrap intervals. Enhancement harmed far more frames than it helped, including unsmoked ones: DCP lowered uterus Dice from %(or).3f to %(od).3f on clean frames and harmed %(hbx).0f\%% of held-out-smoke frames against %(hbh).0f\%% helped. A learned selector between the two predictions improved on always-unenhanced (held-out smoke, DCP: %(rb)s) but not on a foreground-confidence rule (%(pb)s), and abstention matched confidence ranking. On DeSmoke-LAP, frozen-model confidence barely moves after learned desmoking while predicted masks change, so confidence cannot arbitrate. The results are negative and bounded: synthetic smoke, a frozen model previously exposed to AutoLaparo, no clinical claim. Protocols, manifests and analysis code are released.
\end{abstract}
\begin{keywords}
surgical smoke, image enhancement, segmentation, robustness, selective prediction
\end{keywords}
\section{Introduction}
A surgeon whose view is lost in smoke waits for the evacuator or wipes the lens; a vision pipeline does the software equivalent, clearing the frame and looking again. Desmoking and contrast enhancement are routinely proposed as a front end for recognition~\citep{pan2022desmoke,lu2026seeing}, and the usual argument is that clearer pixels give better predictions.

That argument treats enhancement as extra information. For a fixed segmenter it is also an intervention on the input: it moves colour and texture statistics away from those the model was trained on, and in practice it is applied to frames that contain no smoke at all. Mean Dice does not show how often the enhanced prediction displaces one that was correct, and for a surgeon the two events are not equivalent even when they cancel in a mean.

We study this with a paired design (Figure~\ref{fig:pipe}). Every enhanced prediction is compared with the unenhanced prediction of the same frozen model on the same frame, so each frame is labelled help, harm or unchanged. The comparison pulls three questions apart that accuracy mixes. Does the pair contain a better prediction at all (headroom)? Can a label-free quality score find it (selection)? Can the system decline when neither prediction is reliable (abstention)? Policies are fitted on development smoke and frozen before the held-out family is evaluated.

\begin{figure}[t]
\floatconts{fig:pipe}{\caption{Study overview. (a)~A clean frame with its existing AutoLaparo annotation is corrupted by a seeded smoke simulator (family~A for development, family~B held out), optionally enhanced (none, CLAHE or DCP), and segmented by the frozen joint model. A ridge selector, fitted on development data from label-free features only, picks the raw or enhanced prediction or abstains. (b)~Final-block results: uterus Dice per regime (DCP as enhanced branch), paired Dice change caused by DCP with 95\%% CIs, and Dice at 80\%% coverage on held-out family~B for the selector versus foreground confidence (fg conf.) and the soft-Dice-style confidence (SDC). Splits: clips 001--170 train, 171--227 validate, 228--300 test.}}
{\includegraphics[width=\linewidth]{figs/fig_pipeline.png}}
\end{figure}

The answer to the first question is mostly no. Enhancement harms about a quarter to a third of frames and helps fewer than one in ten, whether or not smoke is present, and the headroom left for a selector is small. A learned selector gains over always-unenhanced but cannot beat a foreground-confidence rule, and abstention behaves identically. We make four contributions. First, a paired protocol and decomposition that reports help, harm, headroom, selection regret and coverage separately, with clip-clustered inference and a held-out smoke family. Second, evidence on two enhancers and two simulators that enhancement does not preserve uterus segmentation on average. Third, a negative result for learned selection and abstention against confidence controls, including a soft-Dice-style confidence. Fourth, descriptive transfer outputs on DeSmoke-LAP with an explicit statement of what they cannot show, together with released protocols and analysis code. A targeted, abstract-level literature check found no study comparing enhanced and unenhanced segmentation branches with abstention under laparoscopic smoke; it was not a systematic review, so we make no strong novelty claim.

\section{Related work}
\textbf{Desmoking and downstream tasks.} Learned desmoking has been proposed for laparoscopy~\citep{pan2022desmoke}, and a recent transformer-based desmoker is evaluated on instrument segmentation and stereo depth, with benefits for the former and inconsistent effects for the latter~\citep{lu2026seeing}. We instead ask whether a fixed, previously trained segmenter benefits at all, including on frames without smoke. \textbf{Robustness and enhancement.} Common-corruption benchmarks~\citep{hendrycks2019corruptions} motivate testing a fixed model under controlled degradations. Enhancers optimised for recognition rather than appearance have been proposed~\citep{son2020urie}, and dehazing has been shown to degrade object detection on clear images, which motivates applying preprocessing selectively~\citep{kumar2025fog}; our segmentation results point in the same direction. 
\textbf{Selective prediction.} Risk--coverage evaluation goes back to~\citet{geifman2017selective}; for segmentation, image-level confidence designed around Dice (Soft Dice Confidence) is a strong estimator~\citep{borges2024sdc}. We include a soft-Dice-style confidence of this kind as an additional control, implemented as an expected-Dice approximation rather than the authors' exact estimator.

\section{Formal setup}
\label{sec:formal}
Let $x\in[0,1]^{H\times W\times 3}$ be a frame and $y\in\{0,1\}^{H\times W}$ its uterus mask. A corruption $\mathcal{S}$ and an enhancer $\mathcal{E}$ act on pixels only, so $y$ is unchanged and Dice is always measured against the original mask. All formulas below are those implemented in the released code.

\subsection{Corruption and enhancement operators}
\textbf{Family A (development).} With multi-octave fractal noise $N=\sum_{o=0}^{4}2^{-o}U_o/\sum_{o}2^{-o}$ rescaled to $[0,1]$ (each $U_o$ a bicubically upsampled uniform grid), coverage $\kappa$ and severity $s_A$,
\begin{equation}
\tilde x=(1-\alpha_A)\odot x+\alpha_A\odot a,\qquad \alpha_A(p)=s_A\,\mathrm{clip}\Big(\tfrac{N(p)-(1-\kappa)}{\kappa},0,1\Big)^{0.8},
\label{eq:smokeA}
\end{equation}
with airlight $a=(0.86,0.86,0.88)$, $\kappa\in\{0.45,0.60,0.75\}$ and $s_A\in\{0.25,0.50,0.75\}$.
\textbf{Family B (held out).} With $J\in\{1,2\}$ anisotropic Gaussian plumes warped by a fractal field $w$, severity $s_B\in\{0.35,0.60,0.85\}$ and tint $\tau\in\{\text{bluish},\text{brownish}\}$,
\begin{equation}
\alpha_B(p)=s_B\min\Big\{1,\sum_{j=1}^{J}\exp\!\big[-q_j(p)\,(0.6+1.2\,w(p))\big]\Big\},\quad
\tilde x=\mathrm{clip}\big[(1-\alpha_B)\,b+\alpha_B\,\tau\big],
\label{eq:smokeB}
\end{equation}
where $q_j$ is the squared normalised distance to plume centre $j$, $b=\mu+(1-\tfrac12\alpha_B)\odot(b_0-\mu)$ applies contrast loss around the spatial mean $\mu$, and $b_0=(1-\alpha_B)\odot x+\alpha_B\odot(G_{\sigma}\!*x)$ adds scatter blur with $\sigma=3+6s_B$.

\textbf{Enhancers.} CLAHE~\citep{zuiderveld1994clahe} applies contrast-limited histogram equalisation (clip limit $3$, $8{\times}8$ tiles) to the luminance of the LAB image. Dark-channel-prior dehazing~\citep{he2011dcp} estimates the airlight $A$ from the brightest $0.1\%%$ of the dark channel (clipped to $[0.3,0.98]$) and a transmission
\begin{equation}
\tilde t(p)=1-\omega\, m(p),\quad m(p)=\min_{q\in\Omega(p)}\min_{c}\frac{x^{c}(q)}{A^{c}},\qquad
J(p)=A+\frac{x(p)-A}{t(p)},
\label{eq:dcp}
\end{equation}
where $\Omega(p)$ is a $15{\times}15$ window, $\omega=0.95$, and $t$ is $\tilde t$ refined by a guided filter and clipped to $[t_0,1]$ with $t_0=0.1$.

\begin{proposition}[DCP rescales contrast whether or not smoke is present]\label{prop:dcp}
For every frame and pixel, $\|J(p)-A\|=g(p)\,\|x(p)-A\|$ with $g(p)=1/t(p)\ge 1$, and $g(p)>1$ wherever $t(p)<1$. For two pixels with equal $t$, $J(p)-J(q)=g\,(x(p)-x(q))$. If $m\equiv m_0$ is constant then $g=1/(1-\omega m_0)$ before filtering.
\end{proposition}
\begin{proof}
The first identity is the definition of $J$ in~\eqref{eq:dcp}. Since $x^{c}\ge 0$ and $A^{c}>0$ we have $m\ge 0$, so $\tilde t\le 1$; the clip to $[t_0,1]$ gives $t\le 1$ and hence $g\ge 1$. The difference identity follows by subtracting $J(q)=A+(x(q)-A)/t$ from $J(p)$. The case of constant $m_0$ is $t=1-\omega m_0$.
\end{proof}
\noindent The dark-channel prior assumes $m\approx 0$ on haze-free regions; where it fails, which includes bright specular and pink tissue in surgical frames, $g>1$ even when there is no smoke at all. On %(dgN)d unsmoked final-block frames (every sixth, unfiltered transmission) the median $m$ was %(dgM).2f (so $g\approx%(dgMed).2f$), the mean gain was %(dgMean).2f, and %(dgF15).0f\%% of pixels were amplified by more than $1.5$ and %(dgF2).0f\%% by more than $2$. A segmenter trained on unenhanced frames therefore receives a contrast-amplified input for a large share of pixels; this is consistent with, but does not prove, the Dice loss on clean frames in Section~\ref{sec:res}.

\subsection{Branches, label-free features and the selector}
Let $f$ be the frozen segmenter and $\pi_b(p)\in\Delta^{3}$ its anatomy softmax for branch $b\in\{0,e\}$ (unenhanced input $x$, or $\mathcal{E}(x)$). The uterus mask is $\hat y_b=\mathbb{1}[\arg\max_k\pi_{b,k}=1]$ with foreground $F_b=\{p:\hat y_b(p)=1\}$, confidence $c_b(p)=\max_k\pi_{b,k}(p)$ and entropy $h_b(p)=-\sum_k\pi_{b,k}(p)\log\pi_{b,k}(p)$. The label-free feature vector $\phi_b\in\mathbb{R}^{8}$ has entries
\begin{equation}
\bar c_b=\frac{1}{|F_b|}\sum_{p\in F_b}c_b(p),\quad
\bar h_b=\frac{1}{|F_b|}\sum_{p\in F_b}h_b(p),\quad
a_b=\frac{|F_b|}{HW},\quad
\bar c^{\mathrm{img}}_b=\frac{1}{HW}\sum_{p}c_b(p),
\label{eq:feat1}
\end{equation}
\begin{equation}
\delta_b=1-\mathrm{Dice}(\hat y_b,\hat y_0),\qquad
\mathrm{SDC}_b=\frac{2\sum_p\pi_{b,1}(p)\,\hat y_b(p)}{\sum_p\hat y_b(p)+\sum_p\pi_{b,1}(p)},
\label{eq:feat2}
\end{equation}
plus a missing-foreground flag ($\bar c_b,\bar h_b$ are set to $0$ when $F_b=\emptyset$) and a branch indicator. $\mathrm{SDC}_b$ is a first-order approximation of the expected Dice: if $Y_p\sim\mathrm{Bernoulli}(\pi_{b,1}(p))$ independently, $\mathbb{E}\sum_p\hat y_bY_p=\sum_p\hat y_b\pi_{b,1}$ and $\mathbb{E}\sum_pY_p=\sum_p\pi_{b,1}$, so replacing the ratio's numerator and denominator by their expectations gives~\eqref{eq:feat2}. Simulator parameters and reference masks never enter $\phi_b$.

Standardise $z_b=(\phi_b-\mu)\oslash\sigma$ with development statistics and fit a ridge regression of branch Dice on features over development frames $i$ and branches $b$,
\begin{equation}
\hat w=\arg\min_{w_0,w}\sum_{i,b}\big(D_{ib}-w_0-w^{\top}z_{ib}\big)^2+\lambda\|w\|^2=\big(\tilde Z^{\top}\tilde Z+\lambda R\big)^{-1}\tilde Z^{\top}d,
\label{eq:ridge}
\end{equation}
with $\tilde Z=[\mathbf 1,Z]$ and $R=\mathrm{diag}(0,1,\dots,1)$. The quality estimate is $\hat q_b(x)=\hat w_0+\hat w^{\top}z_b(x)$, the selector chooses $\sigma(x)=\mathbb{1}[\hat q_e(x)>\hat q_0(x)]$ and abstains when $\max_b\hat q_b<\tau$. Confidence-only controls have the same form with $\hat q_b$ replaced by $\bar c_b$ or $\mathrm{SDC}_b$.

\subsection{Help, harm, headroom and regret}
Let $D_b$ be the Dice of branch $b$, $\Delta=D_e-D_0$, $\Delta^{+}=\max(\Delta,0)$ and $\Delta^{-}=\max(-\Delta,0)$. With a margin of $0.05$ fixed before final evaluation, a frame is \emph{helped} if $\Delta>0.05$ and \emph{harmed} if $\Delta<-0.05$. For a policy $\sigma\in\{0,1\}$ the realised Dice is $D_\sigma=D_0+\sigma\Delta$ and the oracle is $D^{*}=\max(D_0,D_e)=D_0+\Delta^{+}$, which cannot be deployed. \emph{Headroom} is $\mathrm{HD}=\mathbb{E}[\Delta^{+}]$, \emph{regret} is $\mathrm{RD}(\sigma)=\mathbb{E}[D^{*}-D_\sigma]$, and \emph{coverage} is the share of frames on which the policy does not abstain.

\begin{lemma}[Gain bookkeeping]\label{lem:gain}
For any policy $\sigma$, $\ \mathbb{E}[D_\sigma-D_0]=\mathrm{HD}-\mathrm{RD}(\sigma)$.
\end{lemma}
\begin{proof}
$D_\sigma-D_0=(D^{*}-D_0)-(D^{*}-D_\sigma)$; take expectations.
\end{proof}

\begin{proposition}[Where a selector's value and loss come from]\label{prop:regret}
For any policy $\sigma$,
\begin{align}
\mathrm{RD}(\sigma)&=\underbrace{\mathbb{E}[\Delta^{+}(1-\sigma)]}_{\text{missed help}}+\underbrace{\mathbb{E}[\Delta^{-}\sigma]}_{\text{harmful picks}},\label{eq:rd}\\
\mathbb{E}[D_\sigma-D_0]&=\underbrace{\mathbb{E}[\Delta^{+}\sigma]}_{\text{captured help}}-\underbrace{\mathbb{E}[\Delta^{-}\sigma]}_{\text{harmful picks}},\label{eq:gain0}\\
\mathbb{E}[D_\sigma-D_e]&=\underbrace{\mathbb{E}[\Delta^{-}(1-\sigma)]}_{\text{avoided harm}}-\underbrace{\mathbb{E}[\Delta^{+}(1-\sigma)]}_{\text{missed help}}.\label{eq:gaine}
\end{align}
\end{proposition}
\begin{proof}
Write $\Delta=\Delta^{+}-\Delta^{-}$. Then $D^{*}-D_\sigma=\Delta^{+}-\sigma\Delta^{+}+\sigma\Delta^{-}=\Delta^{+}(1-\sigma)+\Delta^{-}\sigma$, giving~\eqref{eq:rd}; $D_\sigma-D_0=\sigma\Delta^{+}-\sigma\Delta^{-}$ gives~\eqref{eq:gain0}; and $D_\sigma-D_e=(\sigma-1)\Delta=(1-\sigma)\Delta^{-}-(1-\sigma)\Delta^{+}$ gives~\eqref{eq:gaine}.
\end{proof}
\noindent On held-out family~B with DCP (\tableref{tab:hh}), $\mathrm{HD}=%(thHD).4f$, of which the selector captures %(thCap).4f (%(thFrac).0f\%%) while missing %(thMiss).4f and taking harmful picks worth %(thHarm).4f; its gain over unenhanced is $%(thCap).4f-%(thHarm).4f=%(thGain).4f$. Against always-enhanced its gain is $%(thAvoid).4f-%(thMiss).4f=%(thAvoidNet).4f$: almost all of a selector's advantage over always-enhancing is avoided harm, and the part left to learn, the headroom, is under two hundredths of Dice.

\subsection{Abstention and rank invariance}
For a score $s_i$ and Dice $D_i$ on $n$ frames, sort by $s$ in decreasing order and let the \emph{risk at coverage} $c$ be $R_s(c)=1-\frac{1}{\lceil cn\rceil}\sum_{r\le\lceil cn\rceil}D_{(r)}$, with $\mathrm{AURC}(s)=\int_0^1R_s(c)\,dc$.
\begin{proposition}[Rank invariance]\label{prop:rank}
$R_s$ and $\mathrm{AURC}(s)$ depend on $s$ only through the ordering it induces: for any strictly increasing $g$, $R_{g\circ s}=R_s$. Among all scores, ranking by $D$ itself minimises $R_s(c)$ at every $c$.
\end{proposition}
\begin{proof}
Sorting is invariant to strictly increasing maps, so the retained sets are identical. For fixed $c$ the mean of $\lceil cn\rceil$ values of $D$ is maximised by taking the largest ones, which is the ranking by $D$.
\end{proof}
\noindent Two scores with the same ordering therefore give the same curve. On held-out family~B the selector's quality score has Spearman correlation %(rhoF).2f with foreground confidence and %(rhoS).2f with $\mathrm{SDC}$, and its correlation with the realised Dice (%(rhoQD).2f) is slightly below that of foreground confidence (%(rhoCD).2f). The risk--coverage curves in Figure~\ref{fig:rc} overlap because the three scores are nearly rank-equivalent, so no abstention threshold on $\hat q$ can do better than the confidence it re-weights.

\subsection{Clustered inference}
Index frames by clip $k=1,\dots,K$, frame $i$ and smoke repetition $r\in\{1,\dots,R\}$, and let $\bar d_k$ be the mean of a paired difference over frames and repetitions in clip $k$. The estimate is $\hat\theta=\frac{1}{K}\sum_k\bar d_k$ with $K=%(nfc)d$ clips; confidence intervals resample clips with replacement, $\hat\theta^{*}=\frac{1}{K}\sum_k\bar d_{k^{*}}$, 4000 times, and report the 2.5th and 97.5th percentiles. Planned contrasts are adjusted by Holm's procedure.
\begin{proposition}[Repetitions do not shrink the clip-level interval]\label{prop:var}
Under a random-effects model $d_{kir}=\theta+u_k+v_{ki}+e_{kir}$ with independent zero-mean terms of variances $\sigma_u^2,\sigma_v^2,\sigma_e^2$ and $n$ frames per clip, $\mathrm{Var}(\hat\theta)=\dfrac{1}{K}\Big(\sigma_u^2+\dfrac{\sigma_v^2}{n}+\dfrac{\sigma_e^2}{nR}\Big)$.
\end{proposition}
\begin{proof}
$\bar d_k=\theta+u_k+\frac1n\sum_iv_{ki}+\frac{1}{nR}\sum_{i,r}e_{kir}$ has the stated variance, and the $K$ clip means are independent.
\end{proof}
\noindent Smoke repetitions only enter the last term, so averaging three of them cannot reduce the interval below the clip-level terms: the effective sample size is the number of clips, not $n\!\cdot\!K\!\cdot\!R$ frame variants.

\section{Data, exposure and protocol}
\textbf{AutoLaparo Task 3} provides %(nimg)d frames from %(nclip)d clips (six frames per clip) with uterus and instrument masks; %(nu)d frames have a uterus label. A missing uterus label is not treated as evidence of absence, so the uterus endpoint uses only the %(nu)d positive frames and instruments are reported as a secondary task on all frames. Instrument parts are merged into one foreground class. Image--mask alignment, label ranges and duplicates were checked automatically (no missing masks, no size mismatches, no duplicate hashes).

\textbf{Prior exposure.} The recognition model is our previously developed joint model (DINOv2-Base, instrument and anatomy heads, frozen checkpoint with recorded SHA-256, no retraining here). All AutoLaparo frames were used as an external test set in that earlier work, and an earlier abandoned fine-tuning experiment on AutoLaparo has unresolved membership. \emph{These data are therefore not independent external validation}; this study measures robustness and selection, not generalisation.

\textbf{Splits.} Procedure identifiers are unavailable. Clip index is used as a proxy; clips 001--170, 171--227 and 228--300 form selector-development, development-validation and final-test blocks, following the contiguous official partition, which likely separates source videos (an assumption that we could not verify). All variants of a frame stay in one block. The final block contains %(nfin)d uterus-positive frames from %(nfc)d clips. Smoke realisations are not counted as independent samples.

\section{Study design}

\textbf{Branches.} Each input yields an unenhanced and an enhanced prediction from the same frozen model, evaluated at $960{\times}540$ after downscaling; the released DeSmoke-LAP weights were not available, so a learned desmoker is used only on DeSmoke-LAP's released output clips (Section~\ref{sec:ds}). The selector is fitted on original and family-A frames of the development blocks only; $\lambda$ is chosen on the development-validation block, then the model is refit on all development data and frozen.

\textbf{Controls.} Always unenhanced; always enhanced; foreground confidence $\bar c_b$; whole-image confidence $\bar c^{\mathrm{img}}_b$; $\mathrm{SDC}_b$; a disagreement rule (enhance only if $\delta_e$ is below a threshold tuned on development data); and the oracle as an upper bound. The \emph{strongest simple control} is chosen on development data (here: %(strong)s for DCP). The primary contrast is selector minus the strongest simple control, without abstention, on held-out family~B. Secondary measures are Dice at 80\%% coverage, AURC, and development-fixed thresholds with their achieved coverage; the harm margin of 0.05 was fixed before final evaluation.

\section{Results}
\label{sec:res}
\subsection{Enhancement lowers uterus Dice, with or without smoke}
Smoke reduces uterus Dice modestly (clip-averaged unenhanced Dice %(or).3f on original frames, %(ra).3f for family~A, %(rb_).3f for family~B; \figureref{fig:deg}). Enhancement did not recover it on average (\tableref{tab:enh}). DCP lowered Dice in every condition, including on original frames (%(od).3f vs %(or).3f), with %(ld0).0f--%(ld1).0f\%% of frames dropping by more than 0.05. Counted per frame (\tableref{tab:hh}), DCP harmed %(hox).0f\%% of clean frames, %(hax).0f\%% under family~A and %(hbx).0f\%% under held-out family~B, and helped only %(hoh).0f\%%, %(hah).0f\%% and %(hbh).0f\%%. CLAHE was gentler (original %(oc).3f) with intervals that include zero for most conditions, but still harmful on average under family~B. Instrument Dice (secondary, \tableref{tab:ins} in the appendix) changes by at most 0.02 and is improved by DCP only under family~A (high and mid severity).


\begin{table}[!htbp]
\floatconts{tab:hh}{\caption{Help, harm, headroom and regret on the final block (margin 0.05). Help/Harm: share of paired frames with $G_e>0.05$ / $<-0.05$ (pooled over repetitions). $\Delta$ is the mean change of always-enhanced; HD $=\mathbb{E}[D^{*}-D_0]$; Sel. is the selector's gain over unenhanced; RD $=\mathbb{E}[D^{*}-D_\pi]$ for the selector. By the lemma, Sel.\ $=$ HD $-$ RD.}}
{\scriptsize\begin{tabular}{llccccccc}\toprule Enh. & Regime & Help & Harm & $\Delta$ & HD & Sel. & RD\\\midrule
%(thh)s
\bottomrule\end{tabular}}
\end{table}
\subsection{A learned selector does not beat a confidence rule}
\tableref{tab:main} gives all-case performance. Against always-unenhanced, the selector gained %(gA).3f for family~A (DCP) and %(gB)s on held-out family~B (DCP; Holm-adjusted $p=%(pgB).3f$); with CLAHE the held-out gain was %(gBc)s and not significant after correction. Against the strongest simple control, which is the foreground-confidence rule, the primary contrast was %(pb)s ($p_{\mathrm{Holm}}=%(pph).3f$; CLAHE: %(pbc)s). \emph{The selector therefore shows no measurable benefit beyond confidence-only selection}, whether the confidence is the foreground mean or the soft-Dice-style expected-Dice score (point differences below 0.002 Dice, intervals within $\pm0.006$). The oracle leaves little headroom over always-unenhanced for DCP (%(orc).3f vs %(rwb).3f), which bounds any selector. The strongest simple control was %(sD)s for DCP and %(sC)s for CLAHE. The selector chose the enhanced branch for %(pk).0f\%% of held-out family-B frames. Whole-image mean confidence is a poor selection signal (\tableref{tab:main}), because background pixels dominate it; the foreground variant is the fair control.

\begin{table}[!htbp]
\floatconts{tab:main}{\caption{Uterus Dice (clip-averaged mean [95\%% CI]) on the final block, DCP as enhanced branch. Original = unsmoked frames; A = development smoke family; B = held-out family.}}
{\scriptsize\begin{tabular}{lccc}\toprule Policy & Original & Family A & Family B (held out)\\\midrule
%(tmain)s
\bottomrule\end{tabular}}
\end{table}
\subsection{Abstention adds nothing over confidence ranking}
Risk--coverage curves of the selector and the confidence rule overlap (\figureref{fig:rc}, \tableref{tab:sel}); neither Dice at 80\%% coverage nor AURC separates them. With development-fixed thresholds the confidence rule retained 100\%% of frames because about 40\%% of unenhanced predictions on label-positive frames contain no uterus pixels and receive zero foreground confidence, so the 20th-percentile threshold is degenerate rather than evidence of good calibration. The 80\%% coverage setting is a research choice, not a safety standard.

\begin{figure}[!htbp]
\floatconts{fig:rc}{\caption{Risk--coverage curves of the selector quality score and foreground confidence (DCP). Curves overlap.}}
{\includegraphics[width=\linewidth]{figs/fig_risk_coverage.png}}
\end{figure}

\subsection{Variability across realisations and cost}
\textbf{Smoke realisations.} Each variant was generated three times with different seeds; results average the realisations within a frame and never count them as independent. Across realisations, held-out family-B unenhanced Dice ranged %(rB0).3f--%(rB1).3f and DCP-enhanced %(eB0).3f--%(eB1).3f (family A: %(rA0).3f--%(rA1).3f and %(eA0).3f--%(eA1).3f), so simulator randomness is small relative to the enhancement effect.

\textbf{Compute.} On one A10 GPU (batch 1, $1920{\times}1080$ JPEG input) the two-branch pipeline took about 130\,ms per frame (median: decoding and resizing 27\,ms, unenhanced branch 10\,ms, CPU DCP 82\,ms, enhanced branch 11\,ms); selector scoring was not timed separately. This is a research measurement, not evidence for real-time surgical use.

\subsection{Real smoke: DeSmoke-LAP outputs are descriptive only}\label{sec:ds} DeSmoke-LAP has no organ masks, so no accuracy is reported. On its 500 released test-clip frames, the learned desmoker changed mean uterus-branch confidence from %(dc0).3f to %(dc1).3f and predicted-uterus presence from %(dp0).0f\%% to %(dp1).0f\%%, while the mean mask disagreement between input and output was %(dd).3f; variation across the ten procedures was large (disagreement %(dmin).3f--%(dmax).3f). Confidence is thus nearly unchanged while masks differ, so confidence alone cannot say whether the changes are corrections or errors. The AutoLaparo-fitted policy picked the enhanced branch for %(dpick).0f\%% of these frames and abstained for %(dab).0f\%%, showing that its branch choice and threshold do not transfer across sources; these are not accuracy statements and no claim of confidently wrong anatomy is made.


\section{Discussion and limitations}
What clearing the smoke is worth belongs to the model and its training distribution, not to enhancement as such. With this frozen segmenter both enhancers cost Dice on average, even where there is no smoke to remove, so the enhancer changes the input in a way the model does not absorb. Three practical lessons follow. Report help and harm separately: a small mean loss hides that roughly a third of held-out-smoke frames get worse. Compare any learned selector with always-unenhanced \emph{and} with a confidence rule before claiming learned selection, because here the gap to the confidence control is about 0.00 Dice. And do not treat confidence as a check on enhancement: on DeSmoke-LAP confidence stays near its ceiling while masks differ substantially.

\textbf{Limitations.} Results concern two simulators, three realisations each, and downscaled frames; real smoke differs. The frozen model reaches only moderate uterus Dice on these frames (%(or).3f), was previously exposed to AutoLaparo, and enhancement operates on frames the model was never adapted to; a model trained on enhanced data could behave differently. Clip index is a proxy for procedure and independence of the blocks is assumed. The two enhancers are simple; the strongest learned desmoker was not run on AutoLaparo. The uterus endpoint excludes %(nneg)d frames without labels. The analysis was fixed after a first pass in which the whole-image confidence control was found to be a weak baseline; we replaced it by the foreground control and report both; the soft-Dice-style control and the three-fold repetition of smoke realisations were added afterwards in response to the literature check. These post hoc additions did not change the primary conclusion, but the primary-contrast control is defined by the development-data rule stated above. Nothing here establishes accuracy under real smoke, recognition of hidden anatomy, clinical benefit, or real-time suitability (timing above is a research measurement on one GPU). Following the pre-specified decision point, we narrow the claim: under these conditions we found no evidence that a learned selector improves on a confidence rule, and we do not present this as a method contribution.

\section{Conclusion}
Enhancement before segmentation is not safe by default: it lowered uterus Dice on average even on unsmoked frames. Choosing between branches helps relative to always-unenhanced, but a foreground-confidence rule matches the learned selector, and abstention adds nothing measurable. Future work should evaluate on independent annotated cohorts with verified procedure identifiers, add real-smoke references, and test enhancers trained jointly with the segmenter.

\section*{Ethics Statement}
This study uses two released research datasets (AutoLaparo and DeSmoke-LAP) under their non-commercial licences, creates no new patient data, and offers no clinical recommendation. Smoke is simulated, so no conclusion about real surgical smoke or patient outcomes should be drawn.

\section*{Data and Code Availability}
Code, protocols, manifests with content hashes, per-case results and analysis outputs: \url{https://github.com/nhynhi23411/MIDL2027_Smoke_Enhancement_Selection}. AutoLaparo and DeSmoke-LAP images and the frozen checkpoint are not redistributed; image identifiers, hashes and seeds allow every perturbed frame to be rebuilt from the original sources.

\section*{Use of AI tools}
An AI assistant (Claude, Anthropic) was used for writing and running analysis scripts, language edits, consistency checks of the text and reference verification, under the authors' direction. The authors designed the study, checked all results, and take full responsibility for the paper.

\midlacknowledgments{This research was funded by the University of Economics and Law, Ho Chi Minh City, Vietnam, and Vietnam National University, Ho Chi Minh City, Vietnam.}
\bibliography{refs}
\appendix
\section{Supplementary results}
\label{app:supp}
Figure~\ref{fig:deg} and Table~\ref{tab:enh} give the effect of DCP by severity, and Table~\ref{tab:ds} the DeSmoke-LAP descriptive outputs. Tables~\ref{tab:con} and~\ref{tab:sel} give all planned contrasts and the selective-prediction summary for DCP. Tables~\ref{tab:mainc} and~\ref{tab:enhc} repeat the main comparison and the enhancement effect with CLAHE as the enhanced branch; Table~\ref{tab:ins} gives the secondary instrument endpoint.
\begin{table}[htbp]
\floatconts{tab:mainc}{\caption{As Table~\ref{tab:main} with CLAHE as the enhanced branch.}}
{\scriptsize\begin{tabular}{lccc}\toprule Policy & Original & Family A & Family B (held out)\\\midrule
%(tmainc)s
\bottomrule\end{tabular}}
\end{table}
\begin{table}[htbp]
\floatconts{tab:enhc}{\caption{As Table~\ref{tab:enh} with CLAHE.}}
{\scriptsize\begin{tabular}{llcccc}\toprule Family & Sev. & Unenh. & Enh. & $\Delta$ & Drop$>$0.05\\\midrule
%(tenhc)s
\bottomrule\end{tabular}}
\end{table}
\begin{table}[htbp]
\floatconts{tab:ins}{\caption{Instrument union Dice (secondary, all final-block frames) by enhancer.}}
{\scriptsize\begin{tabular}{llccc}\toprule Family & Sev. & Unenh. & CLAHE & DCP\\\midrule
%(tins)s
\bottomrule\end{tabular}}
\end{table}
\begin{table}[htbp]
\floatconts{tab:con}{\caption{Selector minus control, Dice difference [95\%% CI] and Holm-adjusted $p$ (all planned contrasts per enhancer). $^\dagger$ primary contrast. Top: DCP; bottom: CLAHE.}}
{\scriptsize\begin{tabular}{llcc}\toprule Family & Control & Difference & $p_{\mathrm{Holm}}$\\\midrule
%(tcon)s
\midrule
%(tconc)s
\bottomrule\end{tabular}}
\end{table}
\begin{table}[htbp]
\floatconts{tab:sel}{\caption{Selective prediction on the final block (DCP). Dice at 80\%% coverage and AURC use ranking by each frozen score; the last two columns use the threshold fixed on development data.}}
{\scriptsize\begin{tabular}{llccccc}\toprule Fam. & Score & Dice@80\%% & AURC & Cov. & Dice (kept)\\\midrule
%(tsel)s
\bottomrule\end{tabular}}
\end{table}
\begin{figure}[htbp]
\floatconts{fig:deg}{\caption{Uterus Dice versus severity (final block): unenhanced, DCP-enhanced and selector output.}}
{\includegraphics[width=\linewidth]{figs/fig_degradation_recovery.png}}
\end{figure}
\begin{table}[htbp]
\floatconts{tab:enh}{\caption{Enhancement effect on uterus segmentation, final block (DCP). Dice pooled over frames; $\Delta$ is the clip-averaged paired change [95\%% CI]; last column is the share of frames with a drop above 0.05. CLAHE: see Appendix~\ref{app:supp}.}}
{\scriptsize\begin{tabular}{llcccc}\toprule Family & Sev. & Unenh. & Enh. & $\Delta$ & Drop$>$0.05\\\midrule
%(tenh)s
\bottomrule\end{tabular}}
\end{table}
\begin{table}[htbp]
\floatconts{tab:ds}{\caption{DeSmoke-LAP descriptive outputs of the frozen pipeline (no reference masks; not accuracy). Pairs show unenhanced / enhanced. Selector and abstention use the AutoLaparo-fitted policy with a development-fixed threshold.}}
{\tiny\begin{tabular}{llrcccrr}\toprule Source & Enhancer & $n$ & Mean conf. & Uterus present & Disagr. & Picks enh. & Abstain\\\midrule
%(tds)s
\bottomrule\end{tabular}}
\end{table}
\section{Policy definitions}
\begin{table}[htbp]
\floatconts{tab:pol}{\caption{Frozen policies. $\lambda$: ridge penalty chosen on the development-validation block; thr.: disagreement threshold of the disagreement rule; strongest control: chosen on development data; $\tau_{80}$: abstention threshold giving 80\%% coverage on development data; $n_{\mathrm{dev}}$: label-positive development images; picks enh.: share of held-out family-B frames for which the selector picks the enhanced branch.}}
{\scriptsize\begin{tabular}{lcclccc}\toprule Enh. & $\lambda$ & thr. & Strongest control & $\tau_{80}$ & $n_{\mathrm{dev}}$ & Picks enh.\\\midrule
%(tpol)s
\bottomrule\end{tabular}}
\end{table}
\textbf{Simulators.} Family~A: multi-octave fractal haze (five octaves), grey-white airlight $(0.86,0.86,0.88)$, coverage $0.45/0.60/0.75$ and maximum opacity $0.25/0.50/0.75$ for low/mid/high. Family~B: one or two Gaussian plumes warped by fractal noise, bluish or brownish tint, scatter blur with $\sigma=3+6\alpha_{\max}$ and contrast loss $1-0.5\alpha$, maximum opacity $0.35/0.60/0.85$. \textbf{Features.} Foreground confidence, foreground entropy, predicted area, whole-image confidence, branch disagreement, soft-Dice-style expected-Dice confidence, a missing-foreground flag and a branch indicator; features are standardised on development data. \textbf{Enhancers.} CLAHE clip limit 3.0 on an $8{\times}8$ grid applied to LAB luminance; DCP with a $15{\times}15$ patch, $\omega=0.95$, transmission floor $0.1$ and a guided filter of radius 41.

\section{Reproducibility details}
Frozen checkpoint SHA-256 \texttt{c8c76331\ldots}; code, simulator configuration, split and exposure tables are released at \url{https://github.com/nhynhi23411/MIDL2027_Smoke_Enhancement_Selection}. Smoke seeds are derived deterministically from image name, family, severity and repetition index (three repetitions). Both enhancers and the selector are deterministic given the seeds.
\end{document}
"""
CITES = [(r"\textbf{AutoLaparo Task 3} provides", r"\textbf{AutoLaparo Task 3}~\citep{wang2022autolaparo} provides"),
         ("(DINOv2-Base,", r"(DINOv2-Base~\citep{oquab2024dinov2},"), ("DeSmoke-LAP has no organ masks", r"DeSmoke-LAP~\citep{pan2022desmoke} has no organ masks")]
for a, b in CITES:
    assert a in tex, a; tex = tex.replace(a, b)
S = dict(od=orig_dcp, **{"or": orig_raw}, oc=orig_cl, rb=ci(rawB["diff"], rawB.lo, rawB.hi), pb=ci(pB["diff"], pB.lo, pB.hi), nimg=len(aud), nclip=aud["clip"].nunique(), nu=n_u,
         nneg=len(aud) - n_u, nfin=int(R["dcp"]["main"].n_images.iloc[0]), nfc=int(R["dcp"]["main"].n_clips.iloc[0]), strong=NM[d_["pol"]["strongest_simple_control"]],
         ra=m("dcp", "A_in_regime", "always_raw")[0], rb_=m("dcp", "B_heldout", "always_raw")[0], ld0=ldr(eo)[0], ld1=ldr(eo)[1],
         gA=rawA["diff"], gB=ci(rawB["diff"], rawB.lo, rawB.hi), pgB=rawB.p_holm, gBc=ci(rawBc["diff"], rawBc.lo, rawBc.hi), pph=pB.p_holm,
         pbc=ci(pBc["diff"], pBc.lo, pBc.hi), orc=m("dcp", "B_heldout", "oracle")[0], rwb=m("dcp", "B_heldout", "always_raw")[0], pk=100 * pick["B_heldout"],
         dc0=ds_l.conf_raw, dc1=ds_l.conf_enh, dp0=100 * ds_l.presence_raw, dp1=100 * ds_l.presence_enh, dd=ds_l.disagreement, dmin=DV.disagreement.min(), dmax=DV.disagreement.max(),
         sD=NM[STRONG_D], sC=NM[STRONG_C], rB0=rB0, rB1=rB1, eB0=eB0, eB1=eB1, rA0=rA0, rA1=rA1, eA0=eA0, eA1=eA1, dpick=100 * ds_l.selector_picks_enh, dab=100 * ds_l.abstain_rate,
         tmain=table_main("dcp"), tmainc=table_main("clahe"), tcon=table_con("dcp"), tconc=table_con("clahe"), tenh=table_enh("dcp"), tenhc=table_enh("clahe"),
         tsel=table_sel("dcp"), tins=table_ins(), tds=table_ds(), thh=table_hh(), tpol=table_pol(), **HHS, **THS)
(P / "midl-fullpaper.tex").write_text(tex % S, encoding="utf-8")
(P / "refs.bib").write_text(r"""@inproceedings{wang2022autolaparo, title={{AutoLaparo}: A New Dataset of Integrated Multi-tasks for Image-guided Surgical Automation in Laparoscopic Hysterectomy}, author={Wang, Ziyi and Lu, Bo and Long, Yonghao and Zhong, Fangxun and Cheung, Tak-Hong and Dou, Qi and Liu, Yunhui}, booktitle={Medical Image Computing and Computer Assisted Intervention (MICCAI)}, series={LNCS}, volume={13437}, pages={486--496}, year={2022}, publisher={Springer}}
@article{pan2022desmoke, title={{DeSmoke-LAP}: improved unpaired image-to-image translation for desmoking in laparoscopic surgery}, author={Pan, Yirou and Bano, Sophia and Vasconcelos, Francisco and Park, Hyun and Jeong, Taikyeong Ted and Stoyanov, Danail}, journal={International Journal of Computer Assisted Radiology and Surgery}, volume={17}, number={5}, pages={885--893}, year={2022}}
@article{he2011dcp, title={Single image haze removal using dark channel prior}, author={He, Kaiming and Sun, Jian and Tang, Xiaoou}, journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, volume={33}, number={12}, pages={2341--2353}, year={2011}}
@inproceedings{hendrycks2019corruptions, title={Benchmarking Neural Network Robustness to Common Corruptions and Perturbations}, author={Hendrycks, Dan and Dietterich, Thomas}, booktitle={International Conference on Learning Representations}, year={2019}}
@inproceedings{geifman2017selective, title={Selective Classification for Deep Neural Networks}, author={Geifman, Yonatan and El-Yaniv, Ran}, booktitle={Advances in Neural Information Processing Systems}, year={2017}}
@article{oquab2024dinov2, title={DINOv2: Learning Robust Visual Features without Supervision}, author={Oquab, Maxime and others}, journal={Transactions on Machine Learning Research}, year={2024}}
@article{lu2026seeing, title={Seeing Through Smoke: Surgical Desmoking for Improved Visual Perception}, author={Lu, Jingpei and Jiang, Fengyi and Zhang, Xiaorui and Jin, Lingbo and Mohareri, Omid}, journal={arXiv preprint arXiv:2603.25867}, year={2026}}
@inproceedings{son2020urie, title={{URIE}: Universal Image Enhancement for Visual Recognition in the Wild}, author={Son, Taeyoung and Kang, Juwon and Kim, Namyup and Cho, Sunghyun and Kwak, Suha}, booktitle={European Conference on Computer Vision}, year={2020}}
@article{kumar2025fog, title={From Fog to Failure: The Unintended Consequences of Dehazing on Object Detection in Clear Images}, author={Kumar, Ashutosh and Chadha, Aman}, journal={arXiv preprint arXiv:2502.02027}, year={2025}}
@article{borges2024sdc, title={Soft Dice Confidence: A Near-Optimal Confidence Estimator for Selective Prediction in Semantic Segmentation}, author={Borges, Bruno Laboissiere Camargos and Pacheco, Bruno Machado and Silva, Danilo}, journal={arXiv preprint arXiv:2402.10665}, year={2024}}
@incollection{zuiderveld1994clahe, title={Contrast limited adaptive histogram equalization}, author={Zuiderveld, Karel}, booktitle={Graphics Gems IV}, pages={474--485}, year={1994}, publisher={Academic Press}}
""", encoding="utf-8")
print("written", len((P / "midl-fullpaper.tex").read_text(encoding="utf-8")))
