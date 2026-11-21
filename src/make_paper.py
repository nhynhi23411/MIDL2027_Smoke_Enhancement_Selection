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

\section{Setup and definitions}
For one frame let $D_0$ be the Dice of the unenhanced prediction and $D_e$ that of the prediction after enhancer $e$, and let $G_e=D_e-D_0$. With a margin of $0.05$ fixed before final evaluation we call $G_e>0.05$ \emph{help} and $G_e<-0.05$ \emph{harm}. For a policy $\pi$ that chooses between the two branches let $D_\pi$ be the Dice of its choice and $D^{*}=\max(D_0,D_e)$ the reference-aware oracle, which cannot be deployed. \emph{Headroom} is $\mathrm{HD}=\mathbb{E}[D^{*}-D_0]$, \emph{selection regret} is $\mathrm{RD}(\pi)=\mathbb{E}[D^{*}-D_\pi]$, and \emph{coverage} is the fraction of frames a policy does not abstain on.

\begin{lemma}[Gain bookkeeping]
For any policy $\pi$, $\ \mathbb{E}[D_\pi-D_0]=\mathrm{HD}-\mathrm{RD}(\pi)$.
\end{lemma}
\noindent This is an identity, not a new result: $D_\pi-D_0=(D^{*}-D_0)-(D^{*}-D_\pi)$. Its use is to place a loss: either the enhancer offers little headroom (HD small) or selection wastes what is there (RD large). Table~\ref{tab:hh} reports all three terms.

\textbf{Selector and abstention.} A ridge model predicts the Dice of each branch from label-free features $z$ (confidence, entropy, area, soft-Dice-style confidence, branch disagreement). It picks the branch with the higher predicted Dice $\hat q$, and it abstains when the chosen $\hat q$ falls below a threshold $\tau$ fixed on development data. A selector can only add value beyond a confidence rule if $\hat q$ ranks frames better than confidence does, which is what the risk--coverage comparison tests.

\section{Data, exposure and protocol}
\textbf{AutoLaparo Task 3} provides %(nimg)d frames from %(nclip)d clips (six frames per clip) with uterus and instrument masks; %(nu)d frames have a uterus label. A missing uterus label is not treated as evidence of absence, so the uterus endpoint uses only the %(nu)d positive frames and instruments are reported as a secondary task on all frames. Instrument parts are merged into one foreground class. Image--mask alignment, label ranges and duplicates were checked automatically (no missing masks, no size mismatches, no duplicate hashes).

\textbf{Prior exposure.} The recognition model is our previously developed joint model (DINOv2-Base, instrument and anatomy heads, frozen checkpoint with recorded SHA-256, no retraining here). All AutoLaparo frames were used as an external test set in that earlier work, and an earlier abandoned fine-tuning experiment on AutoLaparo has unresolved membership. \emph{These data are therefore not independent external validation}; this study measures robustness and selection, not generalisation.

\textbf{Splits.} Procedure identifiers are unavailable. Clip index is used as a proxy; clips 001--170, 171--227 and 228--300 form selector-development, development-validation and final-test blocks, following the contiguous official partition, which likely separates source videos (an assumption that we could not verify). All variants of a frame stay in one block. The final block contains %(nfin)d uterus-positive frames from %(nfc)d clips. Smoke realisations are not counted as independent samples.

\section{Study design}

\textbf{Smoke simulators} (geometry preserving; masks unchanged). Family~A (development): multi-octave fractal haze blended with a grey-white airlight, three severities (maximum opacity 0.25/0.50/0.75). Family~B (held out, used only for final testing): one to two localised turbulent plumes, bluish or brownish tint, scatter blur and contrast loss, three severities (0.35/0.60/0.85). Severity is defined by simulator parameters, not clinical grading; seeds derive from image, family and severity. Original frames serve as the control.

\textbf{Enhancers.} CLAHE on luminance (conventional contrast control) and single-image dark-channel-prior dehazing with a guided-filter transmission. The released DeSmoke-LAP weights were not available to us; a learned desmoker is used only on DeSmoke-LAP's released output clips (Section~\ref{sec:ds}). All frames are evaluated at $960{\times}540$ after downscaling.

\textbf{Branches and selector.} Each input yields an unenhanced and an enhanced prediction from the same frozen model. A ridge-regularised quality estimator predicts the Dice of each branch from observable quantities only: mean foreground confidence, mean foreground entropy, predicted area, whole-image mean confidence, branch disagreement (one minus Dice between branches) and a missing-foreground flag. Simulator parameters and reference masks are never inputs. A soft-Dice-style expected-Dice confidence is among the features. The selector picks the branch with higher predicted quality; its abstention score is the predicted quality of the chosen branch. It is fitted on original and family-A frames of the development blocks only; the penalty is chosen on the development-validation block, then refit on all development data and frozen.

\textbf{Controls.} Always unenhanced; always enhanced; confidence only (higher foreground confidence); confidence only on whole-image mean confidence; a disagreement rule (enhanced only if branches agree, threshold tuned on development data); and a reference-aware oracle that serves only as an upper bound. The \emph{strongest simple control} is chosen on development data (here: %(strong)s).

\textbf{Statistics.} Dice is averaged over variants within a frame, frames within a clip, and compared with a clustered bootstrap (4000 resamples of clips). Primary contrast: selector minus the strongest simple control, no abstention, on held-out family~B. Planned contrasts are Holm-adjusted. Secondary: Dice at 80\%% coverage ranked by each frozen score, area under the risk--coverage curve (AURC), and deployment thresholds fixed on development data with achieved coverage. Enhancement outcomes: paired Dice change and the share of frames with a drop larger than 0.05 (fixed before final evaluation).

\section{Results}
\subsection{Enhancement lowers uterus Dice, with or without smoke}
Smoke reduces uterus Dice modestly (clip-averaged unenhanced Dice %(or).3f on original frames, %(ra).3f for family~A, %(rb_).3f for family~B; \figureref{fig:deg}). Enhancement did not recover it on average (\tableref{tab:enh}). DCP lowered Dice in every condition, including on original frames (%(od).3f vs %(or).3f), with %(ld0).0f--%(ld1).0f\%% of frames dropping by more than 0.05. Counted per frame (\tableref{tab:hh}), DCP harmed %(hox).0f\%% of clean frames, %(hax).0f\%% under family~A and %(hbx).0f\%% under held-out family~B, and helped only %(hoh).0f\%%, %(hah).0f\%% and %(hbh).0f\%%. CLAHE was gentler (original %(oc).3f) with intervals that include zero for most conditions, but still harmful on average under family~B. Instrument Dice (secondary, \tableref{tab:ins} in the appendix) changes by at most 0.02 and is improved by DCP only under family~A (high and mid severity).

\begin{figure}[!htbp]
\floatconts{fig:deg}{\caption{Uterus Dice versus severity (final block): unenhanced, DCP-enhanced and selector output.}}
{\includegraphics[width=\linewidth]{figs/fig_degradation_recovery.png}}
\end{figure}
\begin{table}[!htbp]
\floatconts{tab:enh}{\caption{Enhancement effect on uterus segmentation, final block (DCP). Dice pooled over frames; $\Delta$ is the clip-averaged paired change [95\%% CI]; last column is the share of frames with a drop above 0.05. CLAHE: see Appendix~\ref{app:supp}.}}
{\scriptsize\begin{tabular}{llcccc}\toprule Family & Sev. & Unenh. & Enh. & $\Delta$ & Drop$>$0.05\\\midrule
%(tenh)s
\bottomrule\end{tabular}}
\end{table}

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

\begin{table}[!htbp]
\floatconts{tab:ds}{\caption{DeSmoke-LAP descriptive outputs of the frozen pipeline (no reference masks; not accuracy). Pairs show unenhanced / enhanced. Selector and abstention use the AutoLaparo-fitted policy with a development-fixed threshold.}}
{\tiny\begin{tabular}{llrcccrr}\toprule Source & Enhancer & $n$ & Mean conf. & Uterus present & Disagr. & Picks enh. & Abstain\\\midrule
%(tds)s
\bottomrule\end{tabular}}
\end{table}

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

\midlacknowledgments{This research was funded by the University of Economics and Law, Ho Chi Minh City, Vietnam, and Vietnam National University, Ho Chi Minh City, Vietnam. The datasets AutoLaparo and DeSmoke-LAP are used under their respective licences.}
\bibliography{refs}
\appendix
\section{Supplementary results}
\label{app:supp}
Tables~\ref{tab:con} and~\ref{tab:sel} give all planned contrasts and the selective-prediction summary for DCP. Tables~\ref{tab:mainc} and~\ref{tab:enhc} repeat the main comparison and the enhancement effect with CLAHE as the enhanced branch; Table~\ref{tab:ins} gives the secondary instrument endpoint.
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
         ("(DINOv2-Base,", r"(DINOv2-Base~\citep{oquab2024dinov2},"), ("CLAHE on luminance (conventional", r"CLAHE~\citep{zuiderveld1994clahe} on luminance (conventional"),
         ("dark-channel-prior dehazing with", r"dark-channel-prior (DCP) dehazing~\citep{he2011dcp} with"), ("DeSmoke-LAP has no organ masks", r"DeSmoke-LAP~\citep{pan2022desmoke} has no organ masks")]
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
         tsel=table_sel("dcp"), tins=table_ins(), tds=table_ds(), thh=table_hh(), tpol=table_pol(), **HHS)
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
