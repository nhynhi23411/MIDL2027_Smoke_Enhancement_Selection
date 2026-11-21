"""Pipeline figure (MIDL/MICCAI style, single-column legibility). Thumbnails are real frames passed through the actual
simulators/enhancers; charts are drawn from the result tables. Also exports an editable draw.io file.
python src/make_pipeline_fig.py"""
import sys, numpy as np, pandas as pd, cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from smoke_sim import smoke_A, smoke_B, clahe, dcp_dehaze, seed_of

R = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
FW, FH = 11, 7; W, H = 110, 70
fig = plt.figure(figsize=(FW, FH)); bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W); bg.set_ylim(0, H); bg.axis("off")
GREY, BLUE, ORG, GRN, RED = "#555", "#4c78a8", "#f58518", "#54a24b", "#e45756"
T_, L_, S_ = 10.5, 10, 9

def box(x, y, w, h, fc="white", ec=GREY, ls="-", r=1.0, lw=1.0, z=1):
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, ls=ls, lw=lw, zorder=z))
ARR = []
def arrow(x0, y0, x1, y1, c=GREY):
    ARR.append((x0, y0, x1, y1, c))
    bg.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, lw=1.2, color=c, zorder=5))
def text(x, y, s, **k):
    k.setdefault("ha", "center"); k.setdefault("va", "center"); k.setdefault("fontsize", L_); bg.text(x, y, s, zorder=6, **k)
def thumb(x, y, w, h, img, ec="k", lw=0.6):
    ax = fig.add_axes([x / W, y / H, w / W, h / H]); ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_edgecolor(ec); s.set_linewidth(lw)
    return ax
def inset(x, y, w, h): return fig.add_axes([x / W, y / H, w / W, h / H])

# ---------- real example frame ----------
imgdir, mdir = R / "data/autolaparo/imgs", R / "data/autolaparo/masks"
name = None
for f in sorted(imgdir.glob("2[4-9]*.jpg"))[::3]:
    m = cv2.imread(str(mdir / (f.stem + ".png")), 0)
    if m is not None and (m == 180).mean() > 0.10: name = f.stem; break
img = cv2.cvtColor(cv2.imread(str(imgdir / (name + ".jpg"))), cv2.COLOR_BGR2RGB)
mask = cv2.imread(str(mdir / (name + ".png")), 0); sm = (960, 540)
img = cv2.resize(img, sm, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
mask = cv2.resize(mask, sm, interpolation=cv2.INTER_NEAREST); ut, ins = (mask == 180), (mask > 0) & (mask < 180)
def overlay(im, a=0.45):
    o = im.copy(); o[ut] = o[ut] * (1 - a) + np.array([.1, .9, .3]) * a; o[ins] = o[ins] * (1 - a) + np.array([.95, .5, .1]) * a
    return o
def sim(fam, sev):
    rng = np.random.default_rng(seed_of(name, fam, sev, 0)); return (smoke_A if fam == "A" else smoke_B)(img, sev, rng)[0]
A_mid, B_hi = sim("A", "mid"), sim("B", "high")

# ======================= (a) PIPELINE =======================
box(1, 35, 108, 34, ls=(0, (4, 3)), r=1.5)
text(3, 66.6, "(a)  Pipeline: corrupt, enhance, segment, select", ha="left", style="italic", color="#444", fontsize=T_)
BX = [3, 24, 45, 66, 87]; BW, BY, BH = 19, 37, 26
for x in BX: box(x, BY, BW, BH, fc="#fafafa")
for i in range(4): arrow(BX[i] + BW + 0.2, 50, BX[i + 1] - 0.2, 50)
# 1 frame
x = BX[0]; text(x + BW / 2, 60.3, "Clean frame", fontweight="bold", fontsize=L_)
thumb(x + 1.5, 49.5, 16, 9, overlay(img)); text(x + BW / 2, 46.3, "AutoLaparo mask:", fontsize=S_)
bg.plot([x + 4], [43.3], "s", c=GRN, ms=5, zorder=7); text(x + 5.2, 43.3, "uterus", ha="left", fontsize=S_)
bg.plot([x + 4], [40], "s", c=ORG, ms=5, zorder=7); text(x + 5.2, 40, "instruments", ha="left", fontsize=S_)
# 2 simulators
x = BX[1]; text(x + BW / 2, 60.3, "Smoke simulators", fontweight="bold", fontsize=L_)
thumb(x + 3.6, 52.2, 11.8, 6.6, A_mid); text(x + BW / 2, 50.4, "A haze (dev)", fontsize=S_)
thumb(x + 3.6, 43.2, 11.8, 6.6, B_hi); text(x + BW / 2, 41.2, "B plume (test)", fontsize=S_)
# 3 enhancers
x = BX[2]; text(x + BW / 2, 60.3, "Enhancers", fontweight="bold", fontsize=L_)
for i, (lab, im) in enumerate([("none", B_hi), ("CLAHE", clahe(B_hi)), ("DCP", dcp_dehaze(B_hi))]):
    thumb(x + 1.2, 52 - i * 7.3, 11.5, 6.4, im); text(x + 13.6, 55.2 - i * 7.3, lab, ha="left", fontsize=S_)
# 4 segmenter
x = BX[3]; box(x, BY, BW, BH, fc="#eef3fa", ec=BLUE); text(x + BW / 2, 60.3, "Frozen segmenter", fontweight="bold", fontsize=L_, color=BLUE)
bg.add_patch(Rectangle((x + 1.5, 49), 16, 7, fc="#9ecae1", ec=BLUE, lw=.8, zorder=3)); text(x + BW / 2, 52.5, "DINOv2-B", fontsize=L_)
bg.add_patch(Rectangle((x + 1.5, 39.5), 7.5, 5.5, fc="#fdd0a2", ec=ORG, lw=.8, zorder=3)); text(x + 5.25, 42.2, "tools", fontsize=S_)
bg.add_patch(Rectangle((x + 10, 39.5), 7.5, 5.5, fc="#c7e9c0", ec=GRN, lw=.8, zorder=3)); text(x + 13.75, 42.2, "uterus", fontsize=S_)
arrow(x + 5.25, 49, x + 5.25, 45.2, c=BLUE); arrow(x + 13.75, 49, x + 13.75, 45.2, c=BLUE)
# 5 selector
x = BX[4]; box(x, BY, BW, BH, fc="#fff4e5", ec=ORG); text(x + BW / 2, 60.3, "Selector", fontweight="bold", fontsize=L_, color="#b35c00")
for k, line in enumerate(["label-free features", "ridge quality $\\hat q$", "raw / enhanced /", "abstain"]):
    text(x + BW / 2, 55.2 - k * 3.6, line, fontsize=S_)
text(x + BW / 2, 40.2, "no smoke params,\nno reference mask", fontsize=S_ - 0.5, color=RED)

# ======================= (b) RESULTS =======================
box(1, 1, 108, 33, ls=(0, (4, 3)), r=1.5)
text(3, 31.6, "(b)  Final block: 149 uterus-positive frames, 31 clips (3 smoke repetitions, 95% clip-bootstrap CI)", ha="left", style="italic", color="#444", fontsize=T_ - 0.5)
cmp_ = pd.read_csv(R / "outputs/analysis/table_main_comparison.csv")
regs = ["original", "A_in_regime", "B_heldout"]; labs = ["clean", "smoke A", "smoke B"]
rules = [("always_raw", "unenhanced", "#9e9e9e"), ("always_enh", "always DCP", RED), ("selector", "selector", BLUE), ("oracle", "oracle", "#cfcfcf")]
ax = inset(10, 6, 26, 18)
for k, (r_, l_, c_) in enumerate(rules):
    v = [cmp_[(cmp_.regime == g) & (cmp_.rule == r_)].dice.iloc[0] for g in regs]
    ax.bar(np.arange(3) + (k - 1.5) * 0.2, v, 0.19, color=c_, label=l_, edgecolor="none")
ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=S_); ax.set_ylim(0.3, 0.72); ax.set_yticks([0.3, 0.4, 0.5]); ax.set_ylabel("Dice (uterus)", fontsize=S_)
ax.tick_params(labelsize=S_); ax.legend(fontsize=8.5, ncol=2, frameon=False, loc="upper center", handlelength=1, columnspacing=.8, borderaxespad=0.1)
for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
ax.set_title("Dice by regime", fontsize=L_)
d = pd.read_csv(R / "outputs/analysis/table_enhancement_outcomes.csv")
order = [("original", "none"), ("A", "low"), ("A", "mid"), ("A", "high"), ("B", "low"), ("B", "mid"), ("B", "high")]
lab = ["clean", "A-lo", "A-mid", "A-hi", "B-lo", "B-mid", "B-hi"]
ax2 = inset(52, 6, 22, 18)
dv = [d[(d.family == f) & (d.severity == s_)].iloc[0] for f, s_ in order]; y = np.arange(len(dv))[::-1]
ax2.barh(y, [r.delta for r in dv], color=[RED if r.family == "B" else ("#999" if r.family == "original" else ORG) for r in dv], height=.6)
ax2.errorbar([r.delta for r in dv], y, xerr=[[r.delta - r.lo for r in dv], [r.hi - r.delta for r in dv]], fmt="none", ecolor="k", lw=.8, capsize=1.5)
ax2.axvline(0, c="k", lw=.7); ax2.set_yticks(y); ax2.set_yticklabels(lab, fontsize=S_); ax2.set_xticks([-0.15, -0.10, -0.05, 0]); ax2.set_xticklabels(["-.15", "-.10", "-.05", "0"], fontsize=S_)
for s_ in ["top", "right"]: ax2.spines[s_].set_visible(False)
ax2.set_title("$\\Delta$Dice (DCP $-$ none)", fontsize=L_)
sel = pd.read_csv(R / "outputs/analysis/table_selective.csv"); sel = sel[sel.regime == "B_heldout"]
ax3 = inset(85, 6, 21, 18)
nm = {"selector_quality": "selector", "confidence_only": "fg conf.", "soft_dice_confidence": "SDC"}
vals = [sel[sel.score == k].dice_at_80pct_ranked.iloc[0] for k in nm]
ax3.bar(range(3), vals, color=[BLUE, "#9e9e9e", "#9e9e9e"], width=.6)
for i_, v in enumerate(vals): ax3.text(i_, v + .002, f"{v:.3f}", ha="center", fontsize=8.5)
ax3.set_xticks(range(3)); ax3.set_xticklabels(list(nm.values()), fontsize=S_); ax3.set_ylim(0.50, 0.59); ax3.set_yticks([0.50, 0.55]); ax3.tick_params(labelsize=S_)
for s_ in ["top", "right"]: ax3.spines[s_].set_visible(False)
ax3.set_title("Dice @ 80% coverage (B)", fontsize=L_)
fig.savefig(R / "paper/figs/fig_pipeline.png", dpi=220); fig.savefig(R / "paper/figs/fig_pipeline.pdf"); print("saved")

# ======================= draw.io export (editable) =======================
import base64, re
from matplotlib.colors import to_hex
from matplotlib.patches import FancyBboxPatch as FB
fig.canvas.draw(); rend = fig.canvas.get_renderer()
F = 1600 / (FW * fig.dpi); HP = 1600 * FH / FW
cells = []; nid = [2]
def _id(): nid[0] += 1; return f"c{nid[0]}"
def P(xd, yd): return xd * F, HP - yd * F
def tex(t):
    for a, b in [(r"\varnothing", "∅"), (r"\approx", "≈"), (r"\Delta", "Δ"), (r"\hat Y", "Ŷ"), (r"\hat q", "q̂"), (r"\hat", ""),
                 (r"\mathcal{S}", "S"), (r"\mathcal{E}", "E"), (r"\mathcal", ""), (r"\phi", "φ"), (r"\tau", "τ"), (r"\times", "×"),
                 (r"\arg", "arg"), (r"\max", "max"), (r"\bullet", "•"), (r"\\", ""), ("$", ""), ("{", ""), ("}", ""), ("_", "")]:
        t = t.replace(a, b)
    return t.replace("\n", "<br>")
def rect(x0, y0, x1, y1, fc, ec, dash=False, rounded=False, sw=1):
    a, b = P(x0, y1); c, d = P(x1, y0)
    st = f"rounded={int(rounded)};arcSize=3;whiteSpace=wrap;html=1;fillColor={fc};strokeColor={ec};strokeWidth={sw};" + ("dashed=1;" if dash else "")
    cells.append(f'<mxCell id="{_id()}" value="" style="{st}" vertex="1" parent="1"><mxGeometry x="{a:.1f}" y="{b:.1f}" width="{c-a:.1f}" height="{d-b:.1f}" as="geometry"/></mxCell>')
def label(xd, yd, t, fs, col="#000000", ha="center", bold=False, ital=False):
    t = tex(t)
    if not t: return
    x, y = P(xd, yd); fsz = fs * fig.dpi / 72 * F; w = max(20, max(len(l) for l in t.split("<br>")) * fsz * 0.62 + 8); h = fsz * 1.5 * (t.count("<br>") + 1) + 4
    x0 = {"center": x - w / 2, "left": x, "right": x - w}[ha]
    fs_style = (1 if bold else 0) | (2 if ital else 0)
    cells.append(f'<mxCell id="{_id()}" value="{__import__("html").escape(t, quote=True)}" style="text;html=1;strokeColor=none;fillColor=none;align={ha};verticalAlign=middle;whiteSpace=nowrap;fontSize={fsz:.0f};fontColor={col};fontStyle={fs_style};" vertex="1" parent="1"><mxGeometry x="{x0:.1f}" y="{y-h/2:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"/></mxCell>')
def line(x0, y0, x1, y1, col="#000000", arrow_=False, sw=1):
    cells.append(f'<mxCell id="{_id()}" style="endArrow={"block" if arrow_ else "none"};html=1;strokeColor={col};strokeWidth={sw};" edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="{x0:.1f}" y="{y0:.1f}" as="sourcePoint"/><mxPoint x="{x1:.1f}" y="{y1:.1f}" as="targetPoint"/></mxGeometry></mxCell>')
# background panels, boxes and rectangles (draw order preserved)
for pt in bg.patches:
    if isinstance(pt, FancyArrowPatch): continue
    e = pt.get_window_extent(rend)
    dash = pt.get_linestyle() != "-"
    rect(e.x0, e.y0, e.x1, e.y1, to_hex(pt.get_facecolor()), to_hex(pt.get_edgecolor()), dash, isinstance(pt, FB), 2 if dash else 1)
for (x0, y0, x1, y1, c) in ARR:
    a = bg.transData.transform((x0, y0)); b = bg.transData.transform((x1, y1)); pa, pb = P(*a), P(*b); line(*pa, *pb, col=c, arrow_=True, sw=2)
for ln in bg.lines:
    xd, yd = bg.transData.transform((ln.get_xdata()[0], ln.get_ydata()[0])); x, y = P(xd, yd)
    cells.append(f'<mxCell id="{_id()}" value="" style="whiteSpace=wrap;html=1;fillColor={to_hex(ln.get_color())};strokeColor=none;" vertex="1" parent="1"><mxGeometry x="{x-4:.1f}" y="{y-4:.1f}" width="8" height="8" as="geometry"/></mxCell>')
# thumbnails and charts
for ax in fig.axes:
    if ax is bg: continue
    bb = ax.get_window_extent(rend)
    if ax.images:
        im = (np.clip(ax.images[0].get_array(), 0, 1) * 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf.tobytes()).decode(); a, b = P(bb.x0, bb.y1); c, d = P(bb.x1, bb.y0)
        cells.append(f'<mxCell id="{_id()}" value="" style="shape=image;imageAspect=0;aspect=fixed;verticalLabelPosition=bottom;verticalAlign=top;strokeColor=#000000;image=data:image/jpeg,{b64};" vertex="1" parent="1"><mxGeometry x="{a:.1f}" y="{b:.1f}" width="{c-a:.1f}" height="{d-b:.1f}" as="geometry"/></mxCell>')
        continue
    for pt in ax.patches:
        e = pt.get_window_extent(rend); rect(e.x0, e.y0, e.x1, e.y1, to_hex(pt.get_facecolor()), "none")
    for ln in ax.lines:
        xy = ax.transData.transform(np.c_[ln.get_xdata(), ln.get_ydata()])
        for u, v in zip(xy[:-1], xy[1:]): line(*P(*u), *P(*v), sw=1)
    # axes frame
    for (u, v) in [((bb.x0, bb.y0), (bb.x0, bb.y1)), ((bb.x0, bb.y0), (bb.x1, bb.y0))]: line(*P(*u), *P(*v), sw=1)
    for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()) + list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]:
        if not t.get_text(): continue
        e = t.get_window_extent(rend); label((e.x0 + e.x1) / 2, (e.y0 + e.y1) / 2, t.get_text(), t.get_fontsize())
    lg = ax.get_legend()
    if lg:
        for ptc, tx in zip(lg.get_patches(), lg.get_texts()):
            e = ptc.get_window_extent(rend); rect(e.x0, e.y0, e.x1, e.y1, to_hex(ptc.get_facecolor()), "none")
            e = tx.get_window_extent(rend); label(e.x0, (e.y0 + e.y1) / 2, tx.get_text(), 5.5, ha="left")
# texts last (on top)
for t in bg.texts:
    xd, yd = bg.transData.transform(t.get_position()); ha = t.get_ha(); bold = t.get_fontweight() in ("bold", 700)
    label(xd, yd, t.get_text(), t.get_fontsize(), to_hex(t.get_color()), ha, bold, t.get_fontstyle() == "italic")
xml = ('<mxfile host="app.diagrams.net"><diagram name="pipeline" id="pipe"><mxGraphModel dx="1600" dy="800" grid="1" gridSize="10" page="1" pageWidth="1600" pageHeight="800" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
       + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")
(R / "paper/figs/fig_pipeline.drawio").write_text(xml, encoding="utf-8"); print("drawio", len(cells), "cells")
