"""Transmission and gain statistics of the dark-channel-prior enhancer on UNSMOKED final-block frames (paper, Proposition on DCP gain).
python src/dcp_gain.py <imgdir> <out.json>"""
import sys, json, cv2, numpy as np
from pathlib import Path
imgdir, out = Path(sys.argv[1]), sys.argv[2]; P, OM, T0 = 15, 0.95, 0.1
names = sorted(imgdir.glob("*.jpg")); names = [f for f in names if int(f.stem[:3]) > 227][::6]
g_all, t_all, m_all = [], [], []
for f in names:
    im = cv2.resize(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB), (960, 540), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    dark = cv2.erode(im.min(2), np.ones((P, P), np.uint8)); idx = np.argsort(dark.ravel())[-max(1, dark.size // 1000):]
    A = np.clip(im.reshape(-1, 3)[idx].mean(0), 0.3, 0.98)
    m = cv2.erode((im / A).min(2), np.ones((P, P), np.uint8)); t = np.clip(1 - OM * m, T0, 1)
    m_all.append(m.ravel()[::50]); t_all.append(t.ravel()[::50]); g_all.append((1 / t).ravel()[::50])
m, t, g = map(np.concatenate, (m_all, t_all, g_all))
res = dict(n_frames=len(names), omega=OM, patch=P, t_floor=T0, median_m=float(np.median(m)), median_t=float(np.median(t)), mean_gain=float(g.mean()),
           median_gain=float(np.median(g)), p90_gain=float(np.percentile(g, 90)), frac_gain_gt_1p5=float((g > 1.5).mean()), frac_gain_gt_2=float((g > 2).mean()))
json.dump(res, open(out, "w"), indent=1); print(res)
