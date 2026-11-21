"""MIDL robustness pilot on Modal. Frozen JBHI joint baseline (sha256 c8c76331...), no training.
modal run src/modal_run.py [--limit N]"""
import modal
app = modal.App("midl2027-smoke-selector")
vol = modal.Volume.from_name("midl2027", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "torchvision==0.22.1", "numpy>=1.26,<2", "pandas>=2,<3", "transformers>=4.53,<5",
                 "opencv-python-headless<5")
    .env({"HF_HOME": "/mnt/v/cache/hf"})
    .add_local_file("src/model_def.py", "/app/model_def.py").add_local_file("src/smoke_sim.py", "/app/smoke_sim.py"))
CKPT_SHA = "c8c763312cba9cff5bb02d492e548cce0cd0975d13c54c1a40f93e8f578bf192"
SEVS = ["low", "mid", "high"]; W, H = 960, 540

@app.function(image=image, gpu="A10G", volumes={"/mnt/v": vol}, timeout=3 * 3600)
def run(limit: int = 0, rep: int = 0):
    import sys, zipfile, hashlib, os, io, csv, time
    sys.path.insert(0, "/app")
    import numpy as np, cv2, torch, torch.nn.functional as F
    from model_def import DinoScene
    import smoke_sim as S
    d = "/mnt/v/data"
    if not os.path.exists(d + "/imgs"):
        for z in ("imgs", "masks"): zipfile.ZipFile(f"/mnt/v/{z}.zip").extractall(d)
    h = hashlib.sha256(open("/mnt/v/baseline_final.pth", "rb").read()).hexdigest(); assert h == CKPT_SHA, h
    ck = torch.load("/mnt/v/baseline_final.pth", map_location="cpu", weights_only=False)
    m = DinoScene(ck["tasks"], head_types={"anatomy": ck["job"].get("ana_head", "multi")})
    m.load_state_dict(ck["model"], strict=True); m = m.cuda().eval()
    mean = torch.tensor((0.485, 0.456, 0.406), device="cuda").view(1, 3, 1, 1); std = torch.tensor((0.229, 0.224, 0.225), device="cuda").view(1, 3, 1, 1)
    @torch.inference_mode()
    def predict(img):
        x = torch.from_numpy(img).cuda().permute(2, 0, 1)[None]
        x = (F.interpolate(x, size=(518, 518), mode="bilinear", align_corners=False) - mean) / std
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = m(x, ["anatomy", "instrument"])
        out = {}
        for t in o:
            p = F.interpolate(o[t].float(), size=(H, W), mode="bilinear", align_corners=False).softmax(1)[0]
            conf, pred = p.max(0); ent = -(p * (p + 1e-8).log()).sum(0)
            out[t] = (pred, conf, ent, p)
        u = out["anatomy"][0] == 1; i = (out["instrument"][0] > 0)
        fg = lambda mask, conf, ent: (float(conf[mask].mean()) if mask.any() else float("nan"), float(ent[mask].mean()) if mask.any() else float("nan"))
        sdc = lambda pf, y: float(2 * (pf * y).sum() / (y.sum() + pf.sum() + 1e-6))   # soft-Dice-style expected-Dice confidence
        sdc_u = sdc(out["anatomy"][3][1], u.float()); sdc_i = sdc(1 - out["instrument"][3][0], i.float())
        cu, eu = fg(u, out["anatomy"][1], out["anatomy"][2]); ci, ei = fg(i, out["instrument"][1], out["instrument"][2])
        return dict(u=u, i=i, conf_u=cu, ent_u=eu, conf_i=ci, ent_i=ei, area_u=float(u.float().mean()), area_i=float(i.float().mean()),
                    sdc_u=sdc_u, sdc_i=sdc_i, mconf_u=float(out["anatomy"][1].mean()), mconf_i=float(out["instrument"][1].mean()))
    dice = lambda a, b: float((2 * (a & b).sum() + 1e-6) / (a.sum() + b.sum() + 1e-6)) if (a.sum() + b.sum()) > 0 else 1.0
    names = sorted(p[:-4] for p in os.listdir(d + "/imgs") if p.endswith(".jpg"))
    if limit: names = names[::max(1, len(names) // limit)][:limit]
    rows = []; t0 = time.time()
    for k, n in enumerate(names):
        im = cv2.cvtColor(cv2.imread(f"{d}/imgs/{n}.jpg"), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        raw = cv2.imread(f"{d}/masks/{n}.png", cv2.IMREAD_GRAYSCALE); raw = cv2.resize(raw, (W, H), interpolation=cv2.INTER_NEAREST)
        gu = torch.from_numpy(raw == 180).cuda(); gi = torch.from_numpy((raw > 0) & (raw < 180)).cuda()
        variants = [("original", "none", im)] if rep == 0 else []
        for fam, fn in (("A", S.smoke_A), ("B", S.smoke_B)):
            for sv in SEVS:
                rng = np.random.default_rng(S.seed_of(n, fam, sv, rep)); variants.append((fam, sv, fn(im, sv, rng)[0].astype(np.float32)))
        for fam, sv, x in variants:
            base = predict(x)
            for enh in ("none", "clahe", "dcp"):
                y = base if enh == "none" else predict(S.ENHANCERS[enh](x).astype(np.float32))
                rows.append(dict(rep=rep, image=n, clip=n[:3], has_uterus_ref=int(gu.any()), family=fam, severity=sv, enhancer=enh,
                    dice_u=dice(y["u"], gu), dice_i=dice(y["i"], gi), disagree_u=1 - dice(y["u"], base["u"]), disagree_i=1 - dice(y["i"], base["i"]),
                    **{q: y[q] for q in ("conf_u", "ent_u", "conf_i", "ent_i", "area_u", "area_i", "mconf_u", "mconf_i", "sdc_u", "sdc_i")}))
        if k % 50 == 0: print(k, len(names), f"{time.time()-t0:.0f}s", flush=True)
    import pandas as pd
    out = "/mnt/v/per_case_v2" + (f"_pilot{limit}" if limit else "") + f"_rep{rep}.csv"
    pd.DataFrame(rows).to_csv(out, index=False); vol.commit()
    return dict(rows=len(rows), seconds=time.time() - t0, torch=torch.__version__, gpu=torch.cuda.get_device_name(0), sha=h)

@app.local_entrypoint()
def main(limit: int = 0, rep: int = 0):
    print(run.remote(limit, rep))
