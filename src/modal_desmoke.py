"""Experiment 3: frozen pipeline on DeSmoke-LAP (no anatomical references -> descriptive outputs only)."""
import modal
app = modal.App("midl2027-desmoke")
vol = modal.Volume.from_name("midl2027")
image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "torchvision==0.22.1", "numpy>=1.26,<2", "pandas>=2,<3", "transformers>=4.53,<5", "opencv-python-headless<5")
    .env({"HF_HOME": "/mnt/v/cache/hf"})
    .add_local_file("src/model_def.py", "/app/model_def.py").add_local_file("src/smoke_sim.py", "/app/smoke_sim.py"))
W, H = 960, 540

@app.function(image=image, gpu="A10G", volumes={"/mnt/v": vol}, timeout=3600)
def run():
    import sys, os, csv, torch, numpy as np, cv2, pandas as pd, torch.nn.functional as F
    sys.path.insert(0, "/app"); from model_def import DinoScene; import smoke_sim as S
    ck = torch.load("/mnt/v/baseline_final.pth", map_location="cpu", weights_only=False)
    m = DinoScene(ck["tasks"], head_types={"anatomy": ck["job"].get("ana_head", "multi")}); m.load_state_dict(ck["model"], strict=True); m = m.cuda().eval()
    mean = torch.tensor((0.485, 0.456, 0.406), device="cuda").view(1, 3, 1, 1); std = torch.tensor((0.229, 0.224, 0.225), device="cuda").view(1, 3, 1, 1)
    @torch.inference_mode()
    def predict(img):
        x = (F.interpolate(torch.from_numpy(img).cuda().permute(2, 0, 1)[None], size=(518, 518), mode="bilinear", align_corners=False) - mean) / std
        with torch.autocast("cuda", dtype=torch.bfloat16): o = m(x, ["anatomy", "instrument"])
        r = {}
        for t in o:
            p = F.interpolate(o[t].float(), size=(H, W), mode="bilinear", align_corners=False).softmax(1)[0]; c, pr = p.max(0); r[t] = (pr, c, -(p * (p + 1e-8).log()).sum(0), p)
        u = r["anatomy"][0] == 1; i = r["instrument"][0] > 0
        f = lambda k, c, e: (float(c[k].mean()) if k.any() else float("nan"), float(e[k].mean()) if k.any() else float("nan"))
        sdc = lambda pf, y: float(2 * (pf * y).sum() / (y.sum() + pf.sum() + 1e-6)); sdc_u = sdc(r["anatomy"][3][1], u.float()); sdc_i = sdc(1 - r["instrument"][3][0], i.float())
        cu, eu = f(u, r["anatomy"][1], r["anatomy"][2]); ci, ei = f(i, r["instrument"][1], r["instrument"][2])
        return dict(u=u, i=i, conf_u=cu, ent_u=eu, conf_i=ci, ent_i=ei, area_u=float(u.float().mean()), area_i=float(i.float().mean()),
                    sdc_u=sdc_u, sdc_i=sdc_i, mconf_u=float(r["anatomy"][1].mean()), mconf_i=float(r["instrument"][1].mean()))
    dice = lambda a, b: float((2 * (a & b).sum() + 1e-6) / (a.sum() + b.sum() + 1e-6)) if (a.sum() + b.sum()) > 0 else 1.0
    man = pd.read_csv("/mnt/v/desmoke/manifest.csv"); rows = []
    for cond, v, i in man.itertuples(index=False):
        if cond == "DeSmoke-LAP": continue
        path = f"/mnt/v/desmoke/clip/Input/{v}_{int(i):03d}.jpg" if cond == "Input" else f"/mnt/v/desmoke/{cond}/{i}.jpg"
        im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        base = predict(im); outs = {"none": base, "clahe": predict(S.clahe(im).astype(np.float32)), "dcp": predict(S.dcp_dehaze(im).astype(np.float32))}
        if cond == "Input":
            dp = f"/mnt/v/desmoke/clip/DeSmoke-LAP/{v}_{int(i):03d}.jpg"
            outs["desmoke_lap"] = predict(cv2.cvtColor(cv2.imread(dp), cv2.COLOR_BGR2RGB).astype(np.float32) / 255)
        for e, y in outs.items():
            rows.append(dict(condition=cond, video=v, frame=str(i), enhancer=e, disagree_u=1 - dice(y["u"], base["u"]), disagree_i=1 - dice(y["i"], base["i"]),
                             **{q: y[q] for q in ("conf_u", "ent_u", "conf_i", "ent_i", "area_u", "area_i", "mconf_u", "mconf_i", "sdc_u", "sdc_i")}))
    pd.DataFrame(rows).to_csv("/mnt/v/desmoke_results.csv", index=False); vol.commit(); return len(rows)

@app.local_entrypoint()
def main(): print(run.remote())
