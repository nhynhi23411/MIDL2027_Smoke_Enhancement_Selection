"""End-to-end timing of the two-branch pipeline on one A10G (batch 1, 1920x1080 JPEG in -> chosen mask)."""
import modal
app = modal.App("midl2027-timing")
vol = modal.Volume.from_name("midl2027")
image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "torchvision==0.22.1", "numpy>=1.26,<2", "pandas>=2,<3", "transformers>=4.53,<5", "opencv-python-headless<5")
    .env({"HF_HOME": "/mnt/v/cache/hf"}).add_local_file("src/model_def.py", "/app/model_def.py").add_local_file("src/smoke_sim.py", "/app/smoke_sim.py"))

@app.function(image=image, gpu="A10G", volumes={"/mnt/v": vol}, timeout=1800)
def run(n: int = 60):
    import sys, os, time, torch, numpy as np, cv2, torch.nn.functional as F, statistics as st
    sys.path.insert(0, "/app"); from model_def import DinoScene; import smoke_sim as S
    ck = torch.load("/mnt/v/baseline_final.pth", map_location="cpu", weights_only=False)
    m = DinoScene(ck["tasks"], head_types={"anatomy": ck["job"].get("ana_head", "multi")}); m.load_state_dict(ck["model"], strict=True); m = m.cuda().eval()
    mean = torch.tensor((0.485, 0.456, 0.406), device="cuda").view(1, 3, 1, 1); std = torch.tensor((0.229, 0.224, 0.225), device="cuda").view(1, 3, 1, 1)
    def sync(): torch.cuda.synchronize(); return time.perf_counter()
    @torch.inference_mode()
    def predict(img):
        x = torch.from_numpy(img).cuda().permute(2, 0, 1)[None]; x = (F.interpolate(x, size=(518, 518), mode="bilinear", align_corners=False) - mean) / std
        with torch.autocast("cuda", dtype=torch.bfloat16): o = m(x, ["anatomy", "instrument"])
        p = F.interpolate(o["anatomy"].float(), size=(540, 960), mode="bilinear", align_corners=False).softmax(1)[0]; c, pr = p.max(0); return pr, c
    files = sorted(os.listdir("/mnt/v/data/imgs"))[:n + 5]; T = {k: [] for k in ("decode_resize", "branch_raw", "enhance_dcp", "branch_enh", "selector_score", "total")}
    for k, f in enumerate(files):
        t0 = time.perf_counter(); im = cv2.imread(f"/mnt/v/data/imgs/{f}"); im = cv2.cvtColor(cv2.resize(im, (960, 540), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        t1 = sync(); predict(im); t2 = sync(); e = S.dcp_dehaze(im).astype(np.float32); t3 = time.perf_counter(); predict(e); t4 = sync()
        a, b = predict(im)[1].mean().item(), 0.0; t5 = sync()
        if k >= 5: [T[x].append(v) for x, v in zip(T, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t5 - t0))]
    return {k: dict(median_ms=1000 * st.median(v), p95_ms=1000 * sorted(v)[int(.95 * len(v))]) for k, v in T.items()} | {"gpu": torch.cuda.get_device_name(0), "n": n}

@app.local_entrypoint()
def main(): print(run.remote())
