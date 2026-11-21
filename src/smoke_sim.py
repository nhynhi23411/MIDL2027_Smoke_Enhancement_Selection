"""Geometry-preserving smoke simulators + conventional enhancers. All operate on float32 RGB in [0,1].
Family A (development): multi-octave fractal haze, grey-white airlight, alpha-blend.
Family B (held-out stress): localized turbulent plumes, tinted, with contrast loss and scatter blur."""
import cv2, numpy as np, hashlib

SEV = {"A": {"low": 0.25, "mid": 0.50, "high": 0.75}, "B": {"low": 0.35, "mid": 0.60, "high": 0.85}}

def seed_of(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)

def _fractal(h, w, rng, octaves=5, base=4):
    out = np.zeros((h, w), np.float32); amp = 1.0; tot = 0
    for o in range(octaves):
        n = max(2, base * 2 ** o)
        g = rng.random((n, int(n * w / h) + 1)).astype(np.float32)
        out += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC); tot += amp; amp *= 0.5
    out /= tot
    return (out - out.min()) / (out.max() - out.min() + 1e-6)

def smoke_A(img, sev, rng):
    h, w = img.shape[:2]
    d = _fractal(h, w, rng)
    cover = {"low": 0.45, "mid": 0.60, "high": 0.75}[sev]          # fraction of frame with smoke
    d = np.clip((d - (1 - cover)) / cover, 0, 1) ** 0.8
    alpha = (SEV["A"][sev] * d)[..., None]
    air = np.array([0.86, 0.86, 0.88], np.float32)
    return img * (1 - alpha) + air * alpha, dict(family="A", coverage=cover, alpha_max=SEV["A"][sev])

def smoke_B(img, sev, rng):
    h, w = img.shape[:2]
    n = int(rng.integers(1, 3)); d = np.zeros((h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    warp = _fractal(h, w, rng, 4, 3)
    for _ in range(n):
        cx, cy = rng.uniform(.2, .8) * w, rng.uniform(.2, .8) * h
        sx, sy = rng.uniform(.18, .35) * w, rng.uniform(.18, .35) * h
        d += np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) * (0.6 + 1.2 * warp))
    d = np.clip(d, 0, 1)
    tint = np.array(rng.choice([[0.80, 0.84, 0.92], [0.88, 0.80, 0.70]]), np.float32)  # bluish or brownish
    alpha_max = SEV["B"][sev]
    alpha = (alpha_max * d)[..., None]
    blur = cv2.GaussianBlur(img, (0, 0), 3 + 6 * alpha_max)               # scatter softens underlying scene
    base = img * (1 - alpha) + blur * alpha
    mean = base.mean((0, 1), keepdims=True)
    base = mean + (base - mean) * (1 - 0.5 * alpha)                         # contrast loss
    return np.clip(base * (1 - alpha) + tint * alpha, 0, 1), dict(family="B", plumes=n, alpha_max=alpha_max)

def clahe(img):
    lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255

def _guided(I, p, r, eps):
    bf = lambda x: cv2.boxFilter(x, -1, (r, r))
    mI, mp = bf(I), bf(p); cov = bf(I * p) - mI * mp; var = bf(I * I) - mI * mI
    a = cov / (var + eps); b = mp - a * mI
    return bf(a) * I + bf(b)

def dcp_dehaze(img, patch=15, omega=0.95, t0=0.1):
    dark = cv2.erode(img.min(2), np.ones((patch, patch), np.uint8))
    idx = np.argsort(dark.ravel())[-max(1, dark.size // 1000):]
    A = np.clip(img.reshape(-1, 3)[idx].mean(0), 0.3, 0.98)
    t = 1 - omega * cv2.erode((img / A).min(2), np.ones((patch, patch), np.uint8))
    t = np.clip(_guided(img.mean(2), t.astype(np.float32), 41, 1e-3), t0, 1)[..., None]
    return np.clip((img - A) / t + A, 0, 1).astype(np.float32)

ENHANCERS = {"clahe": clahe, "dcp": dcp_dehaze}
