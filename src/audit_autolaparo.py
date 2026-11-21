import sys, csv, hashlib, collections
from pathlib import Path
import numpy as np
from PIL import Image
root = Path(sys.argv[1]); out = Path(sys.argv[2])
rows = []; vals = collections.Counter()
for p in sorted((root/'imgs').glob('*.jpg')):
    m = root/'masks'/f'{p.stem}.png'
    im = Image.open(p); ok = m.exists()
    row = dict(name=p.stem, clip=p.stem[:3], frame=int(p.stem[3:]), img_w=im.size[0], img_h=im.size[1], mask_exists=ok)
    if ok:
        a = np.asarray(Image.open(m)); 
        if a.ndim == 3: a = a[..., 0]
        u = np.unique(a); vals.update(u.tolist())
        row.update(mask_w=a.shape[1], mask_h=a.shape[0], size_match=a.shape==(im.size[1], im.size[0]),
                   labels=';'.join(map(str,u)), uterus_px=int((a==180).sum()), tool_px=int(((a>0)&(a<180)).sum()),
                   bad_values=';'.join(str(v) for v in u if v not in range(0,181,20)))
    row['sha1'] = hashlib.sha1(p.read_bytes()).hexdigest()
    rows.append(row)
keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
n = len(rows); print('images', n, 'clips', len({r['clip'] for r in rows}))
print('missing mask', sum(not r['mask_exists'] for r in rows), 'size mismatch', sum(r.get('size_match') is False for r in rows))
print('sizes', collections.Counter((r['img_w'], r['img_h']) for r in rows))
print('mask values', dict(sorted(vals.items())))
print('bad values rows', sum(bool(r.get('bad_values')) for r in rows))
print('uterus present', sum(r.get('uterus_px', 0) > 0 for r in rows), 'tool present', sum(r.get('tool_px', 0) > 0 for r in rows))
d = collections.Counter(r['sha1'] for r in rows); print('duplicate hashes', sum(c > 1 for c in d.values()))
