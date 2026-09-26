"""One short GIF of real footage per evaluation map (every second tic of 256, half size), plus a contact
sheet with one frame per map. CPU only; reads the raw parquet through encode_parquet's frame decoder."""
import json, os, sys, glob, io
import numpy as np, pyarrow.parquet as pq
from PIL import Image
D = "/sata2/data/rnagabhi/doom"
SPLITS, OUT = sys.argv[1], sys.argv[2]
RAW = {"val": f"{D}/raw_arnold_dense/arenas", "arenas_678": f"{D}/raw_arnold_dense/arenas_678",
       "seen": f"{D}/raw_arnold_eval/seen", "unseen": f"{D}/raw_arnold_eval/unseen", "unseen2": f"{D}/raw_arnold_eval/unseen2"}
os.makedirs(OUT, exist_ok=True)
def frames_of(path, start, n, step):
    t = pq.read_table(path, columns=["frame"])
    rows = range(start, min(start + n, t.num_rows), step)
    return [np.asarray(Image.open(io.BytesIO(t["frame"][i].as_py())).convert("RGB")) for i in rows]
tiles = []
for f in sorted(glob.glob(f"{SPLITS}/split_*_map*.json")):
    sp = json.load(open(f)); meta = sp["meta"]; s, m = meta["set"], int(meta["map"])
    eps = sp.get("val") or next(v for k, v in sp.items() if isinstance(v, list))
    ep = int(eps[0]); path = f"{RAW[s]}/ep_{ep:05d}.parquet"
    fr = frames_of(path, 300, 256, 2)
    if not fr:
        print("no frames", s, m); continue
    ims = [Image.fromarray(x).resize((160, 120), Image.BILINEAR).convert("P", palette=Image.ADAPTIVE, colors=128) for x in fr]
    out = f"{OUT}/map{m:02d}_{s}_ep{ep:05d}.gif"
    ims[0].save(out, save_all=True, append_images=ims[1:], duration=57, loop=0, optimize=True)
    tiles.append((m, s, Image.fromarray(fr[len(fr)//2]).resize((160, 120), Image.BILINEAR)))
    print("wrote", out, len(ims), "frames", flush=True)
tiles.sort(key=lambda t: t[0])
cols = 6; rows = (len(tiles) + cols - 1) // cols
sheet = Image.new("RGB", (cols * 164, rows * 138), "white")
from PIL import ImageDraw
d = ImageDraw.Draw(sheet)
for i, (m, s, im) in enumerate(tiles):
    x, y = (i % cols) * 164, (i // cols) * 138
    sheet.paste(im, (x + 2, y + 16)); d.text((x + 4, y + 2), f"map {m} ({s})", fill="black")
sheet.save(f"{OUT}/contact_sheet.png"); print("wrote contact sheet", len(tiles), "maps")
