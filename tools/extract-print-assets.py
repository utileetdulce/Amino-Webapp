#!/usr/bin/env python3
"""Cut the web assets out of the original print-and-play PDFs (github.com/asishallab/Amino/tree/master/print).

usage: python3 tools/extract-print-assets.py <folder with the print PDFs>
needs: pdftoppm (poppler), PyMuPDF, Pillow, numpy

Writes into assets/ (plus js/tile-art.js):
  tiles.webp      the 100 painted landscape triangles (Y, Kurve, Sackgasse) as one atlas
  res-0/1/2.png   resource tokens Blume, Pilz, Raupe (order of KINDS in js/engine.js)
  ant.png         ant silhouette, hill.png grey hill dome (tinted per player in the app)
  card-*.webp     ant card fronts (CARD_TYPES ids), card-back.webp
  colony-XYZ.webp Koloniekarte whose 1/2/3-point chambers hold Blume=X, Pilz=Y, Raupe=Z points
"""
import collections, io, json, math, os, subprocess, sys, tempfile
import numpy as np
import pymupdf
from PIL import Image, ImageDraw

SRC = sys.argv[1]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'assets')
os.makedirs(OUT, exist_ok=True)
TMP = tempfile.mkdtemp()


def render(pdf, dpi):
    pre = os.path.join(TMP, f'{os.path.basename(pdf)}-{dpi}')
    subprocess.run(['pdftoppm', '-r', str(dpi), '-png', os.path.join(SRC, pdf), pre], check=True)
    return [Image.open(os.path.join(TMP, f)).convert('RGB') for f in sorted(os.listdir(TMP)) if f.startswith(os.path.basename(pre) + '-')]


def pdf_images(pdf):
    doc = pymupdf.open(os.path.join(SRC, pdf))
    out = {}
    for page in doc:
        for info in page.get_images(full=True):
            pix = pymupdf.Pixmap(doc, info[0])
            if pix.n - pix.alpha >= 4: pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            out[info[0]] = Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB')
    return list(out.values())


def save_webp(im, name, q=80):
    im.save(os.path.join(OUT, name), 'WEBP', quality=q, method=6)


# ---------- landscape triangles ----------
# The sheets share one lattice (found from the crop marks at 200 dpi): first horizontal cut line y=77,
# row height 430 px, a vertex on that line at x=300. All four pages repeat the same path layout.
Y0, RH, X0 = 77.0, 430.0, 300.0
S = RH / (math.sqrt(3) / 2)
SIDE = 240  # triangle side in the atlas
NAMES = {(1, (1,)): 2, (2, (2,)): 1, (3, (3,)): 0}  # (open edges, path piece sizes) -> TILE_TYPES index (Sackgasse, Kurve, Y)


def classify(a, V):
    """open edges and path pieces of the triangle V (edge i lies opposite vertex i)"""
    hgt, wid, _ = a.shape
    white = (a.min(2) > 200) & (a.max(2) - a.min(2) < 40)
    C = (sum(v[0] for v in V) / 3, sum(v[1] for v in V) / 3)
    m = Image.new('L', (wid, hgt), 0)
    ImageDraw.Draw(m).polygon([(C[0] + (v[0] - C[0]) * 0.97, C[1] + (v[1] - C[1]) * 0.97) for v in V], fill=1)
    wm = white & np.asarray(m).astype(bool)
    seeds, opn = [], []
    for i in range(3):
        p, q = V[(i + 1) % 3], V[(i + 2) % 3]
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        x, y = int(mx + (C[0] - mx) * 0.12), int(my + (C[1] - my) * 0.12)
        seeds.append((x, y))
        opn.append(wm[y - 6:y + 7, x - 6:x + 7].mean() >= 0.5)
    lab = Image.fromarray((wm * 255).astype(np.uint8))
    comp = [-1] * 3
    for i in range(3):
        if not opn[i] or comp[i] >= 0: continue
        x, y = seeds[i]
        ys, xs = np.where(wm[y - 6:y + 7, x - 6:x + 7])
        ImageDraw.floodfill(lab, (int(x - 6 + xs[len(xs) // 2]), int(y - 6 + ys[len(ys) // 2])), 10 + i)
        L = np.asarray(lab)
        for j in range(3):
            xx, yy = seeds[j]
            if opn[j] and comp[j] < 0 and (L[yy - 6:yy + 7, xx - 6:xx + 7] == 10 + i).any(): comp[j] = i
    return [int(o) for o in opn], comp


def tiles():
    pages = render('03_Landschaftsdreiecke.pdf', 200)
    found = []
    for page in pages:
        a = np.asarray(page).astype(int)
        hgt, wid, _ = a.shape
        bg = np.abs(a - [230, 232, 232]).sum(2) < 14
        for r in range(6):
            yt, yb = Y0 + r * RH, Y0 + (r + 1) * RH
            for k in range(-3, 6):
                T = X0 + r * S / 2 + k * S
                for up in (True, False):
                    V = [(T, yt), (T + S / 2, yb), (T - S / 2, yb)] if up else [(T, yt), (T + S, yt), (T + S / 2, yb)]
                    C = (sum(v[0] for v in V) / 3, sum(v[1] for v in V) / 3)
                    inner = [(C[0] + (v[0] - C[0]) * 0.85, C[1] + (v[1] - C[1]) * 0.85) for v in V]
                    if any(not (0 <= p[0] < wid and 0 <= p[1] < hgt) or bg[int(p[1]), int(p[0])] for p in inner): continue
                    opn, comp = classify(a, V)
                    sizes = tuple(sorted(collections.Counter(comp[i] for i in range(3) if opn[i]).values(), reverse=True))
                    t = NAMES.get((sum(opn), sizes))
                    if t is None: print('skipped triangle with', opn, comp); continue
                    found.append((page, V, t, opn))
    f = SIDE / S
    cw, ch = int(S * 1.08 * f) + 2, int((S * math.sqrt(3) / 2 + S * 0.08) * f) + 2
    cols = 10
    atlas = Image.new('RGB', (cols * cw, math.ceil(len(found) / cols) * ch), (120, 140, 90))
    meta = []
    for n, (page, V, t, opn) in enumerate(found):
        pad = S * 0.04
        bx, by = min(v[0] for v in V) - pad, min(v[1] for v in V) - pad
        crop = page.crop((int(bx), int(by), int(max(v[0] for v in V) + pad), int(max(v[1] for v in V) + pad)))
        crop = crop.resize((round(crop.width * f), round(crop.height * f)), Image.LANCZOS)
        ax, ay = (n % cols) * cw, (n // cols) * ch
        atlas.paste(crop, (ax, ay))
        # template slot k (TILE_TYPES open/groups index) sits on picture edge (k + s) % 3; both run clockwise
        s = 0 if t == 0 else (opn.index(0) - 2) % 3 if t == 1 else opn.index(1)
        P = [(ax + (v[0] - int(bx)) * f, ay + (v[1] - int(by)) * f) for v in V]
        meta.append({'t': t, 'v': [[round(P[(k + s) % 3][0], 1), round(P[(k + s) % 3][1], 1)] for k in range(3)]})
    save_webp(atlas, 'tiles.webp', 78)
    with open(os.path.join(ROOT, 'js', 'tile-art.js'), 'w') as fh:
        fh.write('// Generated by tools/extract-print-assets.py: the painted triangles in assets/tiles.webp.\n')
        fh.write('// t = TILE_TYPES index; v[k] = atlas pixel of the corner opposite template edge k.\n')
        fh.write('export const TILE_ART = ' + json.dumps(meta, separators=(',', ':')) + ';\n')
    print('tiles:', collections.Counter(m['t'] for m in meta))


# ---------- pieces ----------
def cut_white(im, tol=28):
    """make the white surrounding of a picture transparent (flood fill from the border, so white inside stays)"""
    im = im.convert('RGBA')
    w, h = im.size
    big = Image.new('RGBA', (w + 2, h + 2), (255, 255, 255, 255))
    big.paste(im, (1, 1))
    ImageDraw.floodfill(big, (0, 0), (255, 0, 255, 0), thresh=tol)
    a = np.asarray(big).copy()
    hole = (a[..., 3] == 0)
    a[hole] = 0
    return Image.fromarray(a).crop((1, 1, w + 1, h + 1)).crop(Image.fromarray(a[1:-1, 1:-1, 3]).getbbox())


def fit(im, n):
    im = im.copy(); im.thumbnail((n, n), Image.LANCZOS); return im


def pieces():
    st = sorted(pdf_images('Rohstoffe_Wuerfel_Ameisen_10mm_Kreis_Klebeetiketten.pdf'), key=lambda i: i.size)
    by = {i.size: i for i in st}
    # stickers: 95x105 mushroom, 155x127 caterpillar, 158x158 flower, 184x140 ant
    fit(cut_white(by[(158, 158)]), 128).save(os.path.join(OUT, 'res-0.png'), optimize=True)
    fit(cut_white(by[(95, 105)]), 128).save(os.path.join(OUT, 'res-1.png'), optimize=True)
    fit(cut_white(by[(155, 127)]), 128).save(os.path.join(OUT, 'res-2.png'), optimize=True)
    ant = by[(184, 140)].convert('L')
    a = np.asarray(ant).astype(float)
    rgba = np.zeros(a.shape + (4,), np.uint8)
    rgba[..., 3] = np.clip((255 - a) * 1.15, 0, 255).astype(np.uint8)
    antim = Image.fromarray(rgba)
    fit(antim.crop(antim.getbbox()), 128).save(os.path.join(OUT, 'ant.png'), optimize=True)
    # grey hill dome (221x186); brightened so a multiply with the player colour keeps that colour
    dome = [i for i in pdf_images('04_pieces.pdf') if i.size == (221, 186)][0]
    d = cut_white(dome, 40)
    arr = np.asarray(d).astype(float)
    lum = arr[..., :3].mean(2)
    lo, hi = np.percentile(lum[arr[..., 3] > 0], [2, 98])
    l2 = np.clip(0.55 + 0.55 * (lum - lo) / (hi - lo), 0, 1.1) * 232
    out = np.dstack([np.clip(l2, 0, 255)] * 3 + [arr[..., 3]]).astype(np.uint8)
    fit(Image.fromarray(out), 128).save(os.path.join(OUT, 'hill.png'), optimize=True)


# ---------- cards ----------
def cards():
    # 150 dpi: 3x3 cards of 392x540 px starting at (35, 71); every row holds one card type
    fronts = render('01_Ant_cards.pdf', 150)
    order = [(0, 0, 'superimpose'), (0, 1, 'bridge'), (0, 2, 'blockage'), (1, 0, 'revenge'), (1, 1, 'transport'), (1, 2, 'wings')]
    for pg, row, name in order:
        c = fronts[pg].crop((35 + 5, 71 + row * 540 + 5, 35 + 392 - 5, 71 + (row + 1) * 540 - 5))
        if name == 'transport': c = c.rotate(180)  # printed upside down on the sheet
        save_webp(fit(c, 320), f'card-{name}.webp')
    back = render('01_Ant_cards_background.pdf', 150)[0]
    save_webp(fit(back.crop((40, 76, 422, 606)), 320), 'card-back.webp')
    # Koloniekarten: one per order of resource values; find the order from the chamber icons
    kinds = {'blue': 0, 'red': 1, 'green': 2}  # Blume, Pilz, Raupe
    for im in pdf_images('02_Bautenkarten.pdf'):
        a = np.asarray(im).astype(int)
        vals = [0, 0, 0]
        for pts, fy in ((1, 0.205), (2, 0.415), (3, 0.625)):
            y, x = int(fy * a.shape[0]), int(0.845 * a.shape[1])
            win = a[y - 40:y + 40, x - 40:x + 40].reshape(-1, 3)
            col = [p for p in win if not (p.min() > 200)]
            m = np.mean(col, 0)
            k = 'blue' if m[2] > m[0] and m[2] > m[1] else 'red' if m[0] > m[1] else 'green'
            vals[kinds[k]] = pts
        assert sorted(vals) == [1, 2, 3], vals
        save_webp(fit(im, 760), 'colony-%d%d%d.webp' % tuple(vals), 76)
        print('colony', vals)


# ---------- painted texture for the tile types that are not in print ----------
def texture():
    im = pdf_images('03_Landschaftsdreiecke_Hintergrund.pdf')[0]
    w, h = im.size
    c = im.crop((w // 4, h // 4, w // 4 + 1024, h // 4 + 1024)).resize((512, 512), Image.LANCZOS)
    a = np.asarray(c.convert('L')).astype(float) / 255
    a = 0.56 + 1.5 * (a - a.mean())  # strengthen the brush strokes, centre the brightness
    green = np.array([118, 156, 96]) / 255
    rgb = np.clip(a[..., None] * green * 1.6, 0, 1) * 255
    # make it tile seamlessly: per axis, blend with the copy shifted by half (its seam gets zero weight)
    w = (np.minimum(np.arange(512), 511 - np.arange(512)) / 255.5).clip(0, 1)
    out = rgb * w[None, :, None] + np.roll(rgb, 256, 1) * (1 - w[None, :, None])
    out = out * w[:, None, None] + np.roll(out, 256, 0) * (1 - w[:, None, None])
    save_webp(Image.fromarray(out.astype(np.uint8)), 'texture.webp', 75)


tiles()
pieces()
cards()
texture()
print('done ->', OUT)
