// Canvas board renderer with pan/zoom and hit-testing.
import { cellX, cellY, cellVerts, edgeMid, cellCenter, cellId, isUp, H, oppVert, edgePos } from './geometry.js';
import { makeCellShape, PLAYER_COLORS, KINDS } from './engine.js';
import { ART, TILE_ART_BY_TYPE, hillSprite, onArtLoad } from './art.js';

export const KIND_ICON = ['🌸', '🍄', '🐛'];

function hash(n) {
  let x = (n * 2654435761) >>> 0;
  x ^= x >>> 13; x = Math.imul(x, 0x5bd1e995) >>> 0; x ^= x >>> 15;
  return (x >>> 0) / 4294967296;
}

const GREENS = [[96, 138, 84], [110, 150, 92], [88, 126, 96], [124, 150, 84], [104, 140, 110], [138, 150, 92]];
export function tileFill(id) {
  const g = GREENS[Math.floor(hash(id) * GREENS.length)];
  const j = (hash(id + 7) - 0.5) * 18;
  return `rgb(${g[0] + j | 0},${g[1] + j | 0},${g[2] + j | 0})`;
}

const hasArt = (tile) => !!(ART.tiles.ok && TILE_ART_BY_TYPE[tile]);
// Paint the original printed triangle for (tile, rot) onto cell (x, y); false if there is none
// (type not in the print, atlas not loaded). `seed` picks one of the paintings of that type.
function drawTileArt(ctx, x, y, tile, rot, seed, tx, ty) {
  if (!hasArt(tile)) return false;
  const list = TILE_ART_BY_TYPE[tile];
  const s = list[Math.floor(hash(seed) * list.length)].v;
  const d = [];
  for (let e = 0; e < 3; e++) { const p = oppVert(x, y, e); d[(edgePos(x, y, e) + rot) % 3] = [tx(p[0]), ty(p[1])]; }
  // affine atlas -> screen through the three corners
  const sx1 = s[1][0] - s[0][0], sy1 = s[1][1] - s[0][1], sx2 = s[2][0] - s[0][0], sy2 = s[2][1] - s[0][1];
  const dx1 = d[1][0] - d[0][0], dy1 = d[1][1] - d[0][1], dx2 = d[2][0] - d[0][0], dy2 = d[2][1] - d[0][1];
  const det = sx1 * sy2 - sx2 * sy1;
  const A = (dx1 * sy2 - dx2 * sy1) / det, C = (dx2 * sx1 - dx1 * sx2) / det;
  const B = (dy1 * sy2 - dy2 * sy1) / det, D = (dy2 * sx1 - dy1 * sx2) / det;
  const cx = (d[0][0] + d[1][0] + d[2][0]) / 3, cy = (d[0][1] + d[1][1] + d[2][1]) / 3;
  const grow = 1 + 0.8 / Math.max(1, Math.hypot(d[0][0] - cx, d[0][1] - cy)); // ~1px overlap hides seams
  ctx.save();
  ctx.beginPath();
  d.forEach(([px, py], i) => { const X = cx + (px - cx) * grow, Y = cy + (py - cy) * grow; if (i) ctx.lineTo(X, Y); else ctx.moveTo(X, Y); });
  ctx.closePath(); ctx.clip();
  ctx.transform(A, B, C, D, d[0][0] - A * s[0][0] - C * s[0][1], d[0][1] - B * s[0][0] - D * s[0][1]);
  const im = ART.tiles;
  const bx = Math.max(0, Math.min(s[0][0], s[1][0], s[2][0]) - 3), by = Math.max(0, Math.min(s[0][1], s[1][1], s[2][1]) - 3);
  const bw = Math.min(im.width - bx, Math.max(s[0][0], s[1][0], s[2][0]) + 3 - bx), bh = Math.min(im.height - by, Math.max(s[0][1], s[1][1], s[2][1]) + 3 - by);
  ctx.drawImage(im, bx, by, bw, bh, bx, by, bw, bh);
  ctx.restore();
  return true;
}

function triPath(ctx, v, tx, ty, sc) {
  ctx.beginPath();
  ctx.moveTo(tx(v[0][0]), ty(v[0][1]));
  ctx.lineTo(tx(v[1][0]), ty(v[1][1]));
  ctx.lineTo(tx(v[2][0]), ty(v[2][1]));
  ctx.closePath();
}

// Draw path network of a cell shape. cell: {x,y,open,comps,allOpen}, junction: draw all open edges to hub
export function drawPaths(ctx, cell, junction, tx, ty, sc, color = '#fbfaf4', width = 0.085) {
  const [cx, cy] = cellCenter(cell.x, cell.y);
  const C = [tx(cx), ty(cy)];
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  const comps = junction ? [cell.allOpen] : cell.comps;
  const base = Math.max(2, width * sc);
  for (const pass of [0, 1]) {
    ctx.strokeStyle = pass === 0 ? 'rgba(28,46,22,.28)' : color;
    ctx.lineWidth = pass === 0 ? base + Math.max(1.5, sc * 0.03) : base;
    for (const comp of comps) {
      const pts = comp.map((e) => { const m = edgeMid(cell.x, cell.y, e); return [tx(m[0]), ty(m[1])]; });
      ctx.beginPath();
      if (pts.length === 1) {
        ctx.moveTo(pts[0][0], pts[0][1]);
        ctx.lineTo(pts[0][0] + (C[0] - pts[0][0]) * 0.55, pts[0][1] + (C[1] - pts[0][1]) * 0.55);
      } else if (pts.length === 2) {
        ctx.moveTo(pts[0][0], pts[0][1]); ctx.quadraticCurveTo(C[0], C[1], pts[1][0], pts[1][1]);
      } else for (const p of pts) { ctx.moveTo(p[0], p[1]); ctx.lineTo(C[0], C[1]); }
      ctx.stroke();
    }
  }
}

// A point on path piece `comp` (list of edges) of cell c: on a stub, on a curve's apex, at the hub of a Y
export function compSpot(c, comp, stubF = 0.4) {
  const [cx, cy] = cellCenter(c.x, c.y);
  const mids = comp.map((e) => edgeMid(c.x, c.y, e));
  const mx = mids.reduce((q, m) => q + m[0], 0) / mids.length, my = mids.reduce((q, m) => q + m[1], 0) / mids.length;
  const f = comp.length === 1 ? stubF : 0.5;
  return [mx + (cx - mx) * f, my + (cy - my) * f];
}

export class BoardView {
  constructor(canvas) {
    this.cv = canvas;
    this.ctx = canvas.getContext('2d');
    this.state = null;
    this.scale = 72; this.ox = 0; this.oy = 0;
    this.cellColors = new Map(); // cellId -> css color (highlight)
    this.ghost = null; // {cell,tile,rot,piece,ok}
    this.edgeMarks = []; // [{a,b,color}] bridge candidates
    this.hover = -1;
    this.onClick = null; this.onHover = null; this.onWheelRotate = null;
    this._drag = null;
    this.focusSeat = -1;
    this.autoFit = true;
    this.insetBottom = 0; this.insetTop = 0; this.cam = null; this.fx = []; this._raf = 0;
    this.cardView = false;
    this._bind();
    onArtLoad(() => { if (this.cv.isConnected) this.draw(); });
    this._ro = new ResizeObserver(() => this.resize());
    this._ro.observe(canvas.parentElement || canvas);
    this.resize();
  }
  resize() {
    const r = (this.cv.parentElement || this.cv).getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.w = Math.max(200, r.width); this.h = Math.max(200, r.height);
    this.cv.width = this.w * dpr; this.cv.height = this.h * dpr;
    this.cv.style.width = this.w + 'px'; this.cv.style.height = this.h + 'px';
    this.dpr = dpr;
    if (this.autoFit && this.state) this.fit(); else this.draw();
  }
  setState(s, { refit = false } = {}) {
    const first = !this.state;
    this.state = s;
    if (first || refit || this.autoFit) this.fit();
    else this.draw();
  }
  fit() {
    const s = this.state;
    if (!s) return;
    let minx = 1e9, maxx = -1e9, miny = 1e9, maxy = -1e9;
    for (const k in s.cells) {
      for (const v of cellVerts(s.cells[k].x, s.cells[k].y)) {
        minx = Math.min(minx, v[0]); maxx = Math.max(maxx, v[0]); miny = Math.min(miny, v[1]); maxy = Math.max(maxy, v[1]);
      }
    }
    minx -= 2; maxx += 2; miny -= 2; maxy += 2; // room for ghosts
    const vh = Math.max(120, this.h - this.insetBottom - this.insetTop);
    const sc = Math.min(this.w / (maxx - minx), vh / (maxy - miny), 90);
    this.scale = Math.max(34, sc);
    this.ox = this.w / 2 - ((minx + maxx) / 2) * this.scale;
    this.oy = this.insetTop + vh / 2 - ((miny + maxy) / 2) * this.scale;
    this.draw();
  }
  // painted fill for tile types that are not in print; moves and zooms with the board
  texture() {
    if (!ART.texture.ok) return null;
    this._pat = this._pat || this.ctx.createPattern(ART.texture, 'repeat');
    const k = (this.scale * 3) / ART.texture.width;
    this._pat.setTransform(new DOMMatrix([k, 0, 0, k, this.ox, this.oy]));
    return this._pat;
  }
  tx(x) { return this.ox + x * this.scale; }
  ty(y) { return this.oy + y * this.scale; }
  pick(px, py) {
    // find lattice cell containing screen point
    const ux = (px - this.ox) / this.scale, uy = (py - this.oy) / this.scale;
    const y = Math.floor(uy / H);
    for (let yy = y - 1; yy <= y + 1; yy++) {
      for (let xx = Math.floor(ux * 2) - 3; xx <= Math.floor(ux * 2) + 3; xx++) {
        const v = cellVerts(xx, yy);
        if (inTri(ux, uy, v)) return cellId(xx, yy);
      }
    }
    return -1;
  }
  _bind() {
    const cv = this.cv;
    this._ptrs = new Map();
    const zoomAt = (px, py, ns) => {
      ns = Math.max(20, Math.min(240, ns));
      const k = ns / this.scale;
      this.ox = px - (px - this.ox) * k; this.oy = py - (py - this.oy) * k; this.scale = ns;
    };
    cv.addEventListener('pointerdown', (e) => {
      this._ptrs.set(e.pointerId, [e.clientX, e.clientY]);
      cv.setPointerCapture(e.pointerId);
      if (this._ptrs.size === 2) {
        const [a, b] = [...this._ptrs.values()];
        this._pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), scale: this.scale };
        this._drag = null; this.autoFit = false;
      } else this._drag = { x: e.clientX, y: e.clientY, ox: this.ox, oy: this.oy, moved: false, type: e.pointerType };
    });
    cv.addEventListener('pointermove', (e) => {
      const r = cv.getBoundingClientRect();
      if (this._ptrs.has(e.pointerId)) this._ptrs.set(e.pointerId, [e.clientX, e.clientY]);
      if (this._pinch && this._ptrs.size === 2) {
        const [a, b] = [...this._ptrs.values()];
        const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
        zoomAt((a[0] + b[0]) / 2 - r.left, (a[1] + b[1]) / 2 - r.top, this._pinch.scale * d / this._pinch.d);
        this.draw(); return;
      }
      if (this._drag) {
        const dx = e.clientX - this._drag.x, dy = e.clientY - this._drag.y;
        if (Math.abs(dx) + Math.abs(dy) > 6) { this._drag.moved = true; this.autoFit = false; }
        if (this._drag.moved) { this.ox = this._drag.ox + dx; this.oy = this._drag.oy + dy; this.draw(); return; }
      }
      if (e.pointerType === 'touch') return;
      const c = this.pick(e.clientX - r.left, e.clientY - r.top);
      if (c !== this.hover) { this.hover = c; if (this.onHover) this.onHover(c); this.draw(); }
    });
    const up = (e) => {
      this._ptrs.delete(e.pointerId);
      if (this._pinch) { if (this._ptrs.size < 2) this._pinch = null; this._drag = null; return; }
      const d = this._drag; this._drag = null;
      if (d && !d.moved && this.onClick && e.type === 'pointerup') {
        const r = cv.getBoundingClientRect();
        this.onClick(this.pick(e.clientX - r.left, e.clientY - r.top), e);
      }
    };
    cv.addEventListener('pointerup', up);
    cv.addEventListener('pointercancel', up);
    cv.addEventListener('pointerleave', (e) => { if (e.pointerType === 'touch') return; this.hover = -1; if (this.onHover) this.onHover(-1); this.draw(); });
    cv.addEventListener('wheel', (e) => {
      e.preventDefault();
      const r = cv.getBoundingClientRect();
      this.autoFit = false;
      zoomAt(e.clientX - r.left, e.clientY - r.top, this.scale * Math.exp(-e.deltaY * 0.0015));
      this.draw();
    }, { passive: false });
    cv.addEventListener('contextmenu', (e) => { e.preventDefault(); if (this.onWheelRotate) this.onWheelRotate(1); });
  }


  // ---------- camera ----------
  bboxOf(cells, pad = 1.2) {
    let minx = 1e9, maxx = -1e9, miny = 1e9, maxy = -1e9;
    for (const c of cells) for (const v of cellVerts(cellX(c), cellY(c))) {
      minx = Math.min(minx, v[0]); maxx = Math.max(maxx, v[0]); miny = Math.min(miny, v[1]); maxy = Math.max(maxy, v[1]);
    }
    return { minx: minx - pad, maxx: maxx + pad, miny: miny - pad * 0.866, maxy: maxy + pad * 0.866 };
  }
  allCells() { return this.state ? Object.keys(this.state.cells).map(Number) : []; }
  // smooth camera move so that `cells` fill the view (clamped zoom)
  flyTo(cells, { ms = 650, minScale = 44, maxScale = 120, pad = 1.2 } = {}) {
    if (!cells.length) cells = this.allCells();
    if (!cells.length) return;
    const b = this.bboxOf(cells, pad);
    const vh = Math.max(120, this.h - this.insetBottom - this.insetTop);
    const scale = Math.max(minScale, Math.min(maxScale, this.w / (b.maxx - b.minx), vh / (b.maxy - b.miny)));
    const ox = this.w / 2 - ((b.minx + b.maxx) / 2) * scale;
    const oy = this.insetTop + vh / 2 - ((b.miny + b.maxy) / 2) * scale;
    this.autoFit = false;
    this.cam = { t0: performance.now(), ms, from: [this.ox, this.oy, this.scale], to: [ox, oy, scale] };
    this.kick();
  }
  flyToAll(opts = {}) { this.flyTo(this.allCells(), { maxScale: 90, minScale: 30, pad: 1.4, ...opts }); }

  // ---------- dice ----------
  rollDice({ cell, values, card, ms = 1300, hold = 2200 }) {
    this.fx.push({ kind: 'dice', cell, values, card, t0: performance.now(), ms, hold, tilt: [Math.random() * 0.5 - 0.25, Math.random() * 0.5 - 0.25] });
    this.kick();
  }
  kick() {
    if (this._raf) return;
    const loop = () => {
      this._raf = 0;
      const now = performance.now();
      if (this.cam) {
        const k = Math.min(1, (now - this.cam.t0) / this.cam.ms);
        const e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
        const f = this.cam.from, t = this.cam.to;
        // zoom about the interpolated centre for a smooth "dolly" feel
        this.scale = f[2] + (t[2] - f[2]) * e;
        this.ox = f[0] + (t[0] - f[0]) * e; this.oy = f[1] + (t[1] - f[1]) * e;
        if (k >= 1) this.cam = null;
      }
      this.fx = this.fx.filter((x) => now - x.t0 < x.ms + x.hold + 400);
      this.draw();
      if (this.cam || this.fx.length) this._raf = requestAnimationFrame(loop);
    };
    this._raf = requestAnimationFrame(loop);
  }
  drawFx(ctx) {
    const now = performance.now();
    for (const f of this.fx) {
      if (f.kind !== 'dice') continue;
      const age = now - f.t0;
      const t = Math.min(1, age / f.ms);
      const fade = age > f.ms + f.hold ? 1 - (age - f.ms - f.hold) / 400 : 1;
      const c = this.state.cells[f.cell];
      if (!c) continue;
      const [cx, cy] = cellCenter(c.x, c.y);
      const sz = Math.max(34, this.scale * 0.55);
      const dice = [f.values[0], f.values[1]];
      for (let i = 0; i < 2; i++) {
        const active = dice[i] != null;
        const rest = [this.tx(cx) + (i ? 1 : -1) * sz * 0.85, this.ty(cy) - sz * 1.15 + (i ? 0.25 : 0) * sz];
        const startOff = [(i ? 1 : -1) * sz * 5 + sz * 4, -sz * 3.5];
        const e = 1 - Math.pow(1 - t, 3);
        const bounce = active ? -Math.abs(Math.sin(t * Math.PI * 3.2)) * (1 - t) * sz * 0.9 : 0;
        const x = rest[0] + startOff[0] * (1 - e), y = rest[1] + startOff[1] * (1 - e) + bounce;
        const ang = active ? (1 - e) * (i ? -9 : 9) + f.tilt[i] : 0;
        const val = !active ? 0 : t < 0.78 ? Math.floor(now / 75 + i * 3) % 6 : dice[i];
        ctx.save(); ctx.globalAlpha = Math.max(0, fade) * (active ? 1 : 0.35);
        drawDie(ctx, x, y, sz, ang, val, active);
        ctx.restore();
      }
    }
  }

  draw() {
    const { ctx, w, h, dpr } = this;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const bg = ctx.createRadialGradient(w / 2, h / 2, 20, w / 2, h / 2, Math.max(w, h) * 0.75);
    bg.addColorStop(0, '#efe8d3'); bg.addColorStop(1, '#d9d0b5');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);
    const s = this.state;
    if (!s) return;
    const tx = (x) => this.tx(x), ty = (y) => this.ty(y), sc = this.scale;
    // frontier hints
    for (const [cid, col] of this.cellColors) {
      const v = cellVerts(cellX(cid), cellY(cid));
      triPath(ctx, v, tx, ty, sc);
      ctx.fillStyle = col; ctx.fill();
      ctx.strokeStyle = 'rgba(40,40,40,.35)'; ctx.setLineDash([4, 4]); ctx.lineWidth = 1; ctx.stroke(); ctx.setLineDash([]);
    }
    // tiles
    for (const k in s.cells) {
      const c = s.cells[k];
      const v = cellVerts(c.x, c.y);
      const art = drawTileArt(ctx, c.x, c.y, c.tile, c.rot, c.id + (c.overlaid ? 999 : 0), tx, ty);
      triPath(ctx, v, tx, ty, sc);
      if (!art) {
        const g = ctx.createLinearGradient(tx(v[0][0]), ty(v[0][1]), tx(v[1][0] + v[2][0] - v[0][0]), ty(v[1][1] + v[2][1] - v[0][1]));
        g.addColorStop(0, tileFill(c.id + (c.overlaid ? 999 : 0))); g.addColorStop(1, tileFill(c.id * 7 + 3));
        const pat = this.texture();
        if (pat) { ctx.fillStyle = pat; ctx.fill(); ctx.globalAlpha = 0.4; }
        ctx.fillStyle = g; ctx.fill(); ctx.globalAlpha = 1;
      }
      ctx.strokeStyle = c.overlaid ? '#d8c27a' : art ? 'rgba(60,50,20,.35)' : 'rgba(255,255,255,.5)'; ctx.lineWidth = c.overlaid ? 2.2 : 1; ctx.stroke();
      if (this.cellColors.has(c.id)) { triPath(ctx, v, tx, ty, sc); ctx.fillStyle = this.cellColors.get(c.id); ctx.fill(); }
      // painted tiles carry their paths; each printed type is one connected path, so a hill adds no junction
      if (!art) drawPaths(ctx, c, c.hill >= 0, tx, ty, sc);
    }
    // bridges
    for (const b of s.bridges) {
      const A = s.cells[b.a], B = s.cells[b.b];
      const ca = cellCenter(A.x, A.y), cb = cellCenter(B.x, B.y);
      ctx.strokeStyle = PLAYER_COLORS[b.owner]; ctx.lineWidth = Math.max(4, sc * 0.12); ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(tx(ca[0]), ty(ca[1])); ctx.lineTo(tx(cb[0]), ty(cb[1])); ctx.stroke();
      ctx.strokeStyle = 'rgba(255,255,255,.55)'; ctx.lineWidth = Math.max(1.5, sc * 0.04);
      ctx.beginPath(); ctx.moveTo(tx(ca[0]), ty(ca[1])); ctx.lineTo(tx(cb[0]), ty(cb[1])); ctx.stroke();
    }
    // edge marks (bridge candidates)
    for (const m of this.edgeMarks) {
      const A = s.cells[m.a], B = s.cells[m.b];
      const ca = cellCenter(A.x, A.y), cb = cellCenter(B.x, B.y);
      ctx.strokeStyle = m.color || 'rgba(255,230,80,.9)'; ctx.lineWidth = Math.max(3, sc * 0.09); ctx.setLineDash([6, 5]);
      ctx.beginPath(); ctx.moveTo(tx(ca[0]), ty(ca[1])); ctx.lineTo(tx(cb[0]), ty(cb[1])); ctx.stroke(); ctx.setLineDash([]);
    }
    // pieces: opaque with a drop shadow (light from the top left); on drawn tiles the path network is re-drawn faintly on top
    const occupied = [];
    for (const k in s.cells) {
      const c = s.cells[k];
      const [cx, cy] = cellCenter(c.x, c.y);
      const X = tx(cx), Y = ty(cy);
      const r = sc * 0.145;
      const hillImg = c.hill >= 0 && hillSprite(PLAYER_COLORS[c.hill]);
      if (hillImg) {
        const hw = r * 2.7, hh = hw * hillImg.height / hillImg.width;
        shadowed(ctx, sc, () => ctx.drawImage(hillImg, X - hw / 2, Y + r * 0.5 - hh, hw, hh));
      } else if (c.hill >= 0) {
        pieceShadow(ctx, sc);
        const hr = r * 1.3, hy = Y + r * 0.45;
        ctx.fillStyle = PLAYER_COLORS[c.hill];
        ctx.beginPath(); ctx.arc(X, hy, hr, Math.PI, 0); ctx.closePath(); ctx.fill();
        ctx.strokeStyle = 'rgba(0,0,0,.45)'; ctx.lineWidth = 1.2; ctx.stroke();
        ctx.fillStyle = 'rgba(0,0,0,.55)'; ctx.beginPath(); ctx.ellipse(X, hy, hr * 0.32, hr * 0.22, 0, Math.PI, 0); ctx.fill();
      }
      if (c.res >= 0) {
        const res = s.resources[c.res];
        if (c.hill >= 0) drawResource(ctx, X + r * 1.35, Y - r * 0.75, r, res.kind);
        else { // on its path piece
          const [rx, ry] = compSpot(c, c.comps.find((g) => g.includes(c.re)) || c.comps[0]);
          drawResource(ctx, tx(rx), ty(ry), r, res.kind);
        }
      }
      if (c.block >= 0) {
        pieceShadow(ctx, sc);
        ctx.fillStyle = PLAYER_COLORS[c.block]; ctx.strokeStyle = '#000'; ctx.lineWidth = 1.5;
        const q = r * 1.6;
        ctx.fillRect(X - q / 2, Y - q / 2, q, q); ctx.strokeRect(X - q / 2, Y - q / 2, q, q);
        ctx.beginPath(); ctx.moveTo(X - q / 2, Y - q / 2); ctx.lineTo(X + q / 2, Y + q / 2); ctx.moveTo(X + q / 2, Y - q / 2); ctx.lineTo(X - q / 2, Y + q / 2); ctx.stroke();
      }
      noShadow(ctx);
      if (c.hill >= 0 || c.res >= 0 || c.block >= 0) occupied.push(c);
    }
    for (const c of occupied) { if (hasArt(c.tile)) continue; ctx.globalAlpha = 0.5; drawPaths(ctx, c, c.hill >= 0, tx, ty, sc, '#ffffff', 0.04); ctx.globalAlpha = 1; }
    // ants: placed on the path group (component) they actually stand in
    const byCell = {};
    for (const p of s.players) if (p.ant) (byCell[p.ant.cell] = byCell[p.ant.cell] || []).push(p);
    for (const k in byCell) {
      const list = byCell[k];
      const c = s.cells[k];
      const junction = c.hill >= 0;
      const comps = junction ? [c.allOpen] : c.comps;
      const [cx, cy] = cellCenter(c.x, c.y);
      list.forEach((p, i) => {
        let ax = cx, ay = cy;
        const comp = comps.find((g) => g.includes(p.ant.edge)) || comps[0];
        if (comp && !junction) [ax, ay] = compSpot(c, comp, 0.5);
        const withRes = c.res >= 0 && comp && comp.includes(c.re); // shares its path piece with a resource
        const crowd = junction || withRes ? [-0.17, 0.12] : [0, 0];
        const off = (i - (list.length - 1) / 2) * sc * 0.22;
        const X = tx(ax) + crowd[0] * sc + off, Y = ty(ay) + crowd[1] * sc;
        const r = sc * 0.125;
        token(ctx, X, Y, r * 1.15, '#fff', p.color, Math.max(2, sc * 0.045));
        if (ART.ant.ok) { const aw = r * 1.8, ah = aw * ART.ant.height / ART.ant.width; ctx.drawImage(ART.ant, X - aw / 2, Y - ah / 2, aw, ah); }
        else { ctx.font = `${Math.round(r * 1.7)}px serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = '#000'; ctx.fillText('🐜', X, Y + 1); }
        if (p.carry >= 0) { drawResource(ctx, X + r * 1.15, Y - r * 1.15, r * 0.78, s.resources[p.carry].kind); ctx.strokeStyle = p.color; ctx.lineWidth = Math.max(2, sc * 0.035); ctx.beginPath(); ctx.arc(X + r * 1.15, Y - r * 1.15, r * 0.9, 0, 7); ctx.stroke(); }
      });
    }
    // ghost
    if (this.ghost) {
      const g = this.ghost;
      const sh = makeCellShape(cellX(g.cell), cellY(g.cell), g.tile, g.rot);
      const cell = { x: cellX(g.cell), y: cellY(g.cell), ...sh };
      const v = cellVerts(cell.x, cell.y);
      ctx.globalAlpha = 0.85;
      if (g.ok === false || !drawTileArt(ctx, cell.x, cell.y, g.tile, g.rot, g.cell, tx, ty)) {
        triPath(ctx, v, tx, ty, sc);
        ctx.fillStyle = g.ok === false ? '#b55' : '#9bbf86'; ctx.fill();
        drawPaths(ctx, cell, g.piece === 1, tx, ty, sc);
      }
      ctx.globalAlpha = 1;
      ctx.strokeStyle = g.ok === false ? '#c22' : '#fff'; ctx.lineWidth = 2.5; triPath(ctx, v, tx, ty, sc); ctx.stroke();
      const [cx, cy] = cellCenter(cell.x, cell.y);
      if (g.piece === 1) {
        ctx.fillStyle = PLAYER_COLORS[g.seat ?? 0]; ctx.beginPath(); ctx.arc(tx(cx), ty(cy) + sc * 0.07, sc * 0.25, Math.PI, 0); ctx.fill();
      } else if (g.piece === 2) {
        const [rx, ry] = compSpot(cell, cell.comps[g.comp ?? 0] || cell.comps[0]);
        ctx.font = `${Math.round(sc * 0.3)}px serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('❔', tx(rx), ty(ry));
      }
    } else if (this.hover >= 0 && this.cellColors.has(this.hover)) {
      const v = cellVerts(cellX(this.hover), cellY(this.hover));
      triPath(ctx, v, tx, ty, sc); ctx.strokeStyle = '#fff'; ctx.lineWidth = 3; ctx.stroke();
    }
    this.drawFx(ctx);
  }
}

// A rounded die with pips (faces 0..5; 0 is drawn as an empty ring)
function drawDie(ctx, x, y, size, ang, v, active) {
  ctx.translate(x, y); ctx.rotate(ang);
  const r = size * 0.18, h = size / 2;
  ctx.shadowColor = 'rgba(0,0,0,.4)'; ctx.shadowBlur = size * 0.25; ctx.shadowOffsetY = size * 0.12;
  ctx.beginPath();
  ctx.moveTo(-h + r, -h); ctx.arcTo(h, -h, h, h, r); ctx.arcTo(h, h, -h, h, r); ctx.arcTo(-h, h, -h, -h, r); ctx.arcTo(-h, -h, h, -h, r); ctx.closePath();
  const g = ctx.createLinearGradient(-h, -h, h, h);
  g.addColorStop(0, '#fffdf4'); g.addColorStop(1, active ? '#e9dfc2' : '#d8d2c0');
  ctx.fillStyle = g; ctx.fill();
  ctx.shadowColor = 'transparent';
  ctx.strokeStyle = active ? '#6b5e3a' : '#8a8366'; ctx.lineWidth = Math.max(1.5, size * 0.04);
  if (!active) ctx.setLineDash([size * 0.12, size * 0.1]);
  ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = '#2a2417';
  const q = size * 0.24, pr = size * 0.085;
  const P = { 1: [[0, 0]], 2: [[-q, -q], [q, q]], 3: [[-q, -q], [0, 0], [q, q]], 4: [[-q, -q], [q, -q], [-q, q], [q, q]], 5: [[-q, -q], [q, -q], [0, 0], [-q, q], [q, q]] };
  if (!active) { ctx.font = `${Math.round(size * 0.42)}px serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('🃏', 0, 1); }
  else if (v === 0) { ctx.beginPath(); ctx.arc(0, 0, size * 0.17, 0, 7); ctx.lineWidth = size * 0.07; ctx.strokeStyle = '#8b1e1e'; ctx.stroke(); }
  else for (const [px, py] of P[v]) { ctx.beginPath(); ctx.arc(px, py, pr, 0, 7); ctx.fill(); }
  ctx.rotate(-ang); ctx.translate(-x, -y);
}

function inTri(px, py, v) {
  const [a, b, c] = v;
  const d = (p1, p2, p3) => (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]);
  const p = [px, py];
  const d1 = d(p, a, b), d2 = d(p, b, c), d3 = d(p, c, a);
  const neg = d1 < 0 || d2 < 0 || d3 < 0, pos = d1 > 0 || d2 > 0 || d3 > 0;
  return !(neg && pos);
}

// Drop shadow for pieces standing on the board, light from the top left: a wide soft one, or (contact) a tight dark one
// at the base. Canvas shadows ignore the transform, so they are scaled by the device pixel ratio.
function pieceShadow(ctx, sc, contact = false) {
  const d = ctx.getTransform().a;
  ctx.shadowColor = contact ? 'rgba(15,10,0,.6)' : 'rgba(15,10,0,.55)';
  ctx.shadowBlur = Math.max(contact ? 1.5 : 3, sc * (contact ? 0.025 : 0.09)) * d;
  ctx.shadowOffsetX = Math.max(1, sc * (contact ? 0.012 : 0.035)) * d; ctx.shadowOffsetY = Math.max(1, sc * (contact ? 0.022 : 0.065)) * d;
}
function noShadow(ctx) { ctx.shadowColor = 'transparent'; ctx.shadowBlur = 0; ctx.shadowOffsetX = 0; ctx.shadowOffsetY = 0; }
// draw() twice: with the soft and with the contact shadow
function shadowed(ctx, sc, draw) {
  pieceShadow(ctx, sc); draw();
  pieceShadow(ctx, sc, true); draw();
  noShadow(ctx);
}

// A round token: casts both shadows, then gets a convex shading (bright top left, darker rim) and its outline.
// Leaves the shadow off, so what is drawn on the token does not cast a second one.
function token(ctx, X, Y, R, fill, stroke, lw) {
  const sc = R / 0.15; // shadow size follows the token size (board tokens are ~0.15 cell sides)
  ctx.fillStyle = fill; ctx.beginPath(); ctx.arc(X, Y, R, 0, 7);
  shadowed(ctx, sc, () => ctx.fill());
  const g = ctx.createRadialGradient(X - R * 0.35, Y - R * 0.4, R * 0.05, X, Y, R);
  g.addColorStop(0, 'rgba(255,255,255,.6)'); g.addColorStop(0.55, 'rgba(255,255,255,0)'); g.addColorStop(1, 'rgba(60,45,20,.28)');
  ctx.fillStyle = g; ctx.fill();
  ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke();
}

export function drawResource(ctx, X, Y, r, kind) {
  token(ctx, X, Y, r * 1.05, '#f4efe2', '#6b5e3a', 1.5);
  const im = ART.res[kind];
  if (im?.ok) { const f = r * 1.5 / Math.max(im.width, im.height); ctx.drawImage(im, X - im.width * f / 2, Y - im.height * f / 2, im.width * f, im.height * f); return; }
  ctx.font = `${Math.round(r * 1.5)}px serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = '#000';
  ctx.fillText(KIND_ICON[kind], X, Y + 1);
}

// small standalone tile preview (for hand display)
export function drawTilePreview(canvas, tile, rot, { color = '#8db37a', size = 54 } = {}) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = size * dpr; canvas.height = size * dpr * 0.9;
  canvas.style.width = size + 'px'; canvas.style.height = size * 0.9 + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, size, size);
  // always render as an upward triangle at cell (0,0)
  const sh = makeCellShape(0, 0, tile, rot);
  const cell = { x: 0, y: 0, ...sh };
  const sc = size * 0.92;
  const tx = (x) => 2 + x * sc, ty = (y) => 2 + y * sc;
  const v = cellVerts(0, 0);
  const art = drawTileArt(ctx, 0, 0, tile, rot, tile * 31 + 5, tx, ty);
  triPath(ctx, v, tx, ty, sc);
  if (!art) { ctx.fillStyle = color; ctx.fill(); }
  ctx.strokeStyle = art ? 'rgba(60,50,20,.45)' : 'rgba(255,255,255,.7)'; ctx.lineWidth = art ? 1 : 1.5; ctx.stroke();
  if (!art) drawPaths(ctx, cell, false, tx, ty, sc, '#f7f4e8', 0.085);
}
