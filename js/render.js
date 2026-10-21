// Canvas board renderer with pan/zoom and hit-testing.
import { cellX, cellY, cellVerts, edgeMid, cellCenter, cellId, isUp, H } from './geometry.js';
import { makeCellShape, PLAYER_COLORS, KINDS } from './engine.js';

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

function triPath(ctx, v, tx, ty, sc) {
  ctx.beginPath();
  ctx.moveTo(tx(v[0][0]), ty(v[0][1]));
  ctx.lineTo(tx(v[1][0]), ty(v[1][1]));
  ctx.lineTo(tx(v[2][0]), ty(v[2][1]));
  ctx.closePath();
}

// Draw path network of a cell shape. cell: {x,y,open,comps,allOpen}, junction: draw all open edges to hub
export function drawPaths(ctx, cell, junction, tx, ty, sc, color = '#f6f2e2', width = 0.07) {
  const [cx, cy] = cellCenter(cell.x, cell.y);
  const C = [tx(cx), ty(cy)];
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  const comps = junction ? [cell.allOpen] : cell.comps;
  const base = Math.max(2, width * sc);
  for (const pass of [0, 1]) {
    ctx.strokeStyle = pass === 0 ? 'rgba(28,46,22,.38)' : color;
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
    this.cardView = false;
    this._bind();
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
    const sc = Math.min(this.w / (maxx - minx), this.h / (maxy - miny), 90);
    this.scale = Math.max(34, sc);
    this.ox = this.w / 2 - ((minx + maxx) / 2) * this.scale;
    this.oy = this.h / 2 - ((miny + maxy) / 2) * this.scale;
    this.draw();
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
      triPath(ctx, v, tx, ty, sc);
      const g = ctx.createLinearGradient(tx(v[0][0]), ty(v[0][1]), tx(v[1][0] + v[2][0] - v[0][0]), ty(v[1][1] + v[2][1] - v[0][1]));
      g.addColorStop(0, tileFill(c.id + (c.overlaid ? 999 : 0))); g.addColorStop(1, tileFill(c.id * 7 + 3));
      ctx.fillStyle = g; ctx.fill();
      ctx.strokeStyle = c.overlaid ? '#d8c27a' : 'rgba(255,255,255,.5)'; ctx.lineWidth = c.overlaid ? 2.2 : 1; ctx.stroke();
      if (this.cellColors.has(c.id)) { triPath(ctx, v, tx, ty, sc); ctx.fillStyle = this.cellColors.get(c.id); ctx.fill(); }
      drawPaths(ctx, c, c.hill >= 0 || c.res >= 0, tx, ty, sc);
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
    // pieces
    ctx.shadowColor = 'rgba(0,0,0,.35)'; ctx.shadowBlur = Math.max(2, sc * 0.06); ctx.shadowOffsetY = Math.max(1, sc * 0.025);
    for (const k in s.cells) {
      const c = s.cells[k];
      const [cx, cy] = cellCenter(c.x, c.y);
      const X = tx(cx), Y = ty(cy);
      const r = sc * 0.2;
      if (c.hill >= 0) {
        ctx.fillStyle = PLAYER_COLORS[c.hill];
        ctx.beginPath(); ctx.arc(X, Y + r * 0.35, r * 1.25, Math.PI, 0); ctx.closePath(); ctx.fill();
        ctx.strokeStyle = 'rgba(0,0,0,.45)'; ctx.lineWidth = 1.2; ctx.stroke();
        ctx.fillStyle = 'rgba(0,0,0,.55)'; ctx.beginPath(); ctx.ellipse(X, Y + r * 0.35, r * 0.4, r * 0.28, 0, Math.PI, 0); ctx.fill();
      }
      if (c.res >= 0) {
        const res = s.resources[c.res];
        drawResource(ctx, X + (c.hill >= 0 ? r * 1.0 : 0), Y - (c.hill >= 0 ? r * 0.2 : 0), r, res.kind);
      }
      if (c.block >= 0) {
        ctx.fillStyle = PLAYER_COLORS[c.block]; ctx.strokeStyle = '#000'; ctx.lineWidth = 1.5;
        const q = r * 1.7;
        ctx.fillRect(X - q / 2, Y - q / 2, q, q); ctx.strokeRect(X - q / 2, Y - q / 2, q, q);
        ctx.beginPath(); ctx.moveTo(X - q / 2, Y - q / 2); ctx.lineTo(X + q / 2, Y + q / 2); ctx.moveTo(X + q / 2, Y - q / 2); ctx.lineTo(X - q / 2, Y + q / 2); ctx.stroke();
      }
    }
    // ants
    const byCell = {};
    for (const p of s.players) if (p.ant) (byCell[p.ant.cell] = byCell[p.ant.cell] || []).push(p);
    for (const k in byCell) {
      const list = byCell[k];
      const c = s.cells[k];
      const [cx, cy] = cellCenter(c.x, c.y);
      list.forEach((p, i) => {
        const off = (i - (list.length - 1) / 2) * sc * 0.3;
        const X = tx(cx) + off, Y = ty(cy) + sc * 0.1;
        const r = sc * 0.17;
        ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(X, Y, r * 1.15, 0, 7); ctx.fill();
        ctx.lineWidth = Math.max(2, sc * 0.05); ctx.strokeStyle = p.color; ctx.stroke();
        ctx.font = `${Math.round(r * 1.7)}px serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = '#000';
        ctx.fillText('🐜', X, Y + 1);
        if (p.carry >= 0) drawResource(ctx, X + r * 1.1, Y - r * 1.1, r * 0.8, s.resources[p.carry].kind);
      });
    }
    ctx.shadowColor = 'transparent'; ctx.shadowBlur = 0; ctx.shadowOffsetY = 0;
    // ghost
    if (this.ghost) {
      const g = this.ghost;
      const sh = makeCellShape(cellX(g.cell), cellY(g.cell), g.tile, g.rot);
      const cell = { x: cellX(g.cell), y: cellY(g.cell), ...sh };
      const v = cellVerts(cell.x, cell.y);
      triPath(ctx, v, tx, ty, sc);
      ctx.globalAlpha = 0.85; ctx.fillStyle = g.ok === false ? '#b55' : '#9bbf86'; ctx.fill();
      drawPaths(ctx, cell, g.piece > 0, tx, ty, sc);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = g.ok === false ? '#c22' : '#fff'; ctx.lineWidth = 2.5; triPath(ctx, v, tx, ty, sc); ctx.stroke();
      const [cx, cy] = cellCenter(cell.x, cell.y);
      if (g.piece === 1) {
        ctx.fillStyle = PLAYER_COLORS[g.seat ?? 0]; ctx.beginPath(); ctx.arc(tx(cx), ty(cy) + sc * 0.07, sc * 0.25, Math.PI, 0); ctx.fill();
      } else if (g.piece === 2) {
        ctx.font = `${Math.round(sc * 0.3)}px serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('❔', tx(cx), ty(cy));
      }
    } else if (this.hover >= 0 && this.cellColors.has(this.hover)) {
      const v = cellVerts(cellX(this.hover), cellY(this.hover));
      triPath(ctx, v, tx, ty, sc); ctx.strokeStyle = '#fff'; ctx.lineWidth = 3; ctx.stroke();
    }
  }
}

function inTri(px, py, v) {
  const [a, b, c] = v;
  const d = (p1, p2, p3) => (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]);
  const p = [px, py];
  const d1 = d(p, a, b), d2 = d(p, b, c), d3 = d(p, c, a);
  const neg = d1 < 0 || d2 < 0 || d3 < 0, pos = d1 > 0 || d2 > 0 || d3 > 0;
  return !(neg && pos);
}

export function drawResource(ctx, X, Y, r, kind) {
  ctx.fillStyle = '#f4efe2'; ctx.strokeStyle = '#6b5e3a'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(X, Y, r * 1.05, 0, 7); ctx.fill(); ctx.stroke();
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
  triPath(ctx, v, tx, ty, sc);
  ctx.fillStyle = color; ctx.fill();
  ctx.strokeStyle = 'rgba(255,255,255,.7)'; ctx.lineWidth = 1.5; ctx.stroke();
  drawPaths(ctx, cell, false, tx, ty, sc, '#f7f4e8', 0.075);
}
