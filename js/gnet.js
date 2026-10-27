// Graph network (inference only): a small message-passing network over the board's path components feeds the MLP heads.
// Port of the `hg > 0` architecture in rust-trainer/src/nn.rs + build_graph in features.rs (training happens in the Rust trainer).
//   node = one path component (cell, comp); edges follow open tile edges; per-node features in the current player's view
//   h0 = relu(x Win + b); per layer: h += relu(b + h Wself + sum_e h[nbr_e] We[e]);
//   state input = [41 state features, mean node embedding, own ant embedding, enemy ant embedding, (v2) global extras];
//   candidate input additionally gets the mean embedding of the nodes the action targets and (v2) placement/transport extras.
// Feature version fv: 1 = 20 node features; 2 adds enemy reach, regions, frontier, global extras and candidate extras;
// 3 adds "is my ant cut off?" (global) and a one-action lookahead per candidate (does it reconnect the ant with a hill?).
import { SF, AF } from './features.js';
import { cellX, cellY, isUp, neighborId, neighbor } from './geometry.js';
import { compsOf, bfs, antStarts, hillCells, nodeCell, makeCellShape, resNode, hasCard, fastClone, applyAction } from './engine.js';

export const NF = 20, NF2 = 32, GXN = 14, CXN = 17, GXN3 = 24, CXN3 = 21;
const EXTK = 3;
const R_ANT = 1, R_FOE = 2, R_MYH = 4, R_EH = 8, R_FREE = 16;
const nfOf = (a) => (a.fv >= 2 ? NF2 : NF), gxnOf = (a) => (a.fv >= 3 ? GXN3 : a.fv >= 2 ? GXN : 0), cxnOf = (a) => (a.fv >= 3 ? CXN3 : a.fv >= 2 ? CXN : 0);
// actions whose effect on the ant's connections is looked ahead (fv 3)
const LOOK = new Set(['place', 'bridge', 'overlay', 'inc_hill', 'wings', 'move', 'tmove']);

// where player pi's ant can get to: any own hill (paths, bridges, tunnels), transport distance to the nearest own hill
// (no tunnels; 99 = none), any free resource
function reachInfo(s, pi) {
  const p = s.players[pi];
  if (!p.ant) return { any: false, td: 99, free: false, a: null };
  const st = antStarts(s, pi);
  const a = bfs(s, pi, st), t = bfs(s, pi, st, { underground: false });
  let any = false, td = 99;
  for (const h of hillCells(s, pi)) { const n = h * 3; if (a.has(n)) any = true; if (t.has(n)) td = Math.min(td, t.get(n)); }
  const free = s.resources.some((r) => r.owner < 0 && r.carrier < 0 && r.cell >= 0 && a.has(resNode(s.cells[r.cell])));
  return { any, td, free, a };
}

function layout(a) {
  const { h1, h2, hv, hg, gl } = a;
  const nin = SF + EXTK * hg + gxnOf(a);
  let o = 0; const L = { nin };
  const take = (n) => { const r = o; o += n; return r; };
  L.ws = take(nin * h1); L.wa = take(AF * h1); L.b1 = take(h1); L.w2 = take(h1 * h2); L.b2 = take(h2); L.w3 = take(h2); L.b3 = take(1);
  L.v1 = take(nin * hv); L.bv1 = take(hv); L.v2 = take(hv); L.bv2 = take(1);
  L.wc = take(hg * h1); L.v3 = take(hv); L.bv3 = take(1); L.win = take(nfOf(a) * hg); L.bin = take(hg);
  L.lsz = 4 * hg * hg + hg; L.gl0 = take(L.lsz * gl); L.wx = take(cxnOf(a) * h1);
  L.size = o;
  return L;
}

function addmv(out, x, xo, n, w, wo, width) {
  for (let i = 0; i < n; i++) {
    const xi = x[xo + i];
    if (xi === 0) continue;
    const row = wo + i * width;
    for (let j = 0; j < width; j++) out[j] += xi * w[row + j];
  }
}

const compIdx = (c, edge) => { const cs = compsOf(c); for (let i = 0; i < cs.length; i++) if (cs[i].includes(edge)) return i; return 0; };
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

function cellMin(map) {
  const m = new Map();
  for (const [n, d] of map) { const c = nodeCell(n); if (!m.has(c) || m.get(c) > d) m.set(c, d); }
  return m;
}
function mergeMin(a, b) { for (const [k, d] of b) if (!a.has(k) || d < a.get(k)) a.set(k, d); }

// P(roll >= d) and E[min(roll, d)] / 6 for one die (0..5) or two dice
function rollOdds(two, d) {
  let p = 0, e = 0, tot = 0;
  for (let a = 0; a < 6; a++) for (let b = 0; b < (two ? 6 : 1); b++) { const r = a + b; tot++; if (r >= d) p++; e += Math.min(r, d); }
  return [p / tot, e / tot / 6];
}

export class GNet {
  constructor(arch, theta) {
    this.arch = { fv: 1, ...arch };
    this.L = layout(this.arch);
    if (theta.length !== this.L.size) throw new Error(`theta size ${theta.length} != ${this.L.size}`);
    this.theta = Float32Array.from(theta);
    this.graph = true;
  }
  static fromJSON(j) { return new GNet(j.arch, j.theta); }

  buildGraph(ctx, acts) {
    const { s, pi, p } = ctx;
    const v2 = this.arch.fv >= 2, nf = nfOf(this.arch);
    const ids = Object.keys(s.cells).map(Number);
    const base = new Map(); let n = 0;
    for (const cid of ids) { base.set(cid, n); n += compsOf(s.cells[cid]).length; }
    const x = new Float32Array(n * nf);
    const nbr = new Int16Array(n * 3).fill(-1);
    const antCell = p.ant ? p.ant.cell : -1;
    const antNode = p.ant ? base.get(antCell) + compIdx(s.cells[antCell], p.ant.edge) : -1;
    let foe = -1; const foeCells = new Set(); const foeNodes = [];
    for (let q = 0; q < s.n; q++) {
      if (q === pi) continue;
      const a = s.players[q].ant;
      if (a) { foeCells.add(a.cell); const nd = base.get(a.cell) + compIdx(s.cells[a.cell], a.edge); foeNodes.push(nd); if (foe < 0) foe = nd; }
    }
    const carryVal = p.carry >= 0 ? p.values[s.resources[p.carry].kind] / 3 : 0;
    for (const cid of ids) {
      const c = s.cells[cid];
      const fr = ctx.freeRes.find((r) => r.cell === cid);
      const ca = ctx.carriers.find((r) => r.cell === cid);
      const da = ctx.distAnt.get(cid), dh = ctx.distHill.get(cid);
      const comps = compsOf(c);
      for (let ci = 0; ci < comps.length; ci++) {
        const node = base.get(cid) + ci, o = node * nf, edges = comps[ci];
        x[o] = isUp(c.x, c.y) ? 1 : 0;
        for (const e of edges) x[o + 1 + e] = 1;
        x[o + 4] = comps.length / 3;
        x[o + 5] = c.hill === pi ? 1 : 0;
        x[o + 6] = c.hill >= 0 && c.hill !== pi ? 1 : 0;
        x[o + 7] = fr && fr.node === cid * 3 + ci ? fr.val / 3 : 0;
        x[o + 8] = c.block >= 0 ? 1 : 0;
        x[o + 9] = c.bridge > 0 ? 1 : 0;
        x[o + 10] = node === antNode ? 1 : 0;
        x[o + 11] = antCell === cid ? 1 : 0;
        x[o + 12] = foeCells.has(cid) ? 1 : 0;
        x[o + 13] = ca ? ca.val / 3 : 0;
        x[o + 14] = da !== undefined ? Math.min(da, 14) / 14 : 1;
        x[o + 15] = da !== undefined ? 1 : 0;
        x[o + 16] = dh !== undefined ? Math.min(dh, 14) / 14 : 1;
        x[o + 17] = dh !== undefined ? 1 : 0;
        x[o + 18] = c.overlaid ? 1 : 0;
        x[o + 19] = antCell === cid ? carryVal : 0;
        for (const e of edges) {
          const [nid, ne] = neighborId(cid, e);
          const nc = s.cells[nid];
          if (nc && nc.open[ne]) nbr[node * 3 + e] = base.get(nid) + compIdx(nc, ne);
        }
      }
    }
    const g = { n, nf, x, nbr, ant: antNode, foe, cand: null, gx: null, cx: null, region: null, rflags: null, base };
    if (v2) this.addV2(g, ctx, ids, foeNodes);
    const cxn = cxnOf(this.arch), v3 = this.arch.fv >= 3;
    const r0 = v3 ? reachInfo(s, pi) : null;
    if (v3) this.addV3(g, ctx, r0);
    // candidates
    const cellNodes = (cid, out) => { const c = s.cells[cid]; if (!c) return; const b = base.get(cid), k = compsOf(c).length; for (let i = 0; i < k; i++) out.push(b + i); };
    g.cx = v2 ? new Float32Array(acts.length * cxn) : null;
    g.cand = acts.map((a, ci) => {
      const v = [];
      switch (a.t) {
        case 'move': case 'tmove': case 'wings': {
          const cid = Math.floor(a.to / 3), k = a.to - cid * 3, c = s.cells[cid];
          if (c) v.push(base.get(cid) + Math.min(k, compsOf(c).length - 1));
          break;
        }
        case 'place': {
          const cx = cellX(a.cell), cy = cellY(a.cell);
          const regs = []; let frontier = 0;
          const sh = makeCellShape(cx, cy, a.tile, a.rot);
          for (let e = 0; e < 3; e++) {
            const [nid] = neighborId(a.cell, e);
            const nc = s.cells[nid];
            if (nc) {
              const [, , ne] = neighbor(cx, cy, e);
              if (nc.open[ne]) { const node = base.get(nid) + compIdx(nc, ne); v.push(node); if (v2) { const r = g.region[node]; if (!regs.includes(r)) regs.push(r); } }
            } else if (sh.open[e]) frontier++;
          }
          if (v2) {
            const o = ci * cxn, cxv = g.cx;
            for (let e = 0; e < 3; e++) cxv[o + e] = sh.open[e];
            const same = (i, j) => sh.comps.some((m) => m.includes(i) && m.includes(j));
            cxv[o + 3] = same(0, 1) ? 1 : 0; cxv[o + 4] = same(0, 2) ? 1 : 0; cxv[o + 5] = same(1, 2) ? 1 : 0;
            cxv[o + 6] = frontier / 3;
            let un = 0, singleAntFree = false, singleAntFoe = false;
            for (const r of regs) { const f = g.rflags[r]; un |= f; if ((f & R_ANT) && (f & R_FREE)) singleAntFree = true; if ((f & R_ANT) && (f & R_FOE)) singleAntFoe = true; }
            cxv[o + 7] = un & R_ANT ? 1 : 0; cxv[o + 8] = un & R_FOE ? 1 : 0; cxv[o + 9] = un & R_FREE ? 1 : 0;
            cxv[o + 10] = un & R_MYH ? 1 : 0; cxv[o + 11] = un & R_EH ? 1 : 0;
            cxv[o + 12] = regs.length >= 2 ? 1 : 0;
            cxv[o + 13] = regs.length >= 2 && (un & R_ANT) && (un & R_FREE) && !singleAntFree ? 1 : 0;
            cxv[o + 14] = regs.length >= 2 && (un & R_ANT) && (un & R_FOE) && !singleAntFoe ? 1 : 0;
          }
          break;
        }
        case 'bridge': cellNodes(a.a, v); cellNodes(a.b, v); break;
        case 'blockage': case 'overlay': case 'inc_hill': cellNodes(a.cell, v); break;
        case 'transport':
          if (v2) { const [pd, ea] = rollOdds(!!a.card, ctx.curHillDist >= 99 ? 99 : ctx.curHillDist); g.cx[ci * cxn + 15] = pd; g.cx[ci * cxn + 16] = ea; }
          break;
        default: break;
      }
      if (v3) { // lookahead: the ant's connections after this action
        let r1 = r0;
        if (LOOK.has(a.t)) { const s1 = fastClone(s); applyAction(s1, a); r1 = reachInfo(s1, pi); }
        const o = ci * cxn;
        g.cx[o + 17] = r1.td < 99 ? Math.min(r1.td, 14) / 14 : 1;
        g.cx[o + 18] = r1.any ? 1 : 0;
        g.cx[o + 19] = (p.carry >= 0 ? r0.td >= 99 && r1.td < 99 : !r0.any && r1.any) ? 1 : 0;
        g.cx[o + 20] = r1.free ? 1 : 0;
      }
      return v;
    });
    return g;
  }

  // v3 global extras: is the ant cut off, and what could change that
  addV3(g, ctx, r) {
    const { s, pi, p } = ctx, gx = g.gx;
    gx[14] = r.any ? 1 : 0;
    gx[15] = r.td < 99 ? 1 : 0;
    gx[16] = r.td < 99 ? Math.min(r.td, 14) / 14 : 1;
    gx[17] = g.ant >= 0 ? Math.min(g.regFront[g.region[g.ant]] / 6, 1) : 0;
    gx[18] = r.free ? 1 : 0;
    gx[19] = p.carry >= 0 && r.td >= 99 ? 1 : 0;
    gx[20] = p.ant && !r.any && !r.free ? 1 : 0;
    gx[21] = p.ant ? hillCells(s, pi).filter((h) => !r.a.has(h * 3)).length / 2 : 0;
    gx[22] = hasCard(p, 'superimpose') ? 1 : 0;
    gx[23] = p.cards.length >= 2 ? 1 : 0;
  }

  // v2: node features 20..31, regions, global extras
  addV2(g, ctx, ids, foeNodes) {
    const { s, pi, p } = ctx;
    const { n, nf, nbr, base, x } = g;
    const region = new Int32Array(n).fill(-1);
    const size = []; const stack = []; let nreg = 0;
    for (let st = 0; st < n; st++) {
      if (region[st] >= 0) continue;
      region[st] = nreg; stack.push(st); let cnt = 0;
      while (stack.length) {
        const v = stack.pop(); cnt++;
        for (let e = 0; e < 3; e++) { const m = nbr[v * 3 + e]; if (m >= 0 && region[m] < 0) { region[m] = nreg; stack.push(m); } }
      }
      size.push(cnt); nreg++;
    }
    const rflags = new Uint8Array(nreg);
    if (g.ant >= 0) rflags[region[g.ant]] |= R_ANT;
    for (const fn of foeNodes) rflags[region[fn]] |= R_FOE;
    for (const cid of ids) {
      const c = s.cells[cid];
      const fr = ctx.freeRes.find((r) => r.cell === cid);
      const k = compsOf(c).length;
      for (let ci = 0; ci < k; ci++) {
        const r = region[base.get(cid) + ci];
        if (c.hill === pi) rflags[r] |= R_MYH;
        if (c.hill >= 0 && c.hill !== pi) rflags[r] |= R_EH;
        if (fr && fr.node === cid * 3 + ci) rflags[r] |= R_FREE;
      }
    }
    const distFoe = new Map(), distEh = new Map();
    for (let q = 0; q < s.n; q++) {
      if (q === pi) continue;
      if (s.players[q].ant) mergeMin(distFoe, cellMin(bfs(s, q, antStarts(s, q))));
      const hs = hillCells(s, q);
      if (hs.length) mergeMin(distEh, cellMin(bfs(s, q, hs.map((h) => h * 3), { underground: false })));
    }
    const regFront = new Int32Array(nreg);
    const laidNow = new Set(s.resources.filter((r) => r.owner < 0 && r.carrier < 0 && r.cell >= 0 && r.laidTurn === s.turnNo).map((r) => r.cell));
    for (const cid of ids) {
      const c = s.cells[cid];
      const dfo = distFoe.get(cid), deh = distEh.get(cid);
      const comps = compsOf(c);
      for (let ci = 0; ci < comps.length; ci++) {
        const node = base.get(cid) + ci, o = node * nf;
        const r = region[node], fl = rflags[r];
        let frE = 0;
        for (const e of comps[ci]) { const [nid] = neighborId(cid, e); if (!s.cells[nid]) frE++; }
        regFront[r] += frE;
        x[o + 20] = dfo !== undefined ? Math.min(dfo, 14) / 14 : 1;
        x[o + 21] = dfo !== undefined ? 1 : 0;
        x[o + 22] = deh !== undefined ? Math.min(deh, 14) / 14 : 1;
        x[o + 23] = deh !== undefined ? 1 : 0;
        x[o + 24] = fl & R_ANT ? 1 : 0; x[o + 25] = fl & R_FOE ? 1 : 0; x[o + 26] = fl & R_MYH ? 1 : 0; x[o + 27] = fl & R_EH ? 1 : 0; x[o + 28] = fl & R_FREE ? 1 : 0;
        x[o + 29] = size[r] / n;
        x[o + 30] = frE / 3;
        x[o + 31] = laidNow.has(cid) ? 1 : 0;
      }
    }
    // global extras
    let fo = -1;
    for (let q = 0; q < s.n; q++) if (q !== pi) { fo = q; break; }
    const gx = new Float32Array(gxnOf(this.arch));
    gx[6] = 1;
    if (fo >= 0) {
      const pq = s.players[fo];
      gx[0] = Math.min(pq.hand.length, 12) / 8;
      gx[1] = pq.cards.length / 6;
      gx[2] = pq.hillsInHand / 2;
      gx[3] = pq.ant ? 1 : 0;
      gx[4] = pq.carry >= 0 ? 1 : 0;
      if (pq.carry >= 0) gx[5] = p.values[s.resources[pq.carry].kind] / 3;
      if (pq.carry >= 0 && pq.ant) { const d = distEh.get(pq.ant.cell); if (d !== undefined) gx[6] = Math.min(d, 14) / 14; }
      gx[12] = s.resources.filter((r) => r.owner === fo).length / 6;
    }
    gx[7] = p.ant && distFoe.has(p.ant.cell) ? 1 : 0;
    let dm = 14, df = 14, nm = 0, nfo = 0, nregFree = 0;
    for (const r of ctx.freeRes) {
      const cid = r.cell;
      if (ctx.distAnt.has(cid)) { dm = Math.min(dm, ctx.distAnt.get(cid)); nm++; }
      if (distFoe.has(cid)) { df = Math.min(df, distFoe.get(cid)); nfo++; }
      if (g.ant >= 0 && region[base.get(cid) + (r.node - cid * 3)] === region[g.ant]) nregFree++;
    }
    gx[8] = clamp((df - dm) / 14, -1, 1);
    gx[9] = nm / 4; gx[10] = nfo / 4;
    let dhMy = 14;
    for (const h of ctx.hs) if (distFoe.has(h)) dhMy = Math.min(dhMy, distFoe.get(h));
    gx[11] = dhMy / 14;
    gx[13] = nregFree / 4;
    g.gx = gx; g.region = region; g.rflags = rflags; g.regFront = regFront;
  }

  // GNN forward; returns the final node embeddings
  embed(g) {
    const { hg, gl } = this.arch, L = this.L, th = this.theta, n = g.n;
    let h = new Float32Array(n * hg);
    const tmp = new Float32Array(hg);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < hg; j++) tmp[j] = th[L.bin + j];
      addmv(tmp, g.x, i * g.nf, g.nf, th, L.win, hg);
      for (let j = 0; j < hg; j++) h[i * hg + j] = tmp[j] > 0 ? tmp[j] : 0;
    }
    for (let k = 0; k < gl; k++) {
      const base = L.gl0 + k * L.lsz, we = base + hg * hg, bl = base + 4 * hg * hg;
      const hn = Float32Array.from(h);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < hg; j++) tmp[j] = th[bl + j];
        addmv(tmp, h, i * hg, hg, th, base, hg);
        for (let e = 0; e < 3; e++) { const m = g.nbr[i * 3 + e]; if (m >= 0) addmv(tmp, h, m * hg, hg, th, we + e * hg * hg, hg); }
        for (let j = 0; j < hg; j++) if (tmp[j] > 0) hn[i * hg + j] += tmp[j];
      }
      h = hn;
    }
    return h;
  }

  /** value (actor's view) and, when A (candidate matrix) is given, candidate logits */
  evaluate(ctx, acts, sf, A = null) {
    const { h1, h2, hv, hg } = this.arch, L = this.L, th = this.theta;
    const wantLogits = !!A;
    const g = this.buildGraph(ctx, wantLogits ? acts : []);
    const h = this.embed(g);
    const xs = new Float32Array(L.nin);
    xs.set(sf);
    if (g.n) {
      for (let i = 0; i < g.n; i++) for (let k = 0; k < hg; k++) xs[SF + k] += h[i * hg + k];
      for (let k = 0; k < hg; k++) xs[SF + k] /= g.n;
    }
    if (g.ant >= 0) for (let k = 0; k < hg; k++) xs[SF + hg + k] = h[g.ant * hg + k];
    if (g.foe >= 0) for (let k = 0; k < hg; k++) xs[SF + 2 * hg + k] = h[g.foe * hg + k];
    if (g.gx) xs.set(g.gx, SF + EXTK * hg);
    const a = new Float32Array(hv);
    for (let j = 0; j < hv; j++) a[j] = th[L.bv1 + j];
    addmv(a, xs, 0, L.nin, th, L.v1, hv);
    let v = th[L.bv2];
    for (let j = 0; j < hv; j++) { if (a[j] < 0) a[j] = 0; v += a[j] * th[L.v2 + j]; }
    if (!wantLogits) return { v, logits: null };
    const hs = new Float32Array(h1);
    for (let j = 0; j < h1; j++) hs[j] = th[L.b1 + j];
    addmv(hs, xs, 0, L.nin, th, L.ws, h1);
    const logits = new Float32Array(acts.length);
    const a1 = new Float32Array(h1), a2 = new Float32Array(h2), ce = new Float32Array(hg);
    const cxn = cxnOf(this.arch);
    for (let c = 0; c < acts.length; c++) {
      a1.set(hs);
      addmv(a1, A, c * AF, AF, th, L.wa, h1);
      ce.fill(0);
      const nodes = g.cand[c];
      if (nodes.length) {
        for (const nd of nodes) for (let k = 0; k < hg; k++) ce[k] += h[nd * hg + k] / nodes.length;
        addmv(a1, ce, 0, hg, th, L.wc, h1);
      }
      if (cxn) addmv(a1, g.cx, c * cxn, cxn, th, L.wx, h1);
      for (let j = 0; j < h1; j++) if (a1[j] < 0) a1[j] = 0;
      for (let k = 0; k < h2; k++) a2[k] = th[L.b2 + k];
      addmv(a2, a1, 0, h1, th, L.w2, h2);
      let z = th[L.b3];
      for (let k = 0; k < h2; k++) { if (a2[k] < 0) a2[k] = 0; z += a2[k] * th[L.w3 + k]; }
      logits[c] = z;
    }
    return { v, logits };
  }
}

/** Net.fromJSON for either architecture */
export function netFromJSON(j, NetClass) { return j.arch && j.arch.hg > 0 ? GNet.fromJSON(j) : NetClass.fromJSON(j); }
