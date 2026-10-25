// Graph network (inference only): a small message-passing network over the board's path components feeds the MLP heads.
// Port of the `hg > 0` architecture in rust-trainer/src/nn.rs + build_graph in features.rs (training happens in the Rust trainer).
//   node = one path component (cell, comp); edges follow open tile edges; 20 features per node (own view)
//   h0 = relu(x Win + b); per layer: h += relu(b + h Wself + sum_e h[nbr_e] We[e]);
//   state input = [41 state features, mean node embedding, own ant embedding, enemy ant embedding];
//   candidate input additionally gets the mean embedding of the nodes the action targets.
import { SF, AF } from './features.js';
import { cellX, cellY, isUp, neighborId, neighbor } from './geometry.js';
import { compsOf } from './engine.js';

export const NF = 20;
const EXTK = 3;

function layout(a) {
  const { h1, h2, hv, hg, gl } = a;
  const nin = SF + EXTK * hg;
  let o = 0; const L = { nin };
  const take = (n) => { const r = o; o += n; return r; };
  L.ws = take(nin * h1); L.wa = take(AF * h1); L.b1 = take(h1); L.w2 = take(h1 * h2); L.b2 = take(h2); L.w3 = take(h2); L.b3 = take(1);
  L.v1 = take(nin * hv); L.bv1 = take(hv); L.v2 = take(hv); L.bv2 = take(1);
  L.wc = take(hg * h1); L.v3 = take(hv); L.bv3 = take(1); L.win = take(NF * hg); L.bin = take(hg);
  L.lsz = 4 * hg * hg + hg; L.gl0 = take(L.lsz * gl);
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

export class GNet {
  constructor(arch, theta) {
    this.arch = { ...arch };
    this.L = layout(this.arch);
    if (theta.length !== this.L.size) throw new Error(`theta size ${theta.length} != ${this.L.size}`);
    this.theta = Float32Array.from(theta);
    this.graph = true;
  }
  static fromJSON(j) { return new GNet(j.arch, j.theta); }

  buildGraph(ctx, acts) {
    const { s, pi, p } = ctx;
    const ids = Object.keys(s.cells).map(Number);
    const base = new Map(); let n = 0;
    for (const cid of ids) { base.set(cid, n); n += compsOf(s.cells[cid]).length; }
    const x = new Float32Array(n * NF);
    const nbr = new Int16Array(n * 3).fill(-1);
    const antCell = p.ant ? p.ant.cell : -1;
    const antNode = p.ant ? base.get(antCell) + compIdx(s.cells[antCell], p.ant.edge) : -1;
    let foe = -1; const foeCells = new Set();
    for (let q = 0; q < s.n; q++) {
      if (q === pi) continue;
      const a = s.players[q].ant;
      if (a) { foeCells.add(a.cell); if (foe < 0) foe = base.get(a.cell) + compIdx(s.cells[a.cell], a.edge); }
    }
    const carryVal = p.carry >= 0 ? p.values[s.resources[p.carry].kind] / 3 : 0;
    for (const cid of ids) {
      const c = s.cells[cid];
      const fr = ctx.freeRes.find((r) => r.cell === cid);
      const ca = ctx.carriers.find((r) => r.cell === cid);
      const da = ctx.distAnt.get(cid), dh = ctx.distHill.get(cid);
      const comps = compsOf(c);
      for (let ci = 0; ci < comps.length; ci++) {
        const node = base.get(cid) + ci, o = node * NF, edges = comps[ci];
        x[o] = isUp(c.x, c.y) ? 1 : 0;
        for (const e of edges) x[o + 1 + e] = 1;
        x[o + 4] = comps.length / 3;
        x[o + 5] = c.hill === pi ? 1 : 0;
        x[o + 6] = c.hill >= 0 && c.hill !== pi ? 1 : 0;
        x[o + 7] = fr ? fr.val / 3 : 0;
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
    const cellNodes = (cid, out) => { const c = s.cells[cid]; if (!c) return; const b = base.get(cid), k = compsOf(c).length; for (let i = 0; i < k; i++) out.push(b + i); };
    const cand = acts.map((a) => {
      const v = [];
      switch (a.t) {
        case 'move': case 'tmove': case 'wings': {
          const cid = Math.floor(a.to / 3), ci = a.to - cid * 3, c = s.cells[cid];
          if (c) v.push(base.get(cid) + Math.min(ci, compsOf(c).length - 1));
          break;
        }
        case 'place': {
          const cx = cellX(a.cell), cy = cellY(a.cell);
          for (let e = 0; e < 3; e++) {
            const [nid] = neighborId(a.cell, e);
            const nc = s.cells[nid];
            if (nc) { const [, , ne] = neighbor(cx, cy, e); if (nc.open[ne]) v.push(base.get(nid) + compIdx(nc, ne)); }
          }
          break;
        }
        case 'bridge': cellNodes(a.a, v); cellNodes(a.b, v); break;
        case 'blockage': case 'overlay': case 'inc_hill': cellNodes(a.cell, v); break;
        default: break;
      }
      return v;
    });
    return { n, x, nbr, ant: antNode, foe, cand };
  }

  // GNN forward; returns the final node embeddings
  embed(g) {
    const { hg, gl } = this.arch, L = this.L, th = this.theta, n = g.n;
    let h = new Float32Array(n * hg);
    const tmp = new Float32Array(hg);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < hg; j++) tmp[j] = th[L.bin + j];
      addmv(tmp, g.x, i * NF, NF, th, L.win, hg);
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

  /** value (actor's view) and, when acts are given, candidate logits */
  evaluate(ctx, acts, sf, A = null) {
    const wantLogits = !!A;
    const { h1, h2, hv, hg } = this.arch, L = this.L, th = this.theta;
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
    // value
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
    for (let c = 0; c < acts.length; c++) {
      a1.set(hs);
      addmv(a1, A, c * AF, AF, th, L.wa, h1);
      ce.fill(0);
      const nodes = g.cand[c];
      if (nodes.length) {
        for (const nd of nodes) for (let k = 0; k < hg; k++) ce[k] += h[nd * hg + k] / nodes.length;
        addmv(a1, ce, 0, hg, th, L.wc, h1);
      }
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
