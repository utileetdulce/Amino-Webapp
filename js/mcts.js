// AlphaZero-style Monte-Carlo Tree Search for Amino (PUCT, network priors + value, open-loop for chance).
//
// Hidden information (opponent ant cards, stacks, a defender's hidden fist) is handled by determinization:
// every simulation starts from a fresh random world that is consistent with what the searching player knows.
// Dice / resource draws are re-sampled each simulation (open-loop tree keyed by action sequence).
import { actor, applyAction, legalActions, fastClone, actionKey, placementTiers } from './engine.js';
import { buildCtx, stateFeatures, candidateMatrix } from './features.js';
import { softmax } from './nn.js';

const shuffleIn = (a) => {
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
};

// Re-deal everything `observer` cannot see. Mutates and returns s (call on a clone!).
export function determinize(s, observer) {
  const pool = s.cardStack.slice();
  for (const q of s.players) if (q.idx !== observer) pool.push(...q.cards);
  shuffleIn(pool);
  for (const q of s.players) if (q.idx !== observer) q.cards = pool.splice(0, q.cards.length);
  s.cardStack = pool;
  shuffleIn(s.tileStack); shuffleIn(s.resStack);
  const pd = s.pending;
  if (pd && pd.type === 'guess' && pd.att === observer) pd.hide = Math.random() < 0.5 ? 0 : 1; // attacker does not know the fist
  s.seed = (Math.random() * 2 ** 31) | 0;
  return s;
}

// deterministic candidate cap: keep structural actions; of the placements keep the most relevant ones first (see placementTiers,
// at most 3/4 of the room) and thin out the rest evenly
export function capDeterministic(acts, max, s) {
  if (acts.length <= max) return acts;
  const keep = [], bulk = [];
  for (const a of acts) (a.t === 'place' || a.t === 'overlay' ? bulk : keep).push(a);
  const room = Math.max(4, max - keep.length);
  if (bulk.length <= room) return keep.concat(bulk);
  const tiers = placementTiers(s, bulk);
  const order = bulk.map((_, i) => i).filter((i) => tiers[i][0] > 0)
    .sort((i, j) => tiers[j][0] - tiers[i][0] || tiers[i][1] - tiers[j][1] || i - j);
  const top = order.slice(0, room - Math.floor(room / 4));
  const taken = new Set(top);
  const rest = bulk.map((_, i) => i).filter((i) => !taken.has(i));
  const k = room - top.length;
  const pick = top.slice();
  for (let i = 0; i < k; i++) pick.push(rest[Math.floor((i * rest.length) / k)]);
  pick.sort((i, j) => i - j);
  return keep.concat(pick.map((i) => bulk[i]));
}

function gammaSample(k) { // Marsaglia–Tsang, k >= 1 not required (boost for k < 1)
  if (k < 1) return gammaSample(k + 1) * Math.pow(Math.random(), 1 / k);
  const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x, v;
    do {
      const u1 = Math.random() || 1e-12, u2 = Math.random();
      x = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2); v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = Math.random() || 1e-12;
    if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) return d * v;
  }
}

class Node {
  constructor() { this.N = 0; this.W = 0; this.children = null; } // W: sum of values from the observer's view
}

export class MCTS {
  constructor(net, { sims = 64, cpuct = 1.25, maxDepth = 18, maxCands = 40, topK = 12, timeMs = 0, dirichlet = 0, dirAlpha = 0.3, fpu = 0 } = {}) {
    // topK: only the k best candidates (by network prior, plus 'end') are ever explored at a node. With few simulations a wide
    // tree gives flat visit counts, which is both weaker play and a poor training target.
    Object.assign(this, { net, sims, cpuct, maxDepth, maxCands, topK, timeMs, dirichlet, dirAlpha, fpu });
  }

  candidates(s) { return capDeterministic(legalActions(s, false), this.maxCands, s); }

  // value of state s from the view of the player to move there, plus candidate priors
  evaluate(s, acts, wantPriors) {
    const ctx = buildCtx(s, actor(s));
    const sf = stateFeatures(ctx);
    if (this.net.graph) {
      const r = this.net.evaluate(ctx, acts, sf, wantPriors ? candidateMatrix(ctx, acts) : null);
      return { v: r.v, priors: wantPriors ? softmax(r.logits) : null };
    }
    const v = this.net.forwardValue(sf);
    let priors = null;
    if (wantPriors) {
      const A = candidateMatrix(ctx, acts);
      priors = softmax(this.net.forwardPolicy(sf, A, acts.length).logits);
    }
    return { v, priors };
  }

  // observer-perspective value of a position (2-player zero-sum assumption for the sign flip)
  static terminal(s, observer) { return s.winner < 0 ? 0 : s.winner === observer ? 1 : -1; }

  advanceForced(s) {
    // skip decisions with a single legal action (they carry no information)
    let guard = 0;
    while (!s.over && guard++ < 200) {
      const la = legalActions(s, false);
      if (la.length !== 1) return la;
      applyAction(s, la[0]);
    }
    return s.over ? [] : legalActions(s, false);
  }

  /** Run the search from `s0`; returns { acts, visits, q, value, rootActor } */
  search(s0, { sims = this.sims, noise = this.dirichlet } = {}) {
    const observer = actor(s0);
    const root = new Node();
    const t0 = Date.now();
    let rootActs = null;
    for (let i = 0; i < sims; i++) {
      if (this.timeMs && i > 1 && Date.now() - t0 > this.timeMs) break;
      const s = determinize(fastClone(s0), observer);
      const r = this.simulate(root, s, observer, noise && i === 0);
      if (!rootActs && root.children) rootActs = root.acts;
    }
    const acts = root.acts || [];
    const visits = acts.map((a) => (root.children.get(actionKey(a))?.N || 0));
    const value = root.N ? root.W / root.N : 0;
    const q = acts.map((a) => { const c = root.children.get(actionKey(a)); return c && c.N ? c.W / c.N : value; });
    return { acts, visits, q, value, prior: root.prior || [], observer };
  }

  simulate(root, s, observer, addNoise) {
    const path = [];
    let node = root, depth = 0, value = 0;
    let acts = this.advanceForced(s);
    for (;;) {
      if (s.over) { value = MCTS.terminal(s, observer); break; }
      if (depth >= this.maxDepth) {
        const { v } = this.evaluate(s, acts, false);
        value = actor(s) === observer ? v : -v;
        break;
      }
      acts = capDeterministic(acts, this.maxCands, s);
      if (!node.children) { // expand + evaluate leaf
        const { v, priors } = this.evaluate(s, acts, true);
        node.children = new Map(); node.acts = acts;
        const order = acts.map((_, i) => i).sort((a, b) => priors[b] - priors[a] || a - b);
        const keep = new Set(order.slice(0, Math.max(2, this.topK)));
        const endIdx = acts.findIndex((a) => a.t === 'end');
        if (endIdx >= 0) keep.add(endIdx);
        const idx = [...keep].sort((a, b) => a - b);
        let pk = idx.map((i) => priors[i]);
        const tot = pk.reduce((a, b) => a + b, 0) || 1e-12;
        pk = pk.map((x) => x / tot);
        if (addNoise && node === root) {
          const g = idx.map(() => gammaSample(this.dirAlpha)); const gs = g.reduce((a, b) => a + b, 0) || 1;
          pk = pk.map((p, i) => (1 - this.dirichlet) * p + this.dirichlet * g[i] / gs);
        }
        idx.forEach((i, j) => node.children.set(actionKey(acts[i]), { N: 0, W: 0, P: pk[j], node: null }));
        if (node === root) root.prior = Array.from(priors);
        value = actor(s) === observer ? v : -v;
        break;
      }
      // select among legal actions of THIS simulated state (open loop)
      const sign = actor(s) === observer ? 1 : -1;
      const sqrtN = Math.sqrt(node.N + 1);
      const parentQ = node.N ? node.W / node.N : 0;
      let best = null, bestScore = -Infinity, bestAct = null;
      if (!acts.some((a) => node.children.has(actionKey(a)))) { // no explored candidate is legal in this world: open the node up
        for (const a of acts) node.children.set(actionKey(a), { N: 0, W: 0, P: 1 / acts.length, node: null });
      }
      for (const a of acts) {
        const c = node.children.get(actionKey(a));
        if (!c) continue;
        const q = c.N ? c.W / c.N : parentQ - sign * this.fpu;
        const score = sign * q + this.cpuct * c.P * sqrtN / (1 + c.N);
        if (score > bestScore) { bestScore = score; best = c; bestAct = a; }
      }
      path.push({ node, child: best });
      applyAction(s, bestAct);
      depth++;
      if (!best.node) best.node = new Node();
      node = best.node;
      acts = this.advanceForced(s);
    }
    for (const { node: nd, child } of path) { nd.N++; nd.W += value; child.N++; child.W += value; }
    if (!path.length) { root.N++; root.W += value; }
    return value;
  }
}

// pick an action index from visit counts with temperature (tau=0 -> argmax)
export function pickByVisits(visits, tau = 1) {
  if (tau <= 0.05) { let b = 0; for (let i = 1; i < visits.length; i++) if (visits[i] > visits[b]) b = i; return b; }
  const w = visits.map((v) => Math.pow(v, 1 / tau));
  const z = w.reduce((a, b) => a + b, 0);
  if (!z) return 0;
  let r = Math.random() * z;
  for (let i = 0; i < w.length; i++) { r -= w[i]; if (r <= 0) return i; }
  return w.length - 1;
}
