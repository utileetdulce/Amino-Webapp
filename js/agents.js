// Agents: random, heuristic, neural. All implement act(state) -> action (for state.pending actor or current player).
import { legalActions, actor, hasCard, reconnectInfo, reconnects, homeDist, fastClone, applyAction } from './engine.js';
import { buildCtx, stateFeatures, candidateMatrix, actionNumeric, AF } from './features.js';
import { softmax, sampleIdx } from './nn.js';
import { MCTS, pickByVisits } from './mcts.js';

export const MAX_CANDS = 40;

export function capCandidates(acts, max = MAX_CANDS, rng = Math.random) {
  if (acts.length <= max) return acts;
  const keep = acts.filter((a) => a.t !== 'place' && a.t !== 'overlay');
  const rest = acts.filter((a) => a.t === 'place' || a.t === 'overlay');
  const room = Math.max(4, max - keep.length);
  for (let i = rest.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [rest[i], rest[j]] = [rest[j], rest[i]]; }
  return keep.concat(rest.slice(0, room));
}

export class RandomAgent {
  constructor() { this.name = 'Zufall'; }
  act(s) {
    const acts = legalActions(s, false);
    const nonEnd = acts.filter((a) => a.t !== 'end');
    if (!nonEnd.length || Math.random() < 0.18) return acts.find((a) => a.t === 'end') || acts[0];
    return nonEnd[Math.floor(Math.random() * nonEnd.length)];
  }
}

// ---- heuristic: hand-tuned priorities over the same candidate set ----
export function heuristicScores(ctx, acts) {
  const { s, p } = ctx;
  const out = new Float32Array(acts.length);
  const f = new Float32Array(16);
  const handN = p.hand.length;
  const hillsPlaced = 2 - p.hillsInHand;
  const enemyCarrier = ctx.carriers.length > 0;
  const mayAttack = !s.used.fight && p.carry < 0;
  // carrying, but no transport path to an own hill: transporting is pointless, reconnecting comes first
  const stuck = p.carry >= 0 && !!p.ant && ctx.curHillDist >= 99;
  const rc = stuck ? reconnectInfo(s, ctx.pi) : null;
  // enemy carriers on their way home: overbuilding a triangle on that way (superimpose) can cut them off
  const homeQ = [];
  for (let q = 0; q < s.n; q++) if (q !== ctx.pi && s.players[q].carry >= 0 && homeDist(s, q) < 99) homeQ.push(q);
  const cuts = (s1) => homeQ.some((q) => homeDist(s1, q) >= 99);
  const after = (s0, x) => { const s1 = fastClone(s0); applyAction(s1, x); return s1; };
  for (let i = 0; i < acts.length; i++) {
    const a = acts[i];
    let v = 0;
    switch (a.t) {
      case 'tmove': v = s.cells[Math.floor(a.to / 3)].hill === ctx.pi ? 100 : 50; break;
      case 'tstay': v = 1; break;
      case 'transport': v = stuck ? -20 : a.card ? (ctx.curHillDist > 6 ? 60 : -50) : 70; break;
      case 'fight': v = 68; break;
      case 'load': v = 80; break;
      case 'unload': v = 0; break;
      case 'hide': case 'guess': v = 0; break;
      case 'revenge': v = 90; break;
      case 'decline': v = 0; break;
      case 'move': {
        const cid = Math.floor(a.to / 3);
        const fr = ctx.freeRes.find((r) => r.node === a.to);
        const ca = ctx.carriers.find((c) => c.cell === cid);
        if (p.carry < 0 && fr) v = 72 + fr.val;
        else if (mayAttack && ca) v = 55 + 3 * ca.val;
        else if (p.carry >= 0) {
          const dh = ctx.distHill.has(cid) ? ctx.distHill.get(cid) : 99;
          const gain = Math.min(ctx.curHillDist, 30) - Math.min(dh, 30);
          v = s.used.transport ? 2 : (gain > 0 ? 18 + gain * 3 : 0);
        } else v = 1;
        break;
      }
      case 'place': {
        f.fill(0);
        actionNumeric(ctx, a, f, 0);
        const reach = f[10], dAnt = f[2], toRes = f[11];
        // cut off from the hills with a resource: a new hill close to the ant makes the transport possible again
        if (a.piece === 1) v = hillsPlaced === 0 ? 90 : p.carry >= 0 && ctx.curHillDist >= 99 && reach ? 85 - 30 * dAnt : (ctx.freeRes.length ? 30 + 6 * (1 - toRes) : 22);
        else if (a.piece === 2) v = p.ant ? 52 + 14 * reach - 10 * dAnt : 0;
        else v = (handN > 2 ? 22 : 12) + 6 * reach + 6 * (1 - toRes);
        if (stuck && reconnects(s, rc, a)) v = Math.max(v, 88 - 10 * dAnt);
        break;
      }
      case 'inc_tiles': v = handN <= 1 ? 68 : handN <= 3 ? 40 : 28; break;
      case 'inc_card': v = p.cards.length === 0 ? 48 : 30; break;
      case 'inc_hill': // take back a hill the carrying ant cannot reach, to lay it again near the ant
        v = p.carry >= 0 && ctx.curHillDist >= 99 && p.hillsInHand === 0 && handN > 0 && !ctx.distAnt.has(a.cell) ? 60 : hillsPlaced === 2 && p.ant && ctx.freeRes.length === 0 ? 8 : -5; break;
      case 'swap': v = 4; break;
      case 'wings': {
        const cid = Math.floor(a.to / 3);
        const fr = ctx.freeRes.find((r) => r.node === a.to);
        const ca = ctx.carriers.find((c) => c.cell === cid);
        if (p.carry < 0 && fr && !(ctx.distAnt.has(cid))) v = 58;
        else if (mayAttack && ca && !ctx.distAnt.has(cid)) v = 56;
        else v = -10;
        break;
      }
      case 'blockage': v = enemyCarrier ? 21 : 6; break;
      case 'bridge': v = stuck && reconnects(s, rc, a) ? 80 : 24; break;
      case 'super': {
        v = 9;
        if (homeQ.length) { const s1 = after(s, a); if (legalActions(s1, false).some((o) => o.t === 'overlay' && cuts(after(s1, o)))) v = 75; }
        break;
      }
      case 'overlay': v = homeQ.length && cuts(after(s, a)) ? 95 : 12; break;
      case 'sdone': v = 14; break;
      case 'sundo': v = 0; break;
      case 'end': v = 16; break;
      default: v = 0;
    }
    out[i] = v;
  }
  return out;
}

export class HeuristicAgent {
  constructor(noise = 4) { this.name = 'Heuristik'; this.noise = noise; }
  act(s) {
    const acts = legalActions(s, false);
    if (acts.length === 1) return acts[0];
    const ctx = buildCtx(s, actor(s));
    const sc = heuristicScores(ctx, acts);
    let best = 0, bv = -1e9;
    for (let i = 0; i < acts.length; i++) {
      const v = sc[i] + (Math.random() - 0.5) * this.noise;
      if (v > bv) { bv = v; best = i; }
    }
    return acts[best];
  }
}

// ---- neural agent ----
export class NNAgent {
  constructor(net, { sample = true, temperature = 1, name = 'KI (neuronal)' } = {}) {
    this.net = net; this.sample = sample; this.temperature = temperature; this.name = name;
    this.recorder = null; // optional function(step)
  }
  // returns action; if `info` passed, fills probabilities for UI
  act(s, info = null) {
    let acts = legalActions(s, false);
    if (acts.length === 1) return acts[0];
    acts = capCandidates(acts);
    const pi = actor(s);
    const ctx = buildCtx(s, pi);
    const sf = stateFeatures(ctx);
    const A = candidateMatrix(ctx, acts);
    const ev = this.net.graph ? this.net.evaluate(ctx, acts, sf, A) : null;
    const logits = ev ? ev.logits : this.net.forwardPolicy(sf, A, acts.length).logits;
    const valueOf = () => (ev ? ev.v : this.net.forwardValue(sf));
    const p = softmax(logits, this.sample ? this.temperature : 1);
    let idx;
    if (this.sample) idx = sampleIdx(p);
    else { idx = 0; for (let i = 1; i < p.length; i++) if (p[i] > p[idx]) idx = i; }
    if (info) { info.acts = acts; info.probs = p; info.idx = idx; info.value = valueOf(); info.sf = sf; info.A = A; info.logits = logits; }
    if (this.recorder) {
      this.recorder({ player: pi, s: sf, A, n: acts.length, a: idx, logp: Math.log(Math.max(p[idx], 1e-8)), v: valueOf() });
    }
    return acts[idx];
  }
}

// search agent: network priors/values + PUCT tree search over determinized worlds
export class MCTSAgent {
  constructor(net, { sims = 64, timeMs = 0, tau = 0, maxDepth = 18, topK = 12, cpuct = 1.25, name = 'KI (Suche)' } = {}) {
    this.mcts = new MCTS(net, { sims: Math.max(8, sims), timeMs, maxDepth, topK, cpuct });
    this.tau = tau; this.name = name;
  }
  act(s, info = null) {
    const acts = legalActions(s, false);
    if (acts.length === 1) return acts[0];
    const r = this.mcts.search(s, { noise: 0 });
    const idx = pickByVisits(r.visits, this.tau);
    if (info) {
      const tot = r.visits.reduce((a, b) => a + b, 0) || 1;
      info.acts = r.acts; info.probs = Float32Array.from(r.visits, (v) => v / tot); info.idx = idx; info.value = r.value;
      const ctx = buildCtx(s, actor(s)); info.sf = stateFeatures(ctx); info.A = candidateMatrix(ctx, r.acts); info.search = true;
    }
    return r.acts[idx];
  }
}

export function makeAgent(kind, net) {
  if (kind === 'random') return new RandomAgent();
  if (kind === 'heuristic') return new HeuristicAgent();
  if (kind === 'nn' && net) return new NNAgent(net, { sample: true, temperature: 0.6 });
  return new HeuristicAgent();
}
