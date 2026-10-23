// Agents: random, heuristic, neural. All implement act(state) -> action (for state.pending actor or current player).
import { legalActions, actor, hasCard } from './engine.js';
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
  for (let i = 0; i < acts.length; i++) {
    const a = acts[i];
    let v = 0;
    switch (a.t) {
      case 'tmove': v = s.cells[Math.floor(a.to / 3)].hill === ctx.pi ? 100 : 50; break;
      case 'tstay': v = 1; break;
      case 'transport': v = a.card ? (ctx.curHillDist > 6 ? 60 : -50) : 70; break;
      case 'fight': v = 68; break;
      case 'load': v = 80; break;
      case 'unload': v = 0; break;
      case 'hide': case 'guess': v = 0; break;
      case 'revenge': v = 90; break;
      case 'decline': v = 0; break;
      case 'move': {
        const cid = Math.floor(a.to / 3);
        const fr = ctx.freeRes.find((r) => r.cell === cid);
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
        if (a.piece === 1) v = hillsPlaced === 0 ? 90 : (ctx.freeRes.length ? 30 + 6 * (1 - toRes) : 22);
        else if (a.piece === 2) v = p.ant ? 52 + 14 * reach - 10 * dAnt : 0;
        else v = (handN > 2 ? 22 : 12) + 6 * reach + 6 * (1 - toRes);
        break;
      }
      case 'inc_tiles': v = handN <= 1 ? 68 : handN <= 3 ? 40 : 28; break;
      case 'inc_card': v = p.cards.length === 0 ? 48 : 30; break;
      case 'inc_hill': v = hillsPlaced === 2 && p.ant && ctx.freeRes.length === 0 ? 8 : -5; break;
      case 'swap': v = 4; break;
      case 'wings': {
        const cid = Math.floor(a.to / 3);
        const fr = ctx.freeRes.find((r) => r.cell === cid);
        const ca = ctx.carriers.find((c) => c.cell === cid);
        if (p.carry < 0 && fr && !(ctx.distAnt.has(cid))) v = 58;
        else if (mayAttack && ca && !ctx.distAnt.has(cid)) v = 56;
        else v = -10;
        break;
      }
      case 'blockage': v = enemyCarrier ? 21 : 6; break;
      case 'bridge': v = 24; break;
      case 'super': v = 9; break;
      case 'overlay': v = 12; break;
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
    const { logits } = this.net.forwardPolicy(sf, A, acts.length);
    const p = softmax(logits, this.sample ? this.temperature : 1);
    let idx;
    if (this.sample) idx = sampleIdx(p);
    else { idx = 0; for (let i = 1; i < p.length; i++) if (p[i] > p[idx]) idx = i; }
    if (info) { info.acts = acts; info.probs = p; info.idx = idx; info.value = this.net.forwardValue(sf); info.sf = sf; info.A = A; info.logits = logits; }
    if (this.recorder) {
      this.recorder({ player: pi, s: sf, A, n: acts.length, a: idx, logp: Math.log(Math.max(p[idx], 1e-8)), v: this.net.forwardValue(sf) });
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
