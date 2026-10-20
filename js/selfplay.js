// Episode runner + PPO trainer (used by the browser worker and the node CLI).
import { newGame, applyAction, actor, score, legalActions } from './engine.js';
import { NNAgent, HeuristicAgent, RandomAgent, capCandidates } from './agents.js';
import { buildCtx, stateFeatures, candidateMatrix, SF, AF } from './features.js';
import { Net, softmax } from './nn.js';

// Plays one game. agents[i] decides for seat i. Returns the final state.
export function playGame(agents, { seed, numPlayers = agents.length, maxSteps = 6000, onState } = {}) {
  const s = newGame({ numPlayers, seed });
  let guard = 0;
  while (!s.over && guard++ < maxSteps) {
    const pi = actor(s);
    const a = agents[pi].act(s);
    applyAction(s, a);
    if (onState) onState(s, a, pi);
  }
  if (!s.over) { s.over = true; s.winner = -1; }
  return s;
}

// Collects PPO trajectories. learnerSeats: seats driven by `net` (recorded); others by opponent agents.
export function collectEpisode(net, { opponent = 'self', shape = 0.1, numPlayers = 2, temperature = 1, oppNet = null }) {
  const traj = [[], []];
  const learners = [];
  const agents = [];
  let learnerSeats;
  const mk = (seat) => {
    const ag = new NNAgent(net, { sample: true, temperature });
    ag.recorder = (st) => { st.shapeScore = 0; traj[seat].push(st); };
    return ag;
  };
  const seatsAll = Array.from({ length: numPlayers }, (_, i) => i);
  if (opponent === 'self') learnerSeats = seatsAll;
  else learnerSeats = [Math.floor(Math.random() * numPlayers)];
  for (const i of seatsAll) {
    if (learnerSeats.includes(i)) agents.push(mk(i));
    else if (opponent === 'heuristic') agents.push(new HeuristicAgent());
    else if (opponent === 'random') agents.push(new RandomAgent());
    else if (opponent === 'past' && oppNet) agents.push(new NNAgent(oppNet, { sample: true, temperature: 1 }));
    else agents.push(new HeuristicAgent());
  }
  for (let i = 0; i < numPlayers; i++) traj[i] = [];
  // track score at each recorded step for shaping
  const s = newGame({ numPlayers });
  let guard = 0;
  const scoreAtStep = Array.from({ length: numPlayers }, () => []);
  while (!s.over && guard++ < 6000) {
    const pi = actor(s);
    const before = traj[pi].length;
    const a = agents[pi].act(s);
    if (traj[pi].length > before) scoreAtStep[pi].push(score(s, pi) - Math.max(0, ...s.players.map((_, q) => (q === pi ? 0 : score(s, q)))) * 0);
    applyAction(s, a);
  }
  if (!s.over) { s.over = true; s.winner = -1; }
  const steps = [];
  let nSteps = 0;
  for (const seat of learnerSeats) {
    const T = traj[seat];
    if (!T.length) continue;
    // shaping: own score delta between consecutive recorded steps (approx: use scores recorded)
    for (let i = 0; i < T.length; i++) T[i].r = 0;
    let prev = 0;
    for (let i = 0; i < T.length; i++) {
      const sc = scoreAtStep[seat][i];
      T[i].r = shape * (sc - prev) / 7;
      prev = sc;
    }
    const finalScore = score(s, seat);
    T[T.length - 1].r += shape * (finalScore - prev) / 7;
    const win = s.winner < 0 ? 0 : s.winner === seat ? 1 : -1;
    T[T.length - 1].r += win;
    T[T.length - 1].done = true;
    for (const t of T) steps.push(t);
    nSteps += T.length;
  }
  return { steps, trajs: learnerSeats.map((q) => traj[q]), winner: s.winner, learnerSeats, turns: s.turnNo, scores: s.players.map((_, i) => score(s, i)), over: s };
}

export function computeGAE(T, gamma, lam) {
  let adv = 0;
  for (let i = T.length - 1; i >= 0; i--) {
    const nextV = i === T.length - 1 ? 0 : T[i + 1].v;
    const delta = T[i].r + gamma * nextV - T[i].v;
    adv = delta + gamma * lam * adv;
    T[i].adv = adv;
    T[i].ret = adv + T[i].v;
  }
}

export const DEFAULT_PPO = { lr: 3e-4, gamma: 0.995, lam: 0.95, clip: 0.2, epochs: 4, mb: 96, entCoef: 0.01, vfCoef: 0.5, shape: 0.1 };

export function ppoUpdate(net, steps, cfg = DEFAULT_PPO) {
  // normalise advantages
  let mean = 0, sq = 0;
  for (const t of steps) mean += t.adv;
  mean /= steps.length;
  for (const t of steps) sq += (t.adv - mean) ** 2;
  const sd = Math.sqrt(sq / steps.length) + 1e-6;
  const stats = { pl: 0, vl: 0, ent: 0, kl: 0, clipFrac: 0, gn: 0, cnt: 0 };
  const idx = steps.map((_, i) => i);
  for (let ep = 0; ep < cfg.epochs; ep++) {
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    for (let b = 0; b < idx.length; b += cfg.mb) {
      const batch = idx.slice(b, b + cfg.mb);
      net.zeroGrad();
      for (const k of batch) {
        const t = steps[k];
        const A = (t.adv - mean) / sd;
        const fp = net.forwardPolicy(t.s, t.A, t.n, true);
        const p = softmax(fp.logits);
        const lp = Math.log(Math.max(p[t.a], 1e-8));
        const ratio = Math.exp(lp - t.logp);
        const clipped = (A >= 0 && ratio > 1 + cfg.clip) || (A < 0 && ratio < 1 - cfg.clip);
        let ent = 0;
        for (let c = 0; c < t.n; c++) if (p[c] > 1e-8) ent -= p[c] * Math.log(p[c]);
        const dlog = new Float32Array(t.n);
        const scale = 1 / batch.length;
        for (let c = 0; c < t.n; c++) {
          let g = 0;
          if (!clipped) g += -A * ratio * ((c === t.a ? 1 : 0) - p[c]);
          // entropy bonus gradient: d(-ent)/dlogit_c = p_c (log p_c + ent)
          g += cfg.entCoef * p[c] * (Math.log(Math.max(p[c], 1e-8)) + ent);
          dlog[c] = g * scale;
        }
        const vc = net.forwardValue(t.s, true);
        const dv = cfg.vfCoef * 2 * (vc.v - t.ret) * scale;
        net.backward(t.s, t.A, t.n, fp, dlog, vc, dv);
        stats.pl += -Math.min(A * ratio, A * Math.max(1 - cfg.clip, Math.min(1 + cfg.clip, ratio)));
        stats.vl += (vc.v - t.ret) ** 2;
        stats.ent += ent;
        stats.kl += t.logp - lp;
        stats.clipFrac += clipped ? 1 : 0;
        stats.cnt++;
      }
      stats.gn += net.adamStep(cfg.lr);
    }
  }
  const c = stats.cnt || 1;
  return { policyLoss: stats.pl / c, valueLoss: stats.vl / c, entropy: stats.ent / c, kl: stats.kl / c, clipFrac: stats.clipFrac / c, gradNorm: stats.gn / Math.ceil(c / cfg.mb) };
}

// Behaviour cloning of the heuristic bot (warm start)
import { heuristicScores } from './agents.js';
export function collectHeuristicDemos(nGames = 8, numPlayers = 2) {
  const demos = [];
  for (let g = 0; g < nGames; g++) {
    const s = newGame({ numPlayers });
    const h = new HeuristicAgent(3);
    let guard = 0;
    while (!s.over && guard++ < 4000) {
      const pi = actor(s);
      let acts = legalActions(s, false);
      if (acts.length > 1) {
        acts = capCandidates(acts);
        const ctx = buildCtx(s, pi);
        const sc = heuristicScores(ctx, acts);
        // soft target: softmax of heuristic scores (temperature 6) so unimodal ties spread
        const tgt = softmax(Float32Array.from(sc), 6);
        demos.push({ s: stateFeatures(ctx), A: candidateMatrix(ctx, acts), n: acts.length, tgt });
      }
      // follow heuristic with noise to diversify states
      let a;
      if (acts.length === 1) a = acts[0];
      else {
        const ctx = buildCtx(s, pi);
        const sc = heuristicScores(ctx, acts);
        let best = 0, bv = -1e9;
        for (let i = 0; i < acts.length; i++) { const v = sc[i] + (Math.random() - 0.5) * 12; if (v > bv) { bv = v; best = i; } }
        a = acts[best];
      }
      applyAction(s, a);
    }
  }
  return demos;
}

export function cloneUpdate(net, demos, { epochs = 3, mb = 128, lr = 1e-3 } = {}) {
  let loss = 0, cnt = 0;
  const idx = demos.map((_, i) => i);
  for (let ep = 0; ep < epochs; ep++) {
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    for (let b = 0; b < idx.length; b += mb) {
      const batch = idx.slice(b, b + mb);
      net.zeroGrad();
      for (const k of batch) {
        const d = demos[k];
        const fp = net.forwardPolicy(d.s, d.A, d.n, true);
        const p = softmax(fp.logits);
        const dlog = new Float32Array(d.n);
        for (let c = 0; c < d.n; c++) { dlog[c] = (p[c] - d.tgt[c]) / batch.length; if (d.tgt[c] > 1e-8) loss -= d.tgt[c] * Math.log(Math.max(p[c], 1e-8)); }
        net.backward(d.s, d.A, d.n, fp, dlog, null, 0);
        cnt++;
      }
      net.adamStep(lr, 2.0);
    }
  }
  return loss / Math.max(1, cnt);
}

// Evaluate net (greedy-ish) against an opponent kind; returns win rate with draws as 0.5
export function evaluate(net, oppKind, games = 20, numPlayers = 2) {
  let w = 0, d = 0, l = 0;
  for (let g = 0; g < games; g++) {
    const seat = g % numPlayers;
    const agents = [];
    for (let i = 0; i < numPlayers; i++) {
      if (i === seat) agents.push(new NNAgent(net, { sample: true, temperature: 0.5 }));
      else agents.push(oppKind === 'random' ? new RandomAgent() : new HeuristicAgent());
    }
    const s = playGame(agents, { numPlayers });
    if (s.winner < 0) d++; else if (s.winner === seat) w++; else l++;
  }
  return { win: w, draw: d, loss: l, rate: (w + 0.5 * d) / games };
}
