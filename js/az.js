// AlphaZero-style training pieces: self-play with MCTS targets and the supervised update (policy = visit counts, value = outcome).
import { newGame, applyAction, actor, legalActions } from './engine.js';
import { MCTS, pickByVisits } from './mcts.js';
import { HeuristicAgent } from './agents.js';
import { buildCtx, stateFeatures, candidateMatrix } from './features.js';
import { softmax } from './nn.js';

// Plays one game; returns training samples [{s, A, n, tgt, z}] and the result.
export function selfPlayGame(net, { sims = 64, tauMoves = 30, numPlayers = 2, oppHeuristic = false, dirichlet = 0.25, depth = 18, maxDecisions = 3000 } = {}) {
  const s = newGame({ numPlayers });
  const mcts = new MCTS(net, { sims, dirichlet, maxDepth: depth });
  const heur = new HeuristicAgent();
  const learner = oppHeuristic ? Math.floor(Math.random() * numPlayers) : -1;
  const samples = [];
  let searched = 0, guard = 0;
  while (!s.over && guard++ < maxDecisions) {
    const pi = actor(s);
    const acts = legalActions(s, false);
    if (acts.length === 1) { applyAction(s, acts[0]); continue; }
    if (oppHeuristic && pi !== learner) { applyAction(s, heur.act(s)); continue; }
    const r = mcts.search(s, { sims, noise: dirichlet });
    const tot = r.visits.reduce((a, b) => a + b, 0) || 1;
    const ctx = buildCtx(s, pi);
    samples.push({ pi, s: stateFeatures(ctx), A: candidateMatrix(ctx, r.acts), n: r.acts.length, tgt: Float32Array.from(r.visits, (v) => v / tot) });
    const idx = pickByVisits(r.visits, searched++ < tauMoves ? 1 : 0.1);
    applyAction(s, r.acts[idx]);
  }
  if (!s.over) { s.over = true; s.winner = -1; }
  for (const x of samples) x.z = s.winner < 0 ? 0 : s.winner === x.pi ? 1 : -1;
  return { samples, winner: s.winner, turns: s.turnNo, learner };
}

// One round of minibatch SGD on the replay buffer.
export function azUpdate(net, buffer, { steps = 150, batch = 128, lr = 3e-4, vCoef = 1 } = {}) {
  let pl = 0, vl = 0, cnt = 0;
  for (let st = 0; st < steps; st++) {
    net.zeroGrad();
    for (let b = 0; b < batch; b++) {
      const d = buffer[Math.floor(Math.random() * buffer.length)];
      const fp = net.forwardPolicy(d.s, d.A, d.n, true);
      const p = softmax(fp.logits);
      const dlog = new Float32Array(d.n);
      for (let c = 0; c < d.n; c++) {
        dlog[c] = (p[c] - d.tgt[c]) / batch;
        if (d.tgt[c] > 1e-8) pl -= d.tgt[c] * Math.log(Math.max(p[c], 1e-8));
      }
      const vc = net.forwardValue(d.s, true);
      const err = vc.v - d.z;
      vl += err * err;
      net.backward(d.s, d.A, d.n, fp, dlog, vc, vCoef * 2 * err / batch);
      cnt++;
    }
    net.adamStep(lr, 1.0);
  }
  return { policyLoss: pl / cnt, valueLoss: vl / cnt };
}
