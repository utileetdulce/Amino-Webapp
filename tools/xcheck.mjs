// Deterministic trace (heuristic, noise 0) for cross-checking the Rust trainer's engine/features/net port:
//   node tools/xcheck.mjs models/pretrained.json 30   vs   rust-trainer/target/release/amino-train xcheck models/pretrained.json 30
import fs from 'node:fs';
import { newGame, applyAction, actor, legalActions } from '../js/engine.js';
import { buildCtx, stateFeatures, candidateMatrix, ACTION_TYPES } from '../js/features.js';
import { heuristicScores } from '../js/agents.js';
import { Net } from '../js/nn.js';
import { netFromJSON } from '../js/gnet.js';

const net = netFromJSON(JSON.parse(fs.readFileSync(process.argv[2], 'utf8')).net, Net);
const N = +process.argv[3];
for (let seed = 1; seed <= N; seed++) {
  const s = newGame({ numPlayers: 2, seed });
  let h = 0, steps = 0, guard = 0, sfs = 0, asum = 0, lsum = 0, vsum = 0;
  while (!s.over && guard++ < 6000) {
    const pi = actor(s);
    const acts = legalActions(s, false);
    let a;
    if (acts.length === 1) a = acts[0];
    else {
      const ctx = buildCtx(s, pi);
      const sc = heuristicScores(ctx, acts);
      let best = 0, bv = -1e9;
      for (let i = 0; i < acts.length; i++) if (sc[i] > bv) { bv = sc[i]; best = i; }
      a = acts[best];
      const sf = stateFeatures(ctx), m = candidateMatrix(ctx, acts);
      sfs += sf.reduce((x, y) => x + y, 0); asum += m.reduce((x, y) => x + y, 0);
      if (net.graph) { const r = net.evaluate(ctx, acts, sf, m); lsum += r.logits.reduce((x, y) => x + y, 0); vsum += r.v; }
      else { lsum += net.forwardPolicy(sf, m, acts.length).logits.reduce((x, y) => x + y, 0); vsum += net.forwardValue(sf); }
    }
    h = (Math.imul(h, 31) + acts.length * 7 + ACTION_TYPES.indexOf(a.t)) >>> 0;
    steps++;
    applyAction(s, a);
  }
  const w = s.over ? s.winner : -1;
  console.log(seed, w, s.turnNo, steps, h, sfs.toFixed(2), asum.toFixed(2), lsum.toFixed(3), vsum.toFixed(3));
}
