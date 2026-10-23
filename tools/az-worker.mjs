// worker_threads worker for tools/train-az.mjs
import { parentPort } from 'node:worker_threads';
import { Net } from '../js/nn.js';
import { selfPlayGame } from '../js/az.js';
import { MCTSAgent, NNAgent, HeuristicAgent } from '../js/agents.js';
import { playGame } from '../js/selfplay.js';

parentPort.on('message', (m) => {
  const net = new Net(m.arch, m.theta);
  if (m.type === 'play') {
    const samples = []; let w = 0, l = 0, d = 0, turns = 0;
    for (let g = 0; g < m.games; g++) {
      const heur = Math.random() < m.oppMix;
      const r = selfPlayGame(net, { ...m.cfg, oppHeuristic: heur });
      for (const x of r.samples) samples.push(x);
      turns += r.turns;
      if (heur) { if (r.winner < 0) d++; else if (r.winner === r.learner) w++; else l++; }
    }
    parentPort.postMessage({ type: 'samples', samples, games: m.games, turns, vsHeur: { w, d, l } });
  } else if (m.type === 'eval') {
    let w = 0, d = 0, l = 0;
    for (let g = 0; g < m.games; g++) {
      const seat = (m.offset + g) % 2;
      const me = m.search ? new MCTSAgent(net, { sims: m.sims, maxDepth: m.cfg.depth, topK: m.cfg.topK, cpuct: m.cfg.cpuct }) : new NNAgent(net, { sample: true, temperature: 0.5 });
      const s = playGame([0, 1].map((i) => (i === seat ? me : new HeuristicAgent())), {});
      if (s.winner < 0) d++; else if (s.winner === seat) w++; else l++;
    }
    parentPort.postMessage({ type: 'eval', w, d, l, search: !!m.search });
  }
});
