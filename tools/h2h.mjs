// Head-to-head between two models: node tools/h2h.mjs A.json B.json [games] [sims(0=raw net)]
import fs from 'node:fs';
import { Net } from '../js/nn.js';
import { netFromJSON } from '../js/gnet.js';
import { MCTSAgent, NNAgent } from '../js/agents.js';
import { playGame } from '../js/selfplay.js';
const [fa, fb] = [process.argv[2], process.argv[3]];
const games = +(process.argv[4] || 100), sims = +(process.argv[5] || 0);
const A = netFromJSON(JSON.parse(fs.readFileSync(fa, 'utf8')).net, Net), B = netFromJSON(JSON.parse(fs.readFileSync(fb, 'utf8')).net, Net);
const mk = (net) => (sims ? new MCTSAgent(net, { sims }) : new NNAgent(net, { sample: true, temperature: 0.5 }));
let wa = 0, wb = 0, d = 0, turns = 0;
for (let g = 0; g < games; g++) {
  const seatA = g % 2;
  const s = playGame([0, 1].map((i) => (i === seatA ? mk(A) : mk(B))), {});
  turns += s.turnNo;
  if (s.winner < 0) d++; else if (s.winner === seatA) wa++; else wb++;
}
console.log({ A: fa.split('/').pop(), B: fb.split('/').pop(), games, sims: sims || 'raw', A_wins: wa, B_wins: wb, draws: d, A_rate: ((wa + d / 2) / games).toFixed(2), avgTurns: (turns / games).toFixed(0) });
