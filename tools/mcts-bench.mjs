// node tools/mcts-bench.mjs [model.json] [sims] [games]
import fs from 'node:fs';
import { Net } from '../js/nn.js';
import { netFromJSON } from '../js/gnet.js';
import { MCTSAgent, NNAgent, HeuristicAgent } from '../js/agents.js';
import { playGame } from '../js/selfplay.js';
const file = process.argv[2] || 'models/pretrained.json';
const sims = +(process.argv[3] || 24), games = +(process.argv[4] || 4);
const net = netFromJSON(JSON.parse(fs.readFileSync(file, 'utf8')).net, Net);
for (const [label, mk] of [['MCTS', () => new MCTSAgent(net, { sims })], ['raw net', () => new NNAgent(net, { sample: true, temperature: 0.5 })]]) {
  let w = 0, d = 0, l = 0; const t0 = Date.now(); let turns = 0;
  for (let g = 0; g < games; g++) {
    const seat = g % 2;
    const s = playGame([0, 1].map((i) => (i === seat ? mk() : new HeuristicAgent())), {});
    turns += s.turnNo;
    if (s.winner < 0) d++; else if (s.winner === seat) w++; else l++;
  }
  console.log(label, { sims: label === 'MCTS' ? sims : '-', w, d, l, sec: ((Date.now() - t0) / 1000).toFixed(1), avgTurns: (turns / games).toFixed(0) });
}
