import { HeuristicAgent, RandomAgent } from '../js/agents.js';
import { playGame } from '../js/selfplay.js';
const N = +(process.argv[2] || 20);
let w = [0, 0], d = 0, t0 = Date.now(), turns = 0;
for (let g = 0; g < N; g++) {
  const hSeat = g % 2;
  const ag = [0, 1].map((i) => (i === hSeat ? new HeuristicAgent() : new RandomAgent()));
  const s = playGame(ag);
  turns += s.turnNo;
  if (s.winner < 0) d++; else w[s.winner === hSeat ? 0 : 1]++;
}
console.log('heuristic wins', w[0], 'random wins', w[1], 'draws', d, 'avgTurns', turns / N, 'sec', (Date.now() - t0) / 1000);
