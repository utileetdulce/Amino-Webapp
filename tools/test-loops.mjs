// Regression: AI candidate generation never offers pointless repetition
// (stepping back onto a node visited this turn without another action in between; load<->unload undo pairs).
import { newGame, applyAction, actor, legalActions, antNode } from '../js/engine.js';
import { RandomAgent, HeuristicAgent } from '../js/agents.js';
let moves = 0, loops = 0, undo = 0;
for (let g = 0; g < 150; g++) {
  const s = newGame({ numPlayers: 2, seed: 321 + g });
  const ag = g % 2 ? [new RandomAgent(), new RandomAgent()] : [new HeuristicAgent(), new RandomAgent()];
  let n = 0, trail = [], lastT = null, turn = s.turnNo;
  while (!s.over && n++ < 4000) {
    if (s.turnNo !== turn) { turn = s.turnNo; trail = []; lastT = null; }
    const pi = actor(s);
    const a = ag[pi].act(s);
    if (a.t === 'move') {
      moves++;
      if (!trail.length) trail.push(antNode(s, pi));
      if (trail.includes(a.to)) loops++;
      trail.push(a.to);
    } else trail = [];
    if ((a.t === 'load' && lastT === 'unload') || (a.t === 'unload' && lastT === 'load')) undo++;
    lastT = a.t;
    applyAction(s, a);
  }
}
console.log({ moves, loops, undo });
process.exit(loops || undo || !moves ? 1 : 0);
