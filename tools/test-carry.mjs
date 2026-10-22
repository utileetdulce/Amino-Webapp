// House rule: while an ant carries a resource there is NO plain move action – only transport (dice) moves it.
import { newGame, applyAction, actor, legalActions } from '../js/engine.js';
import { HeuristicAgent, RandomAgent } from '../js/agents.js';
let carryingDecisions = 0, illegalMoves = 0, transportMoves = 0;
for (let g = 0; g < 120; g++) {
  const s = newGame({ numPlayers: 2, seed: 900 + g });
  const ag = g % 2 ? [new RandomAgent(), new RandomAgent()] : [new HeuristicAgent(), new HeuristicAgent()];
  let n = 0;
  while (!s.over && n++ < 3000) {
    const pi = actor(s);
    if (!s.pending && s.players[pi].carry >= 0) {
      carryingDecisions++;
      for (const full of [false, true]) if (legalActions(s, full).some((a) => a.t === 'move')) illegalMoves++;
    }
    const a = ag[pi].act(s);
    if (a.t === 'tmove') transportMoves++;
    applyAction(s, a);
  }
}
console.log({ carryingDecisions, illegalMoves, transportMoves });
process.exit(illegalMoves || !transportMoves ? 1 : 0);
