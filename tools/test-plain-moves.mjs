// AI candidates (non-full): after a plain ant move (not hill-to-hill) no second plain move until another action happened.
import { newGame, applyAction, legalActions, actor, nodeCell } from '../js/engine.js';
import { HeuristicAgent, RandomAgent } from '../js/agents.js';
let games = 0, plainStreakViolations = 0, tunnelAfterPlain = 0, moves = 0, maxChain = 0;
for (let seed = 1; seed <= 60; seed++) {
  const s = newGame({ numPlayers: 2, seed });
  const ag = [new RandomAgent(), seed % 2 ? new HeuristicAgent() : new RandomAgent()];
  let guard = 0, plain = 0, chain = 0;
  while (!s.over && guard++ < 4000) {
    const pi = actor(s);
    const acts = legalActions(s, false);
    const a = acts.length === 1 ? acts[0] : ag[pi].act(s);
    if (a.t === 'move') {
      const p = s.players[pi], from = p.ant.cell, to = nodeCell(a.to);
      const tunnel = from !== to && s.cells[from].hill === pi && s.cells[to].hill === pi;
      if (tunnel) { if (plain > 0) tunnelAfterPlain++; }
      else { if (plain > 0) plainStreakViolations++; plain++; }
      moves++; chain++; maxChain = Math.max(maxChain, chain);
    } else if (a.t !== 'end') { plain = 0; chain = 0; }
    applyAction(s, a);
    if (a.t === 'end') { plain = 0; chain = 0; }
  }
  games++;
}
console.log({ games, moves, plainStreakViolations, tunnelAfterPlain, maxChain });
if (plainStreakViolations) { console.error('FAIL: two plain moves in a row were offered'); process.exit(1); }
console.log('ok');
