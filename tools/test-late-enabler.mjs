// AI candidates (non-full): once a path-preparing action (bridge / overlay / tile) made a target reachable AFTER the ant already
// moved this turn, no plain ant move is offered any more (the action should have been played before the move).
import { newGame, applyAction, legalActions, actor, nodeCell, targetReach } from '../js/engine.js';
import { HeuristicAgent, RandomAgent } from '../js/agents.js';
let late = 0, offeredAfterLate = 0, turnsWithLate = 0, ordered = 0;
for (let seed = 1; seed <= 80; seed++) {
  const s = newGame({ numPlayers: 2, seed });
  const ag = [new RandomAgent(), seed % 2 ? new HeuristicAgent() : new RandomAgent()];
  let guard = 0, moved = false, lateNow = false;
  while (!s.over && guard++ < 4000) {
    const pi = actor(s);
    const acts = legalActions(s, false);
    if (lateNow) {
      const p = s.players[pi];
      const plain = acts.filter((a) => a.t === 'move' && !(s.cells[p.ant.cell].hill === pi && s.cells[nodeCell(a.to)].hill === pi));
      offeredAfterLate += plain.length;
    }
    const a = acts.length === 1 ? acts[0] : ag[pi].act(s);
    const before = moved && ['bridge', 'place', 'overlay'].includes(a.t) ? targetReach(s, pi) : null;
    if (a.t === 'move') { const p = s.players[pi], f = p.ant.cell, t = nodeCell(a.to); if (!(f !== t && s.cells[f].hill === pi && s.cells[t].hill === pi)) moved = true; }
    applyAction(s, a);
    if (before) { const after = targetReach(s, pi); if (after) for (const [c, was] of before) if (!was && after.get(c)) { if (!lateNow) { late++; } lateNow = true; break; } }
    if (a.t === 'end') { moved = false; lateNow = false; }
  }
}
console.log({ lateEnablerEvents: late, plainMovesOfferedAfterLate: offeredAfterLate });
if (offeredAfterLate) { console.error('FAIL'); process.exit(1); }
console.log('ok');
