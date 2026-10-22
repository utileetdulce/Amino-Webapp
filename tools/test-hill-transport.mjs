// Regression: transport while standing in the own hill stores the resource.
import { newGame, applyAction, legalActions, score } from '../js/engine.js';
let ok = 0, bad = 0;
for (let seed = 1; seed < 300 && ok < 10; seed++) {
  const s = newGame({ numPlayers: 2, seed });
  const pi = s.cur, p = s.players[pi];
  p.hand = Array(12).fill(0); p.hillsInHand = 2;
  const place = (piece) => { const a = legalActions(s, true).find((x) => x.t === 'place' && x.tile === 0 && x.piece === piece); if (a) applyAction(s, a); return !!a; };
  if (!place(1)) continue;
  for (let i = 0; i < 4; i++) place(0);
  if (!place(2)) continue;
  const mv = legalActions(s, true).find((x) => x.t === 'move' && s.cells[Math.floor(x.to / 3)].res >= 0);
  if (!mv) continue;
  applyAction(s, mv); applyAction(s, { t: 'load' });
  // fast-forward: back to own hill, next own turn
  applyAction(s, { t: 'end' }); applyAction(s, { t: 'end' });
  const back = legalActions(s, true).find((x) => x.t === 'move' && s.cells[Math.floor(x.to / 3)].hill === pi);
  if (!back) continue;
  applyAction(s, back);
  const t = legalActions(s, true).find((x) => x.t === 'transport' && !x.card);
  if (!t) continue;
  const before = score(s, pi);
  applyAction(s, t);
  if (score(s, pi) > before && p.carry === -1 && !s.pending) ok++; else bad++;
}
console.log({ delivered: ok, failed: bad });
process.exit(bad || !ok ? 1 : 0);
