// Regression: after taking back a hill the ant must be able to leave through any path of its cell.
import { newGame, applyAction, legalActions, actor, bfs, antStarts } from '../js/engine.js';
import { neighborId } from '../js/geometry.js';
let found = 0, bad = 0;
for (let seed = 1; seed <= 400 && found < 20; seed++) {
  const s = newGame({ numPlayers: 2, seed });
  let g = 0;
  while (!s.over && g++ < 2000) {
    const pi = actor(s);
    const la = legalActions(s, true);
    const p = s.players[pi];
    const back = la.find((a) => a.t === 'inc_hill' && p.ant && a.cell === p.ant.cell);
    if (back && !s.pending) {
      applyAction(s, back);
      const c = s.cells[p.ant.cell];
      const reach = bfs(s, pi, antStarts(s, pi));
      found++;
      for (const e of c.allOpen) {
        const [nid, ne] = neighborId(c.id, e);
        const nc = s.cells[nid];
        if (nc && nc.open[ne] && nc.block < 0) { let ok = false; for (const [n] of reach) if (Math.floor(n / 3) === nid) ok = true; if (!ok) bad++; }
      }
      continue;
    }
    const nonEnd = la.filter((a) => a.t !== 'end' && a.t !== 'inc_hill');
    const a = Math.random() < 0.15 || !nonEnd.length ? la[la.length - 1] : nonEnd[Math.floor(Math.random() * nonEnd.length)];
    applyAction(s, a);
  }
}
console.log({ hillsTakenBack: found, unreachableExits: bad });
process.exit(bad || !found ? 1 : 0);
