// Regression: after loading a resource the ant must be able to leave through any path of its cell.
import { newGame, applyAction, legalActions, actor, bfs, antStarts, hillCells, makeCellShape } from '../js/engine.js';
import { cellId, neighborId } from '../js/geometry.js';
let found = 0, bad = 0;
for (let seed = 1; seed <= 200 && found < 20; seed++) {
  const s = newGame({ numPlayers: 2, seed });
  let g = 0;
  while (!s.over && g++ < 2000) {
    const pi = actor(s);
    const la = legalActions(s, true);
    const load = la.find((a) => a.t === 'load');
    if (load && !s.pending) {
      const p = s.players[pi];
      const before = bfs(s, pi, [s.players[pi].ant.cell * 3 + 0]).size; // loose reference
      applyAction(s, load);
      // every open path out of the cell must now be usable as a start
      const c = s.cells[p.ant.cell];
      const starts = antStarts(s, pi);
      const reach = bfs(s, pi, starts);
      found++;
      for (const e of c.allOpen) {
        const [nid, ne] = neighborId(c.id, e);
        const nc = s.cells[nid];
        if (nc && nc.open[ne] && nc.block < 0) { let ok = false; for (const [n] of reach) if (Math.floor(n / 3) === nid) ok = true; if (!ok) bad++; }
      }
      continue;
    }
    const nonEnd = la.filter((a) => a.t !== 'end');
    const a = Math.random() < 0.15 || !nonEnd.length ? la[la.length - 1] : nonEnd[Math.floor(Math.random() * nonEnd.length)];
    applyAction(s, a);
  }
}
console.log({ loadsChecked: found, unreachableExits: bad });
process.exit(bad ? 1 : 0);
