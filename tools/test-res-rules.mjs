// Rules v6: a resource lies on one path piece (loading needs the ant on that piece, loading/unloading never changes the
// path graph), and a hill keeps 2 triangles distance to every resource, also to one being carried.
import { newGame, applyAction, legalActions, actor, resNode, antNode, compsOf } from '../js/engine.js';
import { gridDistWithin } from '../js/geometry.js';
let loads = 0, hills = 0, split = 0, bad = [];
const sig = (s) => Object.values(s.cells).map((c) => c.id + ':' + compsOf(c).map((g) => g.join('')).join(',')).join(';');
for (let seed = 1; seed <= 120; seed++) {
  const s = newGame({ numPlayers: 2 + (seed % 2), seed });
  let g = 0;
  while (!s.over && g++ < 3000) {
    const pi = actor(s), p = s.players[pi];
    const la = legalActions(s, seed % 3 !== 0);
    for (const a of la) {
      if (a.t === 'load') {
        const c = s.cells[p.ant.cell];
        if (resNode(c) !== antNode(s, pi)) bad.push(['load off piece', seed]);
      }
      if (a.t === 'place' && a.piece === 1) {
        hills++;
        const at = new Set();
        for (const r of s.resources) if (r.owner < 0) { if (r.cell >= 0) at.add(r.cell); else if (r.carrier >= 0) at.add(s.players[r.carrier].ant.cell); }
        if (gridDistWithin(a.cell, 2, (c) => at.has(c))) bad.push(['hill too close', seed, a.cell]);
      }
    }
    const nonEnd = la.filter((a) => a.t !== 'end');
    const pick = la.find((a) => a.t === 'load') || la.find((a) => a.t === 'unload' && Math.random() < 0.3);
    const a = pick || (Math.random() < 0.12 || !nonEnd.length ? la[la.length - 1] : nonEnd[Math.floor(Math.random() * nonEnd.length)]);
    if (a.t === 'load' || a.t === 'unload') {
      const before = sig(s);
      const c = s.cells[p.ant.cell];
      if (c.hill < 0 && c.comps.length > 1) split++;
      applyAction(s, a); loads++;
      if (sig(s) !== before) bad.push(['paths changed by ' + a.t, seed]);
      if (a.t === 'unload' && resNode(c) !== antNode(s, pi)) bad.push(['unload off piece', seed]);
    } else applyAction(s, a);
  }
}
console.log({ loadsUnloads: loads, onSplitTiles: split, hillCandidates: hills, bad: bad.slice(0, 10), nbad: bad.length });
process.exit(bad.length || !split ? 1 : 0);
