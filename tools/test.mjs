import { newGame, legalActions, applyAction, actor, score, cloneState, viewFor, consistent } from '../js/engine.js';
let seed = +(process.argv[2] || 1);
const N = +(process.argv[3] || 50);
let done = 0, wins = [0, 0, 0, 0, 0, 0], turns = 0, acts = 0, draws = 0;
const types = {};
for (let g = 0; g < N; g++) {
  const n = 2 + (g % 3);
  const s = newGame({ numPlayers: n, seed: seed++ });
  let guard = 0;
  while (!s.over && guard++ < 20000) {
    const full = g % 2 === 0;
    const la = legalActions(s, full);
    if (!la.length) throw new Error('no legal actions');
    // random with end bias
    let a = la[Math.floor(Math.random() * la.length)];
    if (Math.random() < 0.12) a = la[la.length - 1];
    types[a.t] = (types[a.t] || 0) + 1;
    applyAction(s, a);
    if (!consistent(s) && !(s.pending && s.pending.type === 'super')) throw new Error('inconsistent after ' + JSON.stringify(a));
    acts++;
    if (guard % 50 === 0) JSON.parse(JSON.stringify(viewFor(s, 0)));
  }
  if (!s.over) throw new Error('did not finish');
  done++; turns += s.turnNo;
  if (s.winner < 0) draws++; else wins[s.winner]++;
}
console.log({ done, draws, wins, avgTurns: turns / done, acts, types });
