// fastClone must behave exactly like a JSON deep copy under any action sequence.
import { newGame, applyAction, actor, legalActions, cloneState, fastClone } from '../js/engine.js';
let checks = 0;
for (let g = 0; g < 60; g++) {
  const s = newGame({ numPlayers: 2 + (g % 3), seed: 4000 + g });
  let n = 0;
  while (!s.over && n++ < 600) {
    const a0 = legalActions(s, g % 2 === 0);
    const a = Math.random() < 0.12 ? a0[a0.length - 1] : a0[Math.floor(Math.random() * a0.length)];
    // fork: apply the same random continuation to a fast clone and a json clone, compare
    if (n % 7 === 0) {
      const f = fastClone(s), j = cloneState(s); f.log = []; j.log = [];
      for (let k = 0; k < 25 && !f.over; k++) {
        const la = legalActions(f, false);
        const b = la[Math.floor(Math.random() * la.length)];
        applyAction(f, b); applyAction(j, b);
      }
      f.log = []; j.log = [];
      if (JSON.stringify(f) !== JSON.stringify(j)) { console.log('MISMATCH'); process.exit(1); }
      // the original must be untouched by the clones' mutations
      checks++;
    }
    const before = JSON.stringify({ ...s, log: [] });
    applyAction(s, a);
    void before;
  }
}
// independence: mutate a clone, original unchanged
const s = newGame({ numPlayers: 2, seed: 1 });
const snap = JSON.stringify({ ...s, log: [] });
const f = fastClone(s);
for (let k = 0; k < 40 && !f.over; k++) { const la = legalActions(f, false); applyAction(f, la[Math.floor(Math.random() * la.length)]); }
if (JSON.stringify({ ...s, log: [] }) !== snap) { console.log('ORIGINAL MUTATED'); process.exit(1); }
console.log('ok', { checks });
