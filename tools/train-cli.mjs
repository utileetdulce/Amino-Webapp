// node tools/train-cli.mjs --minutes 10 --out models/pretrained.json [--resume file] [--no-clone]
import fs from 'node:fs';
import { Trainer } from '../js/trainer.js';
import { Net } from '../js/nn.js';
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i < 0 ? d : (process.argv[i + 1] && !process.argv[i + 1].startsWith('--') ? process.argv[i + 1] : true); };
const minutes = +arg('minutes', 5), out = arg('out', 'models/pretrained.json');
const resume = arg('resume', null);
const net = resume && fs.existsSync(resume) ? Net.fromJSON(JSON.parse(fs.readFileSync(resume, 'utf8')).net) : null;
const tr = new Trainer({ net, episodesPerIter: +arg('eps', 12), opponent: arg('opp', 'mixed') });
const t0 = Date.now();
const history = [];
if (!arg('no-clone', false) && !net) {
  for (let i = 0; i < +arg('clone-rounds', 6); i++) { const r = tr.clone(10, 3); console.log('BC', i, r.loss.toFixed(3), r.demos); }
  console.log('after BC', JSON.stringify(tr.evalAll(20)));
}
let nextEval = 0;
while ((Date.now() - t0) / 60000 < minutes) {
  const st = tr.iterate();
  if (st.iter % 5 === 0) console.log(`iter ${st.iter} games ${st.games} ret ${st.meanReturn.toFixed(3)} ent ${st.entropy?.toFixed(2)} kl ${st.kl?.toFixed(4)} turns ${st.avgTurns.toFixed(0)} byOpp ${JSON.stringify(st.byOpp)}`);
  if (st.iter >= nextEval) {
    const ev = tr.evalAll(30);
    history.push({ iter: st.iter, games: st.games, random: ev.random.rate, heuristic: ev.heuristic.rate });
    console.log('EVAL', JSON.stringify(history[history.length - 1]));
    nextEval = st.iter + 25;
    fs.writeFileSync(out, JSON.stringify({ version: 1, name: 'pretrained', games: st.games, iter: st.iter, history, net: tr.net.toJSON() }));
  }
}
const ev = tr.evalAll(60);
history.push({ iter: tr.iter, games: tr.games, random: ev.random.rate, heuristic: ev.heuristic.rate });
console.log('FINAL', JSON.stringify(history[history.length - 1]));
fs.writeFileSync(out, JSON.stringify({ version: 1, name: 'pretrained', games: tr.games, iter: tr.iter, history, net: tr.net.toJSON() }));
