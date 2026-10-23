// AlphaZero-style training: self-play with MCTS (parallel worker threads) + replay-buffer updates.
//
//   node tools/train-az.mjs --init models/pretrained.json --out models/az.json --sims 64 --games 24 --minutes 240
//
// Options: --workers N (default cores-1)  --sims  --games (per iteration)  --steps (SGD minibatches / iter)  --batch
//          --lr  --buffer (replay size)  --eval-every N iters  --eval-games N  --opp-mix 0.2  --depth 18  --minutes / --iters
import fs from 'node:fs';
import os from 'node:os';
import { Worker } from 'node:worker_threads';
import { Net, DEFAULT_ARCH } from '../js/nn.js';
import { azUpdate } from '../js/az.js';
import { childMeta, touchMeta, RULES_VERSION } from '../js/modelmeta.js';
import { hostInfo, gitCommit } from './meta-node.mjs';

const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i < 0 ? d : (process.argv[i + 1] && !process.argv[i + 1].startsWith('--') ? process.argv[i + 1] : true); };
const cfg = {
  workers: +arg('workers', Math.max(1, os.cpus().length - 1)), sims: +arg('sims', 64), games: +arg('games', 24), steps: +arg('steps', 150), batch: +arg('batch', 128),
  lr: +arg('lr', 3e-4), buffer: +arg('buffer', 40000), evalEvery: +arg('eval-every', 5), evalGames: +arg('eval-games', 16), oppMix: +arg('opp-mix', 0.2),
  depth: +arg('depth', 18), topK: +arg('topk', 12), cpuct: +arg('cpuct', 1.25), qbeta: +arg('qbeta', 3), ratio: +arg('ratio', 4), minBuffer: +arg('min-buffer', 5000), minutes: +arg('minutes', 0), iters: +arg('iters', 100000), tauMoves: +arg('tau-moves', 30),
};
const out = arg('out', 'models/az.json');
const init = arg('init', fs.existsSync('models/pretrained.json') ? 'models/pretrained.json' : null);

let net, meta = { games: 0, iter: 0, history: [] }, parentModel = null;
if (init && fs.existsSync(init)) {
  const j = JSON.parse(fs.readFileSync(init, 'utf8'));
  net = Net.fromJSON(j.net); parentModel = j;
  if (arg('resume', false)) meta = { games: j.games || 0, iter: j.iter || 0, history: j.history || [] };
  console.log('init from', init);
} else { net = new Net(DEFAULT_ARCH); console.log('init: random network (consider warm-starting with tools/train-cli.mjs first)'); }

const t0run = Date.now();
let mm = childMeta(parentModel, { name: arg('name', out.replace(/^.*\//, '').replace(/\.json$/, '')), kind: 'az', trainer: 'js-cli', hyper: { sims: cfg.sims, topk: cfg.topK, cpuct: cfg.cpuct, qbeta: cfg.qbeta, games: cfg.games, lr: cfg.lr, ratio: cfg.ratio, minBuffer: cfg.minBuffer, batch: cfg.batch, oppMix: cfg.oppMix, depth: cfg.depth }, host: hostInfo(), git: gitCommit(), trainMinutes: arg('resume', false) ? (parentModel?.meta?.trainMinutes || 0) : 0 });
const minutes0 = mm.trainMinutes;
if (parentModel && (parentModel.meta?.rulesVersion ?? 0) < RULES_VERSION) console.log(`WARNING: parent model was trained under rules v${parentModel.meta?.rulesVersion ?? '?'} (current v${RULES_VERSION}). Warm-starting is usually fine; start fresh only if features/architecture changed.`);
const pool = Array.from({ length: cfg.workers }, () => new Worker(new URL('./az-worker.mjs', import.meta.url)));
const call = (w, msg) => new Promise((res, rej) => { w.once('message', res); w.once('error', rej); w.postMessage(msg); });
const split = (n, k) => Array.from({ length: k }, (_, i) => Math.floor((n * (i + 1)) / k) - Math.floor((n * i) / k));
const save = (file) => { mm = touchMeta(mm, { trainMinutes: minutes0 + (Date.now() - t0run) / 60000 }); fs.writeFileSync(file, JSON.stringify({ version: 3, name: mm.name, kind: 'mcts', sims: cfg.sims, games: meta.games, iter: meta.iter, history: meta.history, meta: mm, net: net.toJSON() })); };

async function evaluate() {
  const g = split(cfg.evalGames, pool.length);
  const run = async (search) => {
    const rs = await Promise.all(pool.map((w, i) => (g[i] ? call(w, { type: 'eval', theta: net.theta, arch: net.arch, games: g[i], offset: g.slice(0, i).reduce((a, b) => a + b, 0), search, sims: cfg.sims, cfg }) : { w: 0, d: 0, l: 0 })));
    const w = rs.reduce((a, r) => a + r.w, 0), d = rs.reduce((a, r) => a + r.d, 0), l = rs.reduce((a, r) => a + r.l, 0);
    return { w, d, l, rate: (w + 0.5 * d) / Math.max(1, w + d + l) };
  };
  return { raw: await run(false), search: await run(true) };
}

const t0 = Date.now();
let buffer = [], best = -1;
console.log('config', JSON.stringify(cfg));
for (let it = 0; it < cfg.iters; it++) {
  if (cfg.minutes && (Date.now() - t0) / 60000 > cfg.minutes) break;
  const t1 = Date.now();
  const gs = split(cfg.games, pool.length);
  const rs = await Promise.all(pool.map((w, i) => (gs[i] ? call(w, { type: 'play', theta: net.theta, arch: net.arch, games: gs[i], oppMix: cfg.oppMix, cfg: { sims: cfg.sims, depth: cfg.depth, tauMoves: cfg.tauMoves, qbeta: cfg.qbeta, topK: cfg.topK, cpuct: cfg.cpuct } }) : { samples: [], games: 0, turns: 0, vsHeur: { w: 0, d: 0, l: 0 } })));
  let nGames = 0, turns = 0; const vh = { w: 0, d: 0, l: 0 };
  for (const r of rs) { buffer.push(...r.samples); nGames += r.games; turns += r.turns; vh.w += r.vsHeur.w; vh.d += r.vsHeur.d; vh.l += r.vsHeur.l; }
  if (buffer.length > cfg.buffer) buffer = buffer.slice(buffer.length - cfg.buffer);
  const tPlay = (Date.now() - t1) / 1000;
  const newSamples = rs.reduce((a, r) => a + r.samples.length, 0);
  const stepsEff = buffer.length < cfg.minBuffer ? 0 : Math.min(cfg.steps, Math.max(1, Math.ceil((newSamples * cfg.ratio) / cfg.batch)));
  const up = stepsEff ? azUpdate(net, buffer, { steps: stepsEff, batch: cfg.batch, lr: cfg.lr }) : { policyLoss: 0, valueLoss: 0 };
  meta.games += nGames; meta.iter++;
  console.log(`iter ${meta.iter} games ${meta.games} buf ${buffer.length} turns/game ${(turns / nGames).toFixed(0)} play ${tPlay.toFixed(0)}s  ploss ${up.policyLoss.toFixed(3)} vloss ${up.valueLoss.toFixed(3)}  vsHeur(self-play seats) ${vh.w}-${vh.d}-${vh.l}`);
  save(out);
  if (cfg.evalEvery && meta.iter % cfg.evalEvery === 0) {
    const ev = await evaluate();
    meta.history.push({ iter: meta.iter, games: meta.games, random: ev.raw.rate, heuristic: ev.search.rate, rawHeuristic: ev.raw.rate });
    console.log(`EVAL iter ${meta.iter}: raw net vs heuristic ${(ev.raw.rate * 100).toFixed(0)}% | MCTS(${cfg.sims}) vs heuristic ${(ev.search.rate * 100).toFixed(0)}%`);
    if (ev.search.rate >= best) { best = ev.search.rate; save(out.replace(/\.json$/, '') + '.best.json'); }
    save(out);
  }
}
await Promise.all(pool.map((w) => w.terminate()));
console.log('done', ((Date.now() - t0) / 60000).toFixed(1), 'min');
