// Function-preserving widening ("Net2Net"): copies a trained network into a larger one that initially computes exactly the same
// function. New hidden neurons get small random incoming weights and ZERO outgoing weights, so outputs are unchanged but the new
// neurons can learn (the gradient w.r.t. their outgoing weights is non-zero).
//   node tools/widen-model.mjs --in models/az-kurz.json --arch 128,64,96 --out models/az-gross-start.json [--name "..."]
import fs from 'node:fs';
import { Net } from '../js/nn.js';
import { SF, AF } from '../js/features.js';
import { childMeta, touchMeta } from '../js/modelmeta.js';
import { hostInfo, gitCommit } from './meta-node.mjs';

const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i < 0 ? d : process.argv[i + 1]; };
const src = JSON.parse(fs.readFileSync(arg('in'), 'utf8'));
const [h1, h2, hv] = arg('arch', '128,64,96').split(',').map(Number);
const old = Net.fromJSON(src.net);
const o = old.arch;
if (h1 < o.h1 || h2 < o.h2 || hv < o.hv) throw new Error('target architecture must be at least as large as the source');
const big = new Net({ h1, h2, hv });           // random init for everything
const sd = (fan) => Math.sqrt(2 / fan) * 0.5;
const gauss = () => { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
big.theta.fill(0);
const put = (name, fn) => { const [off, len] = big.idx[name]; for (let i = 0; i < len; i++) big.theta[off + i] = fn(i); };
// incoming weights of new neurons: small random; outgoing weights of new neurons: zero
put('Ws', (i) => { const r = Math.floor(i / h1), c = i % h1; return c < o.h1 ? old.Ws[r * o.h1 + c] : gauss() * sd(SF + AF); });
put('Wa', (i) => { const r = Math.floor(i / h1), c = i % h1; return c < o.h1 ? old.Wa[r * o.h1 + c] : gauss() * sd(SF + AF); });
put('b1', (i) => (i < o.h1 ? old.b1[i] : 0));
put('W2', (i) => { const r = Math.floor(i / h2), c = i % h2; return r < o.h1 && c < o.h2 ? old.W2[r * o.h2 + c] : (r >= o.h1 && c < o.h2 ? 0 : (c >= o.h2 ? gauss() * sd(h1) : 0)); });
put('b2', (i) => (i < o.h2 ? old.b2[i] : 0));
put('W3', (i) => (i < o.h2 ? old.W3[i] : 0));
put('b3', (i) => old.b3[i]);
put('V1', (i) => { const r = Math.floor(i / hv), c = i % hv; return c < o.hv ? old.V1[r * o.hv + c] : gauss() * sd(SF); });
put('bv1', (i) => (i < o.hv ? old.bv1[i] : 0));
put('V2', (i) => (i < o.hv ? old.V2[i] : 0));
put('bv2', (i) => old.bv2[i]);

// verify: identical outputs on random inputs
let maxDiff = 0;
for (let t = 0; t < 50; t++) {
  const s = Float32Array.from({ length: SF }, () => Math.random()), n = 7;
  const A = Float32Array.from({ length: n * AF }, () => (Math.random() < 0.3 ? Math.random() : 0));
  const a = old.forwardPolicy(s, A, n).logits, b = big.forwardPolicy(s, A, n).logits;
  for (let i = 0; i < n; i++) maxDiff = Math.max(maxDiff, Math.abs(a[i] - b[i]));
  maxDiff = Math.max(maxDiff, Math.abs(old.forwardValue(s) - big.forwardValue(s)));
}
console.log(`widened ${o.h1}/${o.h2}/${o.hv} -> ${h1}/${h2}/${hv}: ${old.size} -> ${big.size} parameters, max output difference ${maxDiff.toExponential(2)}`);
if (maxDiff > 1e-4) throw new Error('widening did not preserve the function');

const name = arg('name', `${(src.meta?.name || 'Modell')} · ${h1}/${h2}/${hv}`);
let meta = childMeta(src, { name, kind: src.meta?.kind || 'az', trainer: 'widen-model', hyper: { arch: `${h1}/${h2}/${hv}`, from: `${o.h1}/${o.h2}/${o.hv}` }, host: hostInfo(), git: gitCommit(), trainMinutes: src.meta?.trainMinutes || 0,
  note: `Vergrößertes Netz (${old.size} → ${big.size} Parameter), funktionsgleich zum Ausgangsmodell – noch nicht weiter trainiert.` });
meta = touchMeta(meta, {});
fs.writeFileSync(arg('out'), JSON.stringify({ version: 3, name, kind: src.kind || 'mcts', sims: src.sims, games: src.games || 0, iter: src.iter || 0, history: src.history || [], meta, net: big.toJSON() }));
console.log('written', arg('out'));
