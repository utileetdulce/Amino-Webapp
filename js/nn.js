// Tiny actor-critic network with hand-written backprop + Adam, no dependencies.
// Policy scores each candidate action:  logit = MLP([state, action])   (variable-size action sets)
// Value head:                            v     = MLP(state)
import { SF, AF } from './features.js';

export const DEFAULT_ARCH = { h1: 64, h2: 32, hv: 48 };

function layout(arch) {
  const { h1, h2, hv } = arch;
  const spec = [['Ws', SF * h1], ['Wa', AF * h1], ['b1', h1], ['W2', h1 * h2], ['b2', h2], ['W3', h2], ['b3', 1],
    ['V1', SF * hv], ['bv1', hv], ['V2', hv], ['bv2', 1]];
  let off = 0;
  const idx = {};
  for (const [n, len] of spec) { idx[n] = [off, len]; off += len; }
  return { idx, size: off };
}

export class Net {
  constructor(arch = DEFAULT_ARCH, theta = null, seedRng = Math.random) {
    this.arch = { ...arch };
    const L = layout(this.arch);
    this.idx = L.idx;
    this.size = L.size;
    this.theta = theta ? Float32Array.from(theta) : new Float32Array(L.size);
    this.grad = new Float32Array(L.size);
    this.m = new Float32Array(L.size);
    this.v = new Float32Array(L.size);
    this.t = 0;
    if (!theta) this.init(seedRng);
    this.bind();
  }
  bind() {
    for (const n in this.idx) {
      const [o, l] = this.idx[n];
      this[n] = this.theta.subarray(o, o + l);
      this['g' + n] = this.grad.subarray(o, o + l);
    }
  }
  init(rng) {
    const gauss = () => { let u = 0, v = 0; while (!u) u = rng(); while (!v) v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const { h1, h2, hv } = this.arch;
    const fill = (name, fanIn, gain = 1) => {
      const [o, l] = this.idx[name];
      const sd = gain * Math.sqrt(2 / fanIn);
      for (let i = 0; i < l; i++) this.theta[o + i] = gauss() * sd;
    };
    fill('Ws', SF + AF); fill('Wa', SF + AF); fill('W2', h1); fill('W3', h2, 0.1); fill('V1', SF); fill('V2', hv, 0.5);
  }
  clone() {
    const n = new Net(this.arch, this.theta);
    return n;
  }
  copyFrom(o) { this.theta.set(o.theta); }
  toJSON() { return { arch: this.arch, theta: Array.from(this.theta, (x) => +x.toFixed(6)) }; }
  static fromJSON(j) { return new Net(j.arch, j.theta); }

  // Forward over candidates. A: Float32Array n*AF. Returns {logits, cache}
  forwardPolicy(s, A, n, keep = false) {
    const { h1, h2 } = this.arch;
    const hs = new Float32Array(h1);
    for (let j = 0; j < h1; j++) hs[j] = this.b1[j];
    for (let i = 0; i < SF; i++) {
      const x = s[i];
      if (x === 0) continue;
      const row = i * h1;
      for (let j = 0; j < h1; j++) hs[j] += x * this.Ws[row + j];
    }
    const logits = new Float32Array(n);
    const H1 = keep ? new Float32Array(n * h1) : null;
    const H2 = keep ? new Float32Array(n * h2) : null;
    const a1 = new Float32Array(h1), a2 = new Float32Array(h2);
    for (let c = 0; c < n; c++) {
      a1.set(hs);
      const base = c * AF;
      for (let i = 0; i < AF; i++) {
        const x = A[base + i];
        if (x === 0) continue;
        const row = i * h1;
        for (let j = 0; j < h1; j++) a1[j] += x * this.Wa[row + j];
      }
      for (let j = 0; j < h1; j++) if (a1[j] < 0) a1[j] = 0;
      for (let k = 0; k < h2; k++) a2[k] = this.b2[k];
      for (let j = 0; j < h1; j++) {
        const x = a1[j];
        if (x === 0) continue;
        const row = j * h2;
        for (let k = 0; k < h2; k++) a2[k] += x * this.W2[row + k];
      }
      let z = this.b3[0];
      for (let k = 0; k < h2; k++) { if (a2[k] < 0) a2[k] = 0; z += a2[k] * this.W3[k]; }
      logits[c] = z;
      if (keep) { H1.set(a1, c * h1); H2.set(a2, c * h2); }
    }
    return { logits, H1, H2 };
  }

  forwardValue(s, keep = false) {
    const { hv } = this.arch;
    const a = new Float32Array(hv);
    for (let j = 0; j < hv; j++) a[j] = this.bv1[j];
    for (let i = 0; i < SF; i++) {
      const x = s[i];
      if (x === 0) continue;
      const row = i * hv;
      for (let j = 0; j < hv; j++) a[j] += x * this.V1[row + j];
    }
    let v = this.bv2[0];
    for (let j = 0; j < hv; j++) { if (a[j] < 0) a[j] = 0; v += a[j] * this.V2[j]; }
    return keep ? { v, a } : v;
  }

  // Accumulate gradients. dlog: dLoss/dlogit per candidate; dv: dLoss/dvalue
  backward(s, A, n, fp, dlog, vcache, dv) {
    const { h1, h2, hv } = this.arch;
    const dhs = new Float32Array(h1);
    const dh1 = new Float32Array(h1), dh2 = new Float32Array(h2);
    for (let c = 0; c < n; c++) {
      const dl = dlog[c];
      if (dl === 0) continue;
      const a1 = fp.H1.subarray(c * h1, (c + 1) * h1), a2 = fp.H2.subarray(c * h2, (c + 1) * h2);
      this.gb3[0] += dl;
      for (let k = 0; k < h2; k++) {
        this.gW3[k] += dl * a2[k];
        dh2[k] = a2[k] > 0 ? dl * this.W3[k] : 0;
        this.gb2[k] += dh2[k];
      }
      for (let j = 0; j < h1; j++) {
        const row = j * h2;
        const x = a1[j];
        let acc = 0;
        for (let k = 0; k < h2; k++) {
          const d = dh2[k];
          if (x !== 0) this.gW2[row + k] += x * d;
          acc += this.W2[row + k] * d;
        }
        dh1[j] = x > 0 ? acc : 0;
        dhs[j] += dh1[j];
        this.gb1[j] += dh1[j];
      }
      const base = c * AF;
      for (let i = 0; i < AF; i++) {
        const x = A[base + i];
        if (x === 0) continue;
        const row = i * h1;
        for (let j = 0; j < h1; j++) this.gWa[row + j] += x * dh1[j];
      }
    }
    for (let i = 0; i < SF; i++) {
      const x = s[i];
      if (x === 0) continue;
      const row = i * h1;
      for (let j = 0; j < h1; j++) this.gWs[row + j] += x * dhs[j];
    }
    if (vcache) {
      this.gbv2[0] += dv;
      const dA = new Float32Array(hv);
      for (let j = 0; j < hv; j++) {
        this.gV2[j] += dv * vcache.a[j];
        dA[j] = vcache.a[j] > 0 ? dv * this.V2[j] : 0;
        this.gbv1[j] += dA[j];
      }
      for (let i = 0; i < SF; i++) {
        const x = s[i];
        if (x === 0) continue;
        const row = i * hv;
        for (let j = 0; j < hv; j++) this.gV1[row + j] += x * dA[j];
      }
    }
  }

  zeroGrad() { this.grad.fill(0); }

  adamStep(lr, clip = 1.0, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
    let norm = 0;
    for (let i = 0; i < this.size; i++) norm += this.grad[i] * this.grad[i];
    norm = Math.sqrt(norm);
    const sc = norm > clip ? clip / norm : 1;
    this.t++;
    const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
    for (let i = 0; i < this.size; i++) {
      const g = this.grad[i] * sc;
      this.m[i] = b1 * this.m[i] + (1 - b1) * g;
      this.v[i] = b2 * this.v[i] + (1 - b2) * g * g;
      this.theta[i] -= lr * (this.m[i] / c1) / (Math.sqrt(this.v[i] / c2) + eps);
    }
    return norm;
  }
}

export function softmax(logits, temp = 1) {
  const n = logits.length;
  let mx = -Infinity;
  for (let i = 0; i < n; i++) if (logits[i] > mx) mx = logits[i];
  const p = new Float32Array(n);
  let z = 0;
  for (let i = 0; i < n; i++) { p[i] = Math.exp((logits[i] - mx) / temp); z += p[i]; }
  for (let i = 0; i < n; i++) p[i] /= z;
  return p;
}

export function sampleIdx(p, rng = Math.random) {
  let r = rng(), acc = 0;
  for (let i = 0; i < p.length; i++) { acc += p[i]; if (r <= acc) return i; }
  return p.length - 1;
}
