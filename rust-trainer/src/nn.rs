// Port of js/nn.js: actor-critic MLP with hand-written backprop + Adam. Weight layout identical to JS.
use crate::features::{AF, SF};
use crate::rng::Rng;

#[derive(Clone, Copy)]
pub struct Arch { pub h1: usize, pub h2: usize, pub hv: usize }
pub const DEFAULT_ARCH: Arch = Arch { h1: 64, h2: 32, hv: 48 };

#[derive(Clone, Copy)]
struct Lay { ws: usize, wa: usize, b1: usize, w2: usize, b2: usize, w3: usize, b3: usize, v1: usize, bv1: usize, v2: usize, bv2: usize, size: usize }

fn layout(a: Arch) -> Lay {
    let (h1, h2, hv) = (a.h1, a.h2, a.hv);
    let mut o = 0;
    let mut take = |n: usize| { let r = o; o += n; r };
    let ws = take(SF * h1); let wa = take(AF * h1); let b1 = take(h1); let w2 = take(h1 * h2); let b2 = take(h2);
    let w3 = take(h2); let b3 = take(1); let v1 = take(SF * hv); let bv1 = take(hv); let v2 = take(hv); let bv2 = take(1);
    Lay { ws, wa, b1, w2, b2, w3, b3, v1, bv1, v2, bv2, size: o }
}

#[derive(Clone)]
pub struct Net { pub arch: Arch, l: Lay, pub theta: Vec<f32>, m: Vec<f32>, v: Vec<f32>, t: i32 }

pub struct Fp { pub logits: Vec<f32>, pub h1: Vec<f32>, pub h2: Vec<f32> }

impl Net {
    pub fn size(&self) -> usize { self.l.size }
    pub fn from_theta(arch: Arch, theta: Vec<f32>) -> Net {
        let l = layout(arch);
        assert_eq!(theta.len(), l.size, "theta size mismatch with arch");
        Net { arch, l, m: vec![0.0; l.size], v: vec![0.0; l.size], t: 0, theta }
    }
    pub fn random(arch: Arch, rng: &mut Rng) -> Net {
        let l = layout(arch);
        let mut n = Net { arch, l, theta: vec![0.0; l.size], m: vec![0.0; l.size], v: vec![0.0; l.size], t: 0 };
        let mut fill = |o: usize, len: usize, fan_in: usize, gain: f64| {
            let sd = gain * (2.0 / fan_in as f64).sqrt();
            for i in 0..len { n.theta[o + i] = (rng.gauss() * sd) as f32; }
        };
        fill(l.ws, SF * arch.h1, SF + AF, 1.0); fill(l.wa, AF * arch.h1, SF + AF, 1.0); fill(l.w2, arch.h1 * arch.h2, arch.h1, 1.0);
        fill(l.w3, arch.h2, arch.h2, 0.1); fill(l.v1, SF * arch.hv, SF, 1.0); fill(l.v2, arch.hv, arch.hv, 0.5);
        n
    }

    pub fn forward_policy(&self, s: &[f32], a: &[f32], n: usize, keep: bool) -> Fp {
        if self.arch.h1 == 64 && self.arch.h2 == 32 { return self.forward_policy_c::<64, 32>(s, a, n, keep); }
        self.forward_policy_dyn(s, a, n, keep)
    }

    /// fixed-size fast path (const generics let the compiler fully unroll/vectorise the matvecs)
    fn forward_policy_c<const H1: usize, const H2: usize>(&self, s: &[f32], a: &[f32], n: usize, keep: bool) -> Fp {
        let l = &self.l; let th = &self.theta;
        let mut hs = [0f32; H1];
        hs.copy_from_slice(&th[l.b1..l.b1 + H1]);
        for i in 0..SF {
            let x = s[i]; if x == 0.0 { continue; }
            let row: &[f32; H1] = th[l.ws + i * H1..l.ws + (i + 1) * H1].try_into().unwrap();
            for j in 0..H1 { hs[j] += x * row[j]; }
        }
        let mut logits = vec![0f32; n];
        let mut o1 = if keep { vec![0f32; n * H1] } else { Vec::new() };
        let mut o2 = if keep { vec![0f32; n * H2] } else { Vec::new() };
        let b2: &[f32; H2] = th[l.b2..l.b2 + H2].try_into().unwrap();
        let w3: &[f32; H2] = th[l.w3..l.w3 + H2].try_into().unwrap();
        let w2 = &th[l.w2..l.w2 + H1 * H2];
        for c in 0..n {
            let mut a1 = hs;
            for i in 0..AF {
                let x = a[c * AF + i]; if x == 0.0 { continue; }
                let row: &[f32; H1] = th[l.wa + i * H1..l.wa + (i + 1) * H1].try_into().unwrap();
                for j in 0..H1 { a1[j] += x * row[j]; }
            }
            for j in 0..H1 { a1[j] = a1[j].max(0.0); }
            let mut a2 = *b2;
            for j in 0..H1 {
                let x = a1[j]; if x == 0.0 { continue; }
                let row: &[f32; H2] = w2[j * H2..(j + 1) * H2].try_into().unwrap();
                for k in 0..H2 { a2[k] += x * row[k]; }
            }
            let mut z = th[l.b3];
            for k in 0..H2 { a2[k] = a2[k].max(0.0); z += a2[k] * w3[k]; }
            logits[c] = z;
            if keep { o1[c * H1..(c + 1) * H1].copy_from_slice(&a1); o2[c * H2..(c + 1) * H2].copy_from_slice(&a2); }
        }
        Fp { logits, h1: o1, h2: o2 }
    }

    fn forward_policy_dyn(&self, s: &[f32], a: &[f32], n: usize, keep: bool) -> Fp {
        let (h1, h2) = (self.arch.h1, self.arch.h2);
        let l = &self.l; let th = &self.theta;
        let mut hs: Vec<f32> = th[l.b1..l.b1 + h1].to_vec();
        for i in 0..SF {
            let x = s[i]; if x == 0.0 { continue; }
            let row = &th[l.ws + i * h1..l.ws + (i + 1) * h1];
            for j in 0..h1 { hs[j] += x * row[j]; }
        }
        let mut logits = vec![0f32; n];
        let mut o1 = if keep { vec![0f32; n * h1] } else { Vec::new() };
        let mut o2 = if keep { vec![0f32; n * h2] } else { Vec::new() };
        let mut a1 = vec![0f32; h1]; let mut a2 = vec![0f32; h2];
        let b2 = &th[l.b2..l.b2 + h2]; let w3 = &th[l.w3..l.w3 + h2];
        for c in 0..n {
            a1.copy_from_slice(&hs);
            for i in 0..AF {
                let x = a[c * AF + i]; if x == 0.0 { continue; }
                let row = &th[l.wa + i * h1..l.wa + (i + 1) * h1];
                for j in 0..h1 { a1[j] += x * row[j]; }
            }
            for j in 0..h1 { if a1[j] < 0.0 { a1[j] = 0.0; } }
            a2.copy_from_slice(b2);
            for j in 0..h1 {
                let x = a1[j]; if x == 0.0 { continue; }
                let row = &th[l.w2 + j * h2..l.w2 + (j + 1) * h2];
                for k in 0..h2 { a2[k] += x * row[k]; }
            }
            let mut z = th[l.b3];
            for k in 0..h2 { if a2[k] < 0.0 { a2[k] = 0.0; } z += a2[k] * w3[k]; }
            logits[c] = z;
            if keep { o1[c * h1..(c + 1) * h1].copy_from_slice(&a1); o2[c * h2..(c + 1) * h2].copy_from_slice(&a2); }
        }
        Fp { logits, h1: o1, h2: o2 }
    }

    /// returns (v, hidden activations)
    pub fn forward_value(&self, s: &[f32], keep: bool) -> (f32, Vec<f32>) {
        let hv = self.arch.hv; let l = &self.l; let th = &self.theta;
        let mut a: Vec<f32> = th[l.bv1..l.bv1 + hv].to_vec();
        for i in 0..SF {
            let x = s[i]; if x == 0.0 { continue; }
            let row = &th[l.v1 + i * hv..l.v1 + (i + 1) * hv];
            for j in 0..hv { a[j] += x * row[j]; }
        }
        let mut v = th[l.bv2];
        let v2 = &th[l.v2..l.v2 + hv];
        for j in 0..hv { if a[j] < 0.0 { a[j] = 0.0; } v += a[j] * v2[j]; }
        (v, if keep { a } else { Vec::new() })
    }

    /// Accumulate gradients into `g` (same layout as theta).
    pub fn backward(&self, g: &mut [f32], s: &[f32], a: &[f32], n: usize, fp: &Fp, dlog: &[f32], vact: &[f32], dv: f32) {
        let (h1, h2, hv) = (self.arch.h1, self.arch.h2, self.arch.hv);
        let l = &self.l; let th = &self.theta;
        let mut dhs = vec![0f32; h1]; let mut dh1 = vec![0f32; h1]; let mut dh2 = vec![0f32; h2];
        let w3 = &th[l.w3..l.w3 + h2];
        for c in 0..n {
            let dl = dlog[c]; if dl == 0.0 { continue; }
            let a1 = &fp.h1[c * h1..(c + 1) * h1]; let a2 = &fp.h2[c * h2..(c + 1) * h2];
            g[l.b3] += dl;
            for k in 0..h2 {
                g[l.w3 + k] += dl * a2[k];
                dh2[k] = if a2[k] > 0.0 { dl * w3[k] } else { 0.0 };
                g[l.b2 + k] += dh2[k];
            }
            for j in 0..h1 {
                let x = a1[j];
                let wrow = &th[l.w2 + j * h2..l.w2 + (j + 1) * h2];
                let mut acc = 0f32;
                for k in 0..h2 { acc += wrow[k] * dh2[k]; }
                if x != 0.0 {
                    let grow = &mut g[l.w2 + j * h2..l.w2 + (j + 1) * h2];
                    for k in 0..h2 { grow[k] += x * dh2[k]; }
                }
                dh1[j] = if x > 0.0 { acc } else { 0.0 };
                dhs[j] += dh1[j];
                g[l.b1 + j] += dh1[j];
            }
            for i in 0..AF {
                let x = a[c * AF + i]; if x == 0.0 { continue; }
                let grow = &mut g[l.wa + i * h1..l.wa + (i + 1) * h1];
                for j in 0..h1 { grow[j] += x * dh1[j]; }
            }
        }
        for i in 0..SF {
            let x = s[i]; if x == 0.0 { continue; }
            let grow = &mut g[l.ws + i * h1..l.ws + (i + 1) * h1];
            for j in 0..h1 { grow[j] += x * dhs[j]; }
        }
        if !vact.is_empty() {
            g[l.bv2] += dv;
            let mut da = vec![0f32; hv];
            for j in 0..hv {
                g[l.v2 + j] += dv * vact[j];
                da[j] = if vact[j] > 0.0 { dv * th[l.v2 + j] } else { 0.0 };
                g[l.bv1 + j] += da[j];
            }
            for i in 0..SF {
                let x = s[i]; if x == 0.0 { continue; }
                let grow = &mut g[l.v1 + i * hv..l.v1 + (i + 1) * hv];
                for j in 0..hv { grow[j] += x * da[j]; }
            }
        }
    }

    pub fn adam_step(&mut self, grad: &[f32], lr: f64, clip: f64) -> f64 {
        let (b1, b2, eps) = (0.9f64, 0.999f64, 1e-8f64);
        let mut norm = 0f64;
        for &g in grad { norm += (g as f64) * (g as f64); }
        norm = norm.sqrt();
        let sc = if norm > clip { clip / norm } else { 1.0 };
        self.t += 1;
        let c1 = 1.0 - b1.powi(self.t); let c2 = 1.0 - b2.powi(self.t);
        for i in 0..self.theta.len() {
            let g = grad[i] as f64 * sc;
            let m = b1 * self.m[i] as f64 + (1.0 - b1) * g;
            let v = b2 * self.v[i] as f64 + (1.0 - b2) * g * g;
            self.m[i] = m as f32; self.v[i] = v as f32;
            self.theta[i] = (self.theta[i] as f64 - lr * (m / c1) / ((v / c2).sqrt() + eps)) as f32;
        }
        norm
    }
}

pub fn softmax(logits: &[f32], temp: f32) -> Vec<f32> {
    let mut mx = f32::NEG_INFINITY;
    for &l in logits { if l > mx { mx = l; } }
    let mut p: Vec<f32> = logits.iter().map(|&l| ((l - mx) / temp).exp()).collect();
    let z: f32 = p.iter().sum();
    for x in p.iter_mut() { *x /= z; }
    p
}

pub fn sample_idx(p: &[f32], rng: &mut Rng) -> usize {
    let r = rng.f64() as f32; let mut acc = 0f32;
    for (i, &x) in p.iter().enumerate() { acc += x; if r <= acc { return i; } }
    p.len() - 1
}
