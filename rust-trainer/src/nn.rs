// Port of js/nn.js: actor-critic MLP with hand-written backprop + Adam. Weight layout identical to JS for hg = 0.
// With hg > 0 the MLP additionally receives embeddings from a small graph network over the board (see features::Graph):
// pooled / own-ant / enemy-ant embeddings join the state input, and the embedding of each candidate's target cells joins
// the candidate input. Extra weights start at zero when grown from a plain MLP (`grow_from`), so the function is unchanged.
use crate::features::{Graph, Obs, AF, NF, SF};
use crate::rng::Rng;

pub const EXTK: usize = 3;

#[derive(Clone, Copy)]
pub struct Arch { pub h1: usize, pub h2: usize, pub hv: usize, pub hg: usize, pub gl: usize }
pub const DEFAULT_ARCH: Arch = Arch { h1: 64, h2: 32, hv: 48, hg: 0, gl: 0 };

#[derive(Clone, Copy)]
struct Lay { nin: usize, ws: usize, wa: usize, b1: usize, w2: usize, b2: usize, w3: usize, b3: usize, v1: usize, bv1: usize, v2: usize, bv2: usize,
    wc: usize, v3: usize, bv3: usize, win: usize, bin: usize, gl0: usize, lsz: usize, size: usize }

fn layout(a: Arch) -> Lay {
    let (h1, h2, hv, hg) = (a.h1, a.h2, a.hv, a.hg);
    let nin = SF + EXTK * hg;
    let mut o = 0;
    let mut take = |n: usize| { let r = o; o += n; r };
    let ws = take(nin * h1); let wa = take(AF * h1); let b1 = take(h1); let w2 = take(h1 * h2); let b2 = take(h2);
    let w3 = take(h2); let b3 = take(1); let v1 = take(nin * hv); let bv1 = take(hv); let v2 = take(hv); let bv2 = take(1);
    let (mut wc, mut v3, mut bv3, mut win, mut bin, mut gl0, mut lsz) = (0, 0, 0, 0, 0, 0, 0);
    if hg > 0 {
        wc = take(hg * h1); v3 = take(hv); bv3 = take(1); win = take(NF * hg); bin = take(hg);
        lsz = 4 * hg * hg + hg; gl0 = take(lsz * a.gl);
    }
    Lay { nin, ws, wa, b1, w2, b2, w3, b3, v1, bv1, v2, bv2, wc, v3, bv3, win, bin, gl0, lsz, size: o }
}

#[derive(Clone)]
pub struct Net { pub arch: Arch, l: Lay, pub theta: Vec<f32>, m: Vec<f32>, v: Vec<f32>, t: i32 }

pub struct Fp { pub logits: Vec<f32>, pub h1: Vec<f32>, pub h2: Vec<f32> }
pub struct Fv { pub v: f32, pub aux: f32, pub act: Vec<f32> }
pub struct GCache { h: Vec<Vec<f32>>, z: Vec<Vec<f32>> }
pub struct Enc { pub xs: Vec<f32>, pub ce: Vec<f32>, cache: Option<GCache> }

#[inline] fn addmv(out: &mut [f32], x: &[f32], w: &[f32], width: usize) {
    for (i, &xi) in x.iter().enumerate() {
        if xi == 0.0 { continue; }
        let row = &w[i * width..(i + 1) * width];
        for (o, r) in out.iter_mut().zip(row.iter()) { *o += xi * *r; }
    }
}

impl Net {
    pub fn size(&self) -> usize { self.l.size }
    pub fn uses_graph(&self) -> bool { self.arch.hg > 0 }
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
        fill(l.ws, l.nin * arch.h1, l.nin + AF, 1.0); fill(l.wa, AF * arch.h1, l.nin + AF, 1.0); fill(l.w2, arch.h1 * arch.h2, arch.h1, 1.0);
        fill(l.w3, arch.h2, arch.h2, 0.1); fill(l.v1, l.nin * arch.hv, l.nin, 1.0); fill(l.v2, arch.hv, arch.hv, 0.5);
        if arch.hg > 0 {
            for i in 0..arch.hg * arch.h1 { n.theta[l.wc + i] = (rng.gauss() * (2.0 / (arch.hg + AF) as f64).sqrt()) as f32; }
            n.init_gnn(rng);
        }
        n
    }
    fn init_gnn(&mut self, rng: &mut Rng) {
        let (l, hg, gl) = (self.l, self.arch.hg, self.arch.gl);
        let mut fill = |th: &mut Vec<f32>, o: usize, len: usize, fan_in: usize, gain: f64| {
            let sd = gain * (2.0 / fan_in as f64).sqrt();
            for i in 0..len { th[o + i] = (rng.gauss() * sd) as f32; }
        };
        fill(&mut self.theta, l.win, NF * hg, NF, 1.0);
        for k in 0..gl { let o = l.gl0 + k * l.lsz; fill(&mut self.theta, o, 4 * hg * hg, 4 * hg, 0.7); }
    }

    /// Plain MLP -> graph network with identical outputs (all weights that read the graph embeddings start at zero).
    pub fn grow_from(old: &Net, hg: usize, gl: usize, rng: &mut Rng) -> Net {
        assert_eq!(old.arch.hg, 0);
        let arch = Arch { hg, gl, ..old.arch };
        let mut n = Net::random(arch, rng);
        let (lo, ln) = (old.l, n.l);
        let (h1, hv) = (arch.h1, arch.hv);
        let cp = |n: &mut Net, dst: usize, src: usize, len: usize| { n.theta[dst..dst + len].copy_from_slice(&old.theta[src..src + len]); };
        for x in n.theta[ln.ws..ln.ws + ln.nin * h1].iter_mut() { *x = 0.0; }
        for x in n.theta[ln.v1..ln.v1 + ln.nin * hv].iter_mut() { *x = 0.0; }
        for x in n.theta[ln.wc..ln.wc + hg * h1].iter_mut() { *x = 0.0; }
        cp(&mut n, ln.ws, lo.ws, SF * h1); cp(&mut n, ln.wa, lo.wa, AF * h1); cp(&mut n, ln.b1, lo.b1, h1);
        cp(&mut n, ln.w2, lo.w2, h1 * arch.h2); cp(&mut n, ln.b2, lo.b2, arch.h2); cp(&mut n, ln.w3, lo.w3, arch.h2); cp(&mut n, ln.b3, lo.b3, 1);
        cp(&mut n, ln.v1, lo.v1, SF * hv); cp(&mut n, ln.bv1, lo.bv1, hv); cp(&mut n, ln.v2, lo.v2, hv); cp(&mut n, ln.bv2, lo.bv2, 1);
        n
    }

    // ---------- graph network ----------
    fn gnn_forward(&self, g: &Graph) -> GCache {
        let (l, hg, gl) = (self.l, self.arch.hg, self.arch.gl);
        let th = &self.theta;
        let nn = g.n;
        let mut h0 = vec![0f32; nn * hg];
        for i in 0..nn {
            let o = &mut h0[i * hg..(i + 1) * hg];
            o.copy_from_slice(&th[l.bin..l.bin + hg]);
            addmv(o, &g.x[i * NF..(i + 1) * NF], &th[l.win..l.win + NF * hg], hg);
            for v in o.iter_mut() { if *v < 0.0 { *v = 0.0; } }
        }
        let mut hs = vec![h0]; let mut zs: Vec<Vec<f32>> = Vec::with_capacity(gl);
        for k in 0..gl {
            let base = l.gl0 + k * l.lsz;
            let (wself, we, bl) = (&th[base..base + hg * hg], &th[base + hg * hg..base + 4 * hg * hg], &th[base + 4 * hg * hg..base + 4 * hg * hg + hg]);
            let h = &hs[k];
            let mut z = vec![0f32; nn * hg];
            for i in 0..nn {
                let o = &mut z[i * hg..(i + 1) * hg];
                o.copy_from_slice(bl);
                addmv(o, &h[i * hg..(i + 1) * hg], wself, hg);
                for e in 0..3 { let m = g.nbr[i][e]; if m >= 0 { addmv(o, &h[m as usize * hg..(m as usize + 1) * hg], &we[e * hg * hg..(e + 1) * hg * hg], hg); } }
            }
            let mut hn = h.clone();
            for (a, b) in hn.iter_mut().zip(z.iter()) { if *b > 0.0 { *a += *b; } }
            zs.push(z); hs.push(hn);
        }
        GCache { h: hs, z: zs }
    }

    /// Build the policy/value input of a decision (plus, for graph nets, the board embeddings).
    pub fn encode(&self, o: &Obs, keep: bool) -> Enc {
        if self.arch.hg == 0 { return Enc { xs: o.s.to_vec(), ce: Vec::new(), cache: None }; }
        let hg = self.arch.hg;
        let g = o.g.as_ref().expect("graph net needs Obs with graph");
        let cache = self.gnn_forward(g);
        let hl = cache.h.last().unwrap();
        let mut xs = vec![0f32; SF + EXTK * hg];
        xs[..SF].copy_from_slice(&o.s);
        if g.n > 0 {
            for i in 0..g.n { for k in 0..hg { xs[SF + k] += hl[i * hg + k]; } }
            let inv = 1.0 / g.n as f32; for k in 0..hg { xs[SF + k] *= inv; }
        }
        if g.ant >= 0 { xs[SF + hg..SF + 2 * hg].copy_from_slice(&hl[g.ant as usize * hg..(g.ant as usize + 1) * hg]); }
        if g.foe >= 0 { xs[SF + 2 * hg..SF + 3 * hg].copy_from_slice(&hl[g.foe as usize * hg..(g.foe as usize + 1) * hg]); }
        let mut ce = vec![0f32; o.n * hg];
        for (c, nodes) in g.cand.iter().enumerate().take(o.n) {
            if nodes.is_empty() { continue; }
            let inv = 1.0 / nodes.len() as f32;
            for &nd in nodes { for k in 0..hg { ce[c * hg + k] += hl[nd as usize * hg + k] * inv; } }
        }
        Enc { xs, ce, cache: if keep { Some(cache) } else { None } }
    }

    fn gnn_backward(&self, grad: &mut [f32], g: &Graph, cache: &GCache, dxs: &[f32], dce: &[f32]) {
        let (l, hg, gl) = (self.l, self.arch.hg, self.arch.gl);
        let th = &self.theta;
        let nn = g.n;
        if nn == 0 { return; }
        let mut dh = vec![0f32; nn * hg];
        let inv = 1.0 / nn as f32;
        for i in 0..nn { for k in 0..hg { dh[i * hg + k] += dxs[SF + k] * inv; } }
        if g.ant >= 0 { for k in 0..hg { dh[g.ant as usize * hg + k] += dxs[SF + hg + k]; } }
        if g.foe >= 0 { for k in 0..hg { dh[g.foe as usize * hg + k] += dxs[SF + 2 * hg + k]; } }
        for (c, nodes) in g.cand.iter().enumerate() {
            if nodes.is_empty() || (c + 1) * hg > dce.len() { continue; }
            let iv = 1.0 / nodes.len() as f32;
            for &nd in nodes { for k in 0..hg { dh[nd as usize * hg + k] += dce[c * hg + k] * iv; } }
        }
        let mut dz = vec![0f32; nn * hg];
        for k in (0..gl).rev() {
            let base = l.gl0 + k * l.lsz;
            let (wself, we) = (&th[base..base + hg * hg], &th[base + hg * hg..base + 4 * hg * hg]);
            let h = &cache.h[k]; let z = &cache.z[k];
            for i in 0..nn * hg { dz[i] = if z[i] > 0.0 { dh[i] } else { 0.0 }; }
            // dh currently holds d(h_next) = identity path; accumulate the layer's contribution on top
            for i in 0..nn {
                let dzi = &dz[i * hg..(i + 1) * hg];
                for kk in 0..hg { grad[base + 4 * hg * hg + kk] += dzi[kk]; }
                let hi = &h[i * hg..(i + 1) * hg];
                for a in 0..hg {
                    let x = hi[a];
                    let (gw, wr) = (&mut grad[base + a * hg..base + (a + 1) * hg], &wself[a * hg..(a + 1) * hg]);
                    let mut acc = 0f32;
                    for b in 0..hg { if x != 0.0 { gw[b] += x * dzi[b]; } acc += wr[b] * dzi[b]; }
                    dh[i * hg + a] += acc;
                }
                for e in 0..3 {
                    let m = g.nbr[i][e]; if m < 0 { continue; }
                    let m = m as usize;
                    let hm = &h[m * hg..(m + 1) * hg];
                    let eb = base + hg * hg + e * hg * hg;
                    for a in 0..hg {
                        let x = hm[a];
                        let (gw, wr) = (&mut grad[eb + a * hg..eb + (a + 1) * hg], &we[e * hg * hg + a * hg..e * hg * hg + (a + 1) * hg]);
                        let mut acc = 0f32;
                        for b in 0..hg { if x != 0.0 { gw[b] += x * dzi[b]; } acc += wr[b] * dzi[b]; }
                        dh[m * hg + a] += acc;
                    }
                }
            }
        }
        // input embedding: h0 = relu(x W + b)
        let h0 = &cache.h[0];
        for i in 0..nn {
            let xi = &g.x[i * NF..(i + 1) * NF];
            for k in 0..hg {
                if h0[i * hg + k] <= 0.0 { continue; }
                let d = dh[i * hg + k];
                if d == 0.0 { continue; }
                grad[l.bin + k] += d;
                for a in 0..NF { if xi[a] != 0.0 { grad[l.win + a * hg + k] += xi[a] * d; } }
            }
        }
    }

    // ---------- policy / value heads ----------
    pub fn forward_policy(&self, e: &Enc, a: &[f32], n: usize, keep: bool) -> Fp {
        if self.arch.hg == 0 && self.arch.h1 == 64 && self.arch.h2 == 32 { return self.forward_policy_c::<64, 32>(&e.xs, a, n, keep); }
        self.forward_policy_dyn(e, a, n, keep)
    }

    /// fixed-size fast path for plain MLPs (const generics let the compiler fully unroll/vectorise the matvecs)
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

    fn forward_policy_dyn(&self, e: &Enc, a: &[f32], n: usize, keep: bool) -> Fp {
        let (h1, h2, hg) = (self.arch.h1, self.arch.h2, self.arch.hg);
        let l = &self.l; let th = &self.theta;
        let mut hs: Vec<f32> = th[l.b1..l.b1 + h1].to_vec();
        addmv(&mut hs, &e.xs, &th[l.ws..l.ws + l.nin * h1], h1);
        let mut logits = vec![0f32; n];
        let mut o1 = if keep { vec![0f32; n * h1] } else { Vec::new() };
        let mut o2 = if keep { vec![0f32; n * h2] } else { Vec::new() };
        let mut a1 = vec![0f32; h1]; let mut a2 = vec![0f32; h2];
        let b2 = &th[l.b2..l.b2 + h2]; let w3 = &th[l.w3..l.w3 + h2];
        for c in 0..n {
            a1.copy_from_slice(&hs);
            addmv(&mut a1, &a[c * AF..(c + 1) * AF], &th[l.wa..l.wa + AF * h1], h1);
            if hg > 0 { addmv(&mut a1, &e.ce[c * hg..(c + 1) * hg], &th[l.wc..l.wc + hg * h1], h1); }
            for j in 0..h1 { if a1[j] < 0.0 { a1[j] = 0.0; } }
            a2.copy_from_slice(b2);
            addmv(&mut a2, &a1, &th[l.w2..l.w2 + h1 * h2], h2);
            let mut z = th[l.b3];
            for k in 0..h2 { if a2[k] < 0.0 { a2[k] = 0.0; } z += a2[k] * w3[k]; }
            logits[c] = z;
            if keep { o1[c * h1..(c + 1) * h1].copy_from_slice(&a1); o2[c * h2..(c + 1) * h2].copy_from_slice(&a2); }
        }
        Fp { logits, h1: o1, h2: o2 }
    }

    pub fn forward_value(&self, e: &Enc, keep: bool) -> Fv {
        let hv = self.arch.hv; let l = &self.l; let th = &self.theta;
        let mut a: Vec<f32> = th[l.bv1..l.bv1 + hv].to_vec();
        addmv(&mut a, &e.xs, &th[l.v1..l.v1 + l.nin * hv], hv);
        let mut v = th[l.bv2];
        let v2 = &th[l.v2..l.v2 + hv];
        let mut aux = 0f32;
        if self.arch.hg > 0 { aux = th[l.bv3]; }
        for j in 0..hv {
            if a[j] < 0.0 { a[j] = 0.0; }
            v += a[j] * v2[j];
            if self.arch.hg > 0 { aux += a[j] * th[l.v3 + j]; }
        }
        Fv { v, aux, act: if keep { a } else { Vec::new() } }
    }

    /// value of the position for its actor
    pub fn value(&self, o: &Obs) -> f32 { self.forward_value(&self.encode(o, false), false).v }

    /// Accumulate gradients into `g` (same layout as theta). dv/daux: d(loss)/d(value), d(loss)/d(aux value).
    pub fn backward(&self, g: &mut [f32], o: &Obs, e: &Enc, fp: &Fp, dlog: &[f32], fv: &Fv, dv: f32, daux: f32) {
        let (h1, h2, hv, hg) = (self.arch.h1, self.arch.h2, self.arch.hv, self.arch.hg);
        let l = &self.l; let th = &self.theta;
        let (n, nin) = (o.n, l.nin);
        let mut dhs = vec![0f32; h1]; let mut dh1 = vec![0f32; h1]; let mut dh2 = vec![0f32; h2];
        let mut dce = vec![0f32; if hg > 0 { n * hg } else { 0 }];
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
                let x = o.a[c * AF + i]; if x == 0.0 { continue; }
                let grow = &mut g[l.wa + i * h1..l.wa + (i + 1) * h1];
                for j in 0..h1 { grow[j] += x * dh1[j]; }
            }
            if hg > 0 {
                for k in 0..hg {
                    let x = e.ce[c * hg + k];
                    let (wr, gr) = (&th[l.wc + k * h1..l.wc + (k + 1) * h1], k * h1);
                    let mut acc = 0f32;
                    for j in 0..h1 { acc += wr[j] * dh1[j]; if x != 0.0 { g[l.wc + gr + j] += x * dh1[j]; } }
                    dce[c * hg + k] += acc;
                }
            }
        }
        let mut dxs = vec![0f32; nin];
        for i in 0..nin {
            let x = e.xs[i];
            if x != 0.0 { let grow = &mut g[l.ws + i * h1..l.ws + (i + 1) * h1]; for j in 0..h1 { grow[j] += x * dhs[j]; } }
            if i >= SF { let wr = &th[l.ws + i * h1..l.ws + (i + 1) * h1]; let mut acc = 0f32; for j in 0..h1 { acc += wr[j] * dhs[j]; } dxs[i] += acc; }
        }
        if !fv.act.is_empty() {
            let vact = &fv.act;
            g[l.bv2] += dv;
            if hg > 0 { g[l.bv3] += daux; }
            let mut da = vec![0f32; hv];
            for j in 0..hv {
                g[l.v2 + j] += dv * vact[j];
                let mut d = dv * th[l.v2 + j];
                if hg > 0 { g[l.v3 + j] += daux * vact[j]; d += daux * th[l.v3 + j]; }
                da[j] = if vact[j] > 0.0 { d } else { 0.0 };
                g[l.bv1 + j] += da[j];
            }
            for i in 0..nin {
                let x = e.xs[i];
                if x != 0.0 { let grow = &mut g[l.v1 + i * hv..l.v1 + (i + 1) * hv]; for j in 0..hv { grow[j] += x * da[j]; } }
                if i >= SF { let wr = &th[l.v1 + i * hv..l.v1 + (i + 1) * hv]; let mut acc = 0f32; for j in 0..hv { acc += wr[j] * da[j]; } dxs[i] += acc; }
            }
        }
        if hg > 0 { if let (Some(gr), Some(c)) = (o.g.as_ref(), e.cache.as_ref()) { self.gnn_backward(g, gr, c, &dxs, &dce); } }
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
