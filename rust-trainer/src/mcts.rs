// Port of js/mcts.js: PUCT + determinization, open-loop tree keyed by action.
use crate::engine::*;
use crate::features::*;
use crate::nn::{softmax, Net};
use crate::rng::Rng;

const NONE: u32 = u32::MAX;

struct Child { act: Act, n: u32, w: f64, p: f32, node: u32 }
struct Node { n: u32, w: f64, expanded: bool, children: Vec<Child> }
impl Node { fn new() -> Node { Node { n: 0, w: 0.0, expanded: false, children: Vec::new() } } }

pub struct SearchResult { pub acts: Vec<Act>, pub visits: Vec<u32>, pub value: f64 }

pub struct Mcts<'a> {
    pub net: &'a Net, pub sims: usize, pub cpuct: f64, pub max_depth: i32, pub max_cands: usize,
    pub dirichlet: f64, pub dir_alpha: f64, pub fpu: f64,
}

pub fn determinize(s: &mut State, observer: usize, rng: &mut Rng) {
    let mut pool: Vec<u8> = s.card_stack.iter().copied().collect();
    for q in 0..s.n { if q != observer { pool.extend(s.players[q].cards.iter().copied()); } }
    rng.shuffle(&mut pool);
    let mut off = 0;
    for q in 0..s.n {
        if q != observer {
            let k = s.players[q].cards.len();
            s.players[q].cards.clear();
            for i in 0..k { s.players[q].cards.push(pool[off + i]); }
            off += k;
        }
    }
    s.card_stack.clear();
    for &c in &pool[off..] { s.card_stack.push(c); }
    rng.shuffle(s.tile_stack.as_mut_slice());
    rng.shuffle(s.res_stack.as_mut_slice());
    if let Some(pd) = s.pending.as_mut() {
        if pd.kind == P_GUESS && pd.att as usize == observer { pd.hide = if rng.f64() < 0.5 { 0 } else { 1 }; }
    }
    s.seed = (rng.u64() >> 33) as i32;
}

pub fn cap_deterministic(acts: Vec<Act>, max: usize) -> Vec<Act> {
    if acts.len() <= max { return acts; }
    let mut keep = Vec::with_capacity(max); let mut bulk = Vec::with_capacity(acts.len());
    for a in acts { if a.t == A_PLACE || a.t == A_OVERLAY { bulk.push(a); } else { keep.push(a); } }
    let room = std::cmp::max(4, max.saturating_sub(keep.len()));
    if bulk.len() <= room { keep.extend(bulk); return keep; }
    for i in 0..room { keep.push(bulk[(i * bulk.len()) / room]); }
    keep
}

fn gamma_sample(k: f64, rng: &mut Rng) -> f64 {
    if k < 1.0 { return gamma_sample(k + 1.0, rng) * rng.f64().powf(1.0 / k); }
    let d = k - 1.0 / 3.0; let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let (mut x, mut v);
        loop {
            let u1 = rng.f64().max(1e-12); let u2 = rng.f64();
            x = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos(); v = 1.0 + c * x;
            if v > 0.0 { break; }
        }
        v = v * v * v;
        let u = rng.f64().max(1e-12);
        if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() { return d * v; }
    }
}

impl<'a> Mcts<'a> {
    pub fn new(net: &'a Net, sims: usize, max_depth: i32, dirichlet: f64) -> Mcts<'a> {
        Mcts { net, sims, cpuct: 1.5, max_depth, max_cands: 40, dirichlet, dir_alpha: 0.3, fpu: 0.0 }
    }

    fn evaluate(&self, s: &State, acts: &[Act], want_priors: bool) -> (f32, Vec<f32>) {
        let ctx = build_ctx(s, s.actor());
        let mut sf = [0f32; SF];
        state_features(s, &ctx, &mut sf);
        let (v, _) = self.net.forward_value(&sf, false);
        let priors = if want_priors {
            let a = candidate_matrix(s, &ctx, acts);
            softmax(&self.net.forward_policy(&sf, &a, acts.len(), false).logits, 1.0)
        } else { Vec::new() };
        (v, priors)
    }

    fn advance_forced(&self, s: &mut State) -> Vec<Act> {
        let mut guard = 0;
        while !s.over && guard < 200 {
            guard += 1;
            let la = s.legal_actions();
            if la.len() != 1 { return la; }
            s.apply(la[0]);
        }
        if s.over { Vec::new() } else { s.legal_actions() }
    }

    pub fn search(&self, s0: &State, sims: usize, noise: bool, rng: &mut Rng) -> SearchResult {
        let observer = s0.actor();
        let mut nodes: Vec<Node> = Vec::with_capacity(sims * 4 + 4);
        nodes.push(Node::new());
        let mut root_acts: Vec<Act> = Vec::new();
        for i in 0..sims {
            let mut s = *s0;
            determinize(&mut s, observer, rng);
            self.simulate(&mut nodes, &mut root_acts, &mut s, observer, noise && i == 0, rng);
        }
        let root = &nodes[0];
        let visits = root_acts.iter().map(|a| root.children.iter().find(|c| c.act == *a).map_or(0, |c| c.n)).collect();
        SearchResult { acts: root_acts, visits, value: if root.n > 0 { root.w / root.n as f64 } else { 0.0 } }
    }

    fn simulate(&self, nodes: &mut Vec<Node>, root_acts: &mut Vec<Act>, s: &mut State, observer: usize, add_noise: bool, rng: &mut Rng) {
        let mut path: Vec<(usize, usize)> = Vec::with_capacity(self.max_depth as usize + 2);
        let mut node = 0usize; let mut depth = 0; let value: f64;
        let mut acts = self.advance_forced(s);
        loop {
            if s.over { value = if s.winner < 0 { 0.0 } else if s.winner as usize == observer { 1.0 } else { -1.0 }; break; }
            if depth >= self.max_depth {
                let (v, _) = self.evaluate(s, &acts, false);
                value = if s.actor() == observer { v as f64 } else { -(v as f64) };
                break;
            }
            acts = cap_deterministic(acts, self.max_cands);
            if !nodes[node].expanded {
                let (v, mut pr) = self.evaluate(s, &acts, true);
                if add_noise && node == 0 {
                    let g: Vec<f64> = acts.iter().map(|_| gamma_sample(self.dir_alpha, rng)).collect();
                    let gs: f64 = { let t: f64 = g.iter().sum(); if t == 0.0 { 1.0 } else { t } };
                    for (i, p) in pr.iter_mut().enumerate() { *p = ((1.0 - self.dirichlet) * *p as f64 + self.dirichlet * g[i] / gs) as f32; }
                }
                let nd = &mut nodes[node];
                nd.expanded = true;
                for (i, a) in acts.iter().enumerate() { nd.children.push(Child { act: *a, n: 0, w: 0.0, p: pr[i], node: NONE }); }
                if node == 0 { *root_acts = acts.clone(); }
                value = if s.actor() == observer { v as f64 } else { -(v as f64) };
                break;
            }
            let sign = if s.actor() == observer { 1.0 } else { -1.0 };
            let (pn, pw) = (nodes[node].n, nodes[node].w);
            let sqrt_n = ((pn + 1) as f64).sqrt();
            let parent_q = if pn > 0 { pw / pn as f64 } else { 0.0 };
            let mut best = usize::MAX; let mut best_score = f64::NEG_INFINITY; let mut best_act = acts[0];
            for a in acts.iter() {
                let ci = match nodes[node].children.iter().position(|c| c.act == *a) {
                    Some(i) => i,
                    None => { nodes[node].children.push(Child { act: *a, n: 0, w: 0.0, p: 1e-3, node: NONE }); nodes[node].children.len() - 1 }
                };
                let c = &nodes[node].children[ci];
                let q = if c.n > 0 { c.w / c.n as f64 } else { parent_q - sign * self.fpu };
                let score = sign * q + self.cpuct * c.p as f64 * sqrt_n / (1.0 + c.n as f64);
                if score > best_score { best_score = score; best = ci; best_act = *a; }
            }
            path.push((node, best));
            s.apply(best_act);
            depth += 1;
            if nodes[node].children[best].node == NONE {
                nodes.push(Node::new());
                let id = (nodes.len() - 1) as u32;
                nodes[node].children[best].node = id;
            }
            node = nodes[node].children[best].node as usize;
            acts = self.advance_forced(s);
        }
        for &(nd, ci) in &path {
            let n = &mut nodes[nd]; n.n += 1; n.w += value;
            let c = &mut n.children[ci]; c.n += 1; c.w += value;
        }
        if path.is_empty() { nodes[0].n += 1; nodes[0].w += value; }
    }
}

pub fn pick_by_visits(visits: &[u32], tau: f64, rng: &mut Rng) -> usize {
    if tau <= 0.05 {
        let mut b = 0;
        for i in 1..visits.len() { if visits[i] > visits[b] { b = i; } }
        return b;
    }
    let w: Vec<f64> = visits.iter().map(|&v| (v as f64).powf(1.0 / tau)).collect();
    let z: f64 = w.iter().sum();
    if z == 0.0 { return 0; }
    let mut r = rng.f64() * z;
    for (i, x) in w.iter().enumerate() { r -= x; if r <= 0.0 { return i; } }
    w.len() - 1
}
