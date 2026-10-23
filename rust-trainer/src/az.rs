// Port of js/az.js: self-play with MCTS targets and the supervised AZ update.
use crate::agents::heuristic_act;
use crate::engine::*;
use crate::features::*;
use crate::mcts::{pick_by_visits, Mcts};
use crate::nn::{softmax, Net};
use crate::rng::Rng;
use rayon::prelude::*;

pub struct Sample { pub s: [f32; SF], pub a: Vec<f32>, pub n: usize, pub tgt: Vec<f32>, pub z: f32 }

pub struct GameOut { pub samples: Vec<Sample>, pub winner: i32, pub turns: i32, pub learner: i32 }

#[derive(Clone, Copy)]
pub struct SpCfg { pub sims: usize, pub tau_moves: usize, pub dirichlet: f64, pub depth: i32 }

pub fn self_play_game(net: &Net, cfg: SpCfg, opp_heuristic: bool, rng: &mut Rng) -> GameOut {
    let num_players = 2;
    let mut s = new_game(num_players, (rng.u64() >> 33) as i32);
    let mcts = Mcts::new(net, cfg.sims, cfg.depth, cfg.dirichlet);
    let learner: i32 = if opp_heuristic { rng.below(num_players) as i32 } else { -1 };
    let mut samples: Vec<(usize, Sample)> = Vec::new();
    let mut searched = 0;
    let mut guard = 0;
    while !s.over && guard < 3000 {
        guard += 1;
        let pi = s.actor();
        let acts = s.legal_actions();
        if acts.len() == 1 { s.apply(acts[0]); continue; }
        if opp_heuristic && pi as i32 != learner { let a = heuristic_act(&s, 4.0, rng); s.apply(a); continue; }
        let r = mcts.search(&s, cfg.sims, true, rng);
        let tot: f64 = { let t: u32 = r.visits.iter().sum(); if t == 0 { 1.0 } else { t as f64 } };
        let ctx = build_ctx(&s, pi);
        let mut sf = [0f32; SF];
        state_features(&s, &ctx, &mut sf);
        let a = candidate_matrix(&s, &ctx, &r.acts);
        let tgt: Vec<f32> = r.visits.iter().map(|&v| (v as f64 / tot) as f32).collect();
        samples.push((pi, Sample { s: sf, a, n: r.acts.len(), tgt, z: 0.0 }));
        let tau = if searched < cfg.tau_moves { 1.0 } else { 0.1 };
        searched += 1;
        let idx = pick_by_visits(&r.visits, tau, rng);
        s.apply(r.acts[idx]);
    }
    if !s.over { s.over = true; s.winner = -1; }
    let out = samples.into_iter().map(|(pi, mut x)| {
        x.z = if s.winner < 0 { 0.0 } else if s.winner as usize == pi { 1.0 } else { -1.0 };
        x
    }).collect();
    GameOut { samples: out, winner: s.winner as i32, turns: s.turn_no, learner }
}

/// Minibatch SGD on the replay buffer; each minibatch gradient is computed in parallel on the rayon pool.
pub fn az_update(net: &mut Net, buffer: &[Sample], steps: usize, batch: usize, lr: f64, rng: &mut Rng) -> (f64, f64) {
    let size = net.size();
    let nthreads = rayon::current_num_threads().max(1);
    let chunk = batch.div_ceil(nthreads).max(1);
    let (mut pl, mut vl) = (0f64, 0f64);
    for _ in 0..steps {
        let idx: Vec<usize> = (0..batch).map(|_| rng.below(buffer.len())).collect();
        let netr: &Net = net;
        let r = idx.par_chunks(chunk).map(|ch| {
            let mut g = vec![0f32; size];
            let (mut pl, mut vl) = (0f64, 0f64);
            for &i in ch {
                let d = &buffer[i];
                let fp = netr.forward_policy(&d.s, &d.a, d.n, true);
                let p = softmax(&fp.logits, 1.0);
                let mut dlog = vec![0f32; d.n];
                for c in 0..d.n {
                    dlog[c] = (p[c] - d.tgt[c]) / batch as f32;
                    if d.tgt[c] > 1e-8 { pl -= d.tgt[c] as f64 * (p[c].max(1e-8) as f64).ln(); }
                }
                let (v, vact) = netr.forward_value(&d.s, true);
                let err = v - d.z;
                vl += (err * err) as f64;
                netr.backward(&mut g, &d.s, &d.a, d.n, &fp, &dlog, &vact, 2.0 * err / batch as f32);
            }
            (g, pl, vl)
        }).reduce_with(|mut a, b| { for (x, y) in a.0.iter_mut().zip(b.0.iter()) { *x += *y; } (a.0, a.1 + b.1, a.2 + b.2) }).unwrap();
        pl += r.1; vl += r.2;
        net.adam_step(&r.0, lr, 1.0);
    }
    let cnt = (steps * batch) as f64;
    (pl / cnt, vl / cnt)
}
