// Port of js/az.js: self-play with MCTS targets and the supervised AZ update.
use crate::agents::heuristic_act;
use crate::engine::*;
use crate::features::*;
use crate::mcts::{pick_by_visits, Mcts};
use crate::nn::{softmax, Net};
use crate::rng::Rng;
use rayon::prelude::*;

/// `vw`: weight of the value loss. With `--vbalance` it is 1/(samples of this player in this game), so every player-game counts
/// equally; otherwise losers (who make ~2x as many decisions) dominate and the value head learns "the side to move is losing",
/// which MCTS turns into a bonus for ending the turn (measured with `amino-train diag`).
pub struct Sample { pub s: [f32; SF], pub a: Vec<f32>, pub n: usize, pub tgt: Vec<f32>, pub z: f32, pub vw: f32 }

pub struct GameOut { pub samples: Vec<Sample>, pub winner: i32, pub turns: i32, pub learner: i32 }

#[derive(Clone, Copy)]
pub struct SpCfg { pub sims: usize, pub tau_moves: usize, pub dirichlet: f64, pub depth: i32, pub qbeta: f64, pub ttemp: f64, pub vbalance: bool, pub vtd: f64 }

pub fn self_play_game(net: &Net, cfg: SpCfg, opp_heuristic: bool, rng: &mut Rng) -> GameOut {
    let num_players = 2;
    let mut s = new_game(num_players, (rng.u64() >> 33) as i32);
    let mcts = Mcts::new(net, cfg.sims, cfg.depth, cfg.dirichlet);
    let learner: i32 = if opp_heuristic { rng.below(num_players) as i32 } else { -1 };
    // (actor, sample, first decision of its turn?, value at the start of the next turn from the actor's view)
    let mut samples: Vec<(usize, Sample, bool, Option<f32>)> = Vec::new();
    let mut open: Vec<usize> = Vec::new(); // samples of the running turn, waiting for the next turn's start value
    let mut last_turn = s.turn_no;
    let mut searched = 0;
    let mut guard = 0;
    while !s.over && guard < 3000 {
        guard += 1;
        if s.turn_no != last_turn {
            // turn-consistent value targets: mid-turn positions learn the value right after the turn ends (see --vtd)
            last_turn = s.turn_no;
            if cfg.vtd > 0.0 && !open.is_empty() {
                let ctx = build_ctx(&s, s.cur);
                let mut sf = [0f32; SF];
                state_features(&s, &ctx, &mut sf);
                let v = net.forward_value(&sf, false).0;
                for &i in &open { let pi = samples[i].0; samples[i].3 = Some(if pi == s.cur { v } else { -v }); }
            }
            open.clear();
        }
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
        let tgt: Vec<f32> = if cfg.qbeta > 0.0 {
            // policy improvement: pi' ∝ prior · exp(beta · (Q(a) - V)); robust with few simulations (visit counts are mostly exploration)
            let mut l: Vec<f64> = r.prior.iter().zip(r.q.iter()).map(|(&p, &q)| (p.max(1e-8) as f64).ln() + cfg.qbeta * (q as f64 - r.value)).collect();
            let mx = l.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut z = 0.0; for x in l.iter_mut() { *x = (*x - mx).exp(); z += *x; }
            l.iter().map(|x| (x / z) as f32).collect()
        } else if cfg.ttemp != 1.0 {
            let w: Vec<f64> = r.visits.iter().map(|&v| (v as f64).powf(1.0 / cfg.ttemp)).collect();
            let z: f64 = w.iter().sum::<f64>().max(1e-12);
            w.iter().map(|x| (x / z) as f32).collect()
        } else { r.visits.iter().map(|&v| (v as f64 / tot) as f32).collect() };
        let anchor = s.actions_this_turn == 0 && s.pending.is_none();
        open.push(samples.len());
        samples.push((pi, Sample { s: sf, a, n: r.acts.len(), tgt, z: 0.0, vw: 1.0 }, anchor, None));
        let tau = if searched < cfg.tau_moves { 1.0 } else { 0.1 };
        searched += 1;
        let idx = pick_by_visits(&r.visits, tau, rng);
        s.apply(r.acts[idx]);
    }
    if !s.over { s.over = true; s.winner = -1; }
    let mut cnt = [0usize; 6];
    for (pi, ..) in &samples { cnt[*pi] += 1; }
    let out = samples.into_iter().map(|(pi, mut x, anchor, boot)| {
        let z = if s.winner < 0 { 0.0 } else if s.winner as usize == pi { 1.0 } else { -1.0 };
        x.z = match boot { Some(b) if !anchor => (cfg.vtd as f32) * b + (1.0 - cfg.vtd as f32) * z, _ => z };
        if cfg.vbalance { x.vw = 1.0 / cnt[pi] as f32; }
        x
    }).collect();
    GameOut { samples: out, winner: s.winner as i32, turns: s.turn_no, learner }
}

/// Minibatch SGD on the replay buffer; each minibatch gradient is computed in parallel on the rayon pool.
pub fn az_update(net: &mut Net, buffer: &[Sample], steps: usize, batch: usize, lr: f64, pcoef: f32, vcoef: f32, rng: &mut Rng) -> (f64, f64) {
    let size = net.size();
    let nthreads = rayon::current_num_threads().max(1);
    let chunk = batch.div_ceil(nthreads).max(1);
    let (mut pl, mut vl) = (0f64, 0f64);
    for _ in 0..steps {
        let idx: Vec<usize> = (0..batch).map(|_| rng.below(buffer.len())).collect();
        let netr: &Net = net;
        let wsum: f32 = idx.iter().map(|&i| buffer[i].vw).sum::<f32>().max(1e-12);
        let r = idx.par_chunks(chunk).map(|ch| {
            let mut g = vec![0f32; size];
            let (mut pl, mut vl) = (0f64, 0f64);
            for &i in ch {
                let d = &buffer[i];
                let fp = netr.forward_policy(&d.s, &d.a, d.n, true);
                let p = softmax(&fp.logits, 1.0);
                let mut dlog = vec![0f32; d.n];
                for c in 0..d.n {
                    dlog[c] = pcoef * (p[c] - d.tgt[c]) / batch as f32;
                    if d.tgt[c] > 1e-8 { pl -= d.tgt[c] as f64 * (p[c].max(1e-8) as f64).ln(); }
                }
                let (v, vact) = netr.forward_value(&d.s, true);
                let err = v - d.z;
                vl += (d.vw * err * err) as f64 * batch as f64 / wsum as f64;
                netr.backward(&mut g, &d.s, &d.a, d.n, &fp, &dlog, &vact, vcoef * 2.0 * err * d.vw / wsum);
            }
            (g, pl, vl)
        }).reduce_with(|mut a, b| { for (x, y) in a.0.iter_mut().zip(b.0.iter()) { *x += *y; } (a.0, a.1 + b.1, a.2 + b.2) }).unwrap();
        pl += r.1; vl += r.2;
        net.adam_step(&r.0, lr, 1.0);
    }
    let cnt = (steps * batch) as f64;
    (pl / cnt, vl / cnt)
}
