// AlphaZero-style trainer for Amino — Rust port of tools/train-az.mjs (same flags, same model JSON).
//
//   cargo run --release -- --init ../models/pretrained.json --out ../models/az.json --sims 64 --games 24 --minutes 240
//
// Options: --workers N (default all cores) --sims --games (per iteration) --steps (SGD minibatches / iter) --batch --lr
//          --buffer (replay size) --eval-every N iters --eval-games N --opp-mix 0.2 --depth 18 --tau-moves 30
//          --minutes / --iters  --resume  --seed N
// Extra:   `xcheck <model.json> <nSeeds>` prints deterministic game traces for cross-checking against tools/xcheck.mjs
mod agents; mod az; mod engine; mod features; mod fv; mod mcts; mod nn; mod rng;

use agents::*;
use az::*;
use features::*;
use nn::*;
use rayon::prelude::*;
use rng::Rng;
use serde_json::{json, Value};
use std::time::Instant;

fn arg(k: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    let key = format!("--{}", k);
    a.iter().position(|x| *x == key).map(|i| match a.get(i + 1) { Some(v) if !v.starts_with("--") => v.clone(), _ => "true".into() })
}
fn argf(k: &str, d: f64) -> f64 { arg(k).and_then(|v| v.parse().ok()).unwrap_or(d) }

fn load_net(path: &str) -> (Net, Value) {
    let j: Value = serde_json::from_str(&std::fs::read_to_string(path).expect("read model")).expect("parse model");
    let a = &j["net"]["arch"];
    let arch = Arch { h1: a["h1"].as_u64().unwrap() as usize, h2: a["h2"].as_u64().unwrap() as usize, hv: a["hv"].as_u64().unwrap() as usize };
    let theta: Vec<f32> = j["net"]["theta"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    (Net::from_theta(arch, theta), j)
}

fn fmt6(x: f32) -> String {
    let v: f64 = format!("{:.6}", x).parse().unwrap();
    if v == 0.0 { "0".into() } else { format!("{}", v) }
}

fn save(path: &str, net: &Net, sims: usize, games: u64, iter: u64, history: &[Value]) {
    let mut s = String::with_capacity(net.theta.len() * 10 + 1024);
    s.push_str(&format!("{{\"version\":2,\"name\":\"az\",\"kind\":\"mcts\",\"sims\":{},\"games\":{},\"iter\":{},\"history\":{},\"net\":{{\"arch\":{{\"h1\":{},\"h2\":{},\"hv\":{}}},\"theta\":[",
        sims, games, iter, serde_json::to_string(history).unwrap(), net.arch.h1, net.arch.h2, net.arch.hv));
    for (i, x) in net.theta.iter().enumerate() { if i > 0 { s.push(','); } s.push_str(&fmt6(*x)); }
    s.push_str("]}}");
    let tmp = format!("{}.tmp", path);
    std::fs::write(&tmp, s).unwrap();
    std::fs::rename(&tmp, path).unwrap();
}

struct Ev { w: u32, d: u32, l: u32 }
impl Ev { fn rate(&self) -> f64 { (self.w as f64 + 0.5 * self.d as f64) / ((self.w + self.d + self.l).max(1) as f64) } }

fn eval(net: &Net, games: usize, search: bool, sims: usize, depth: i32, base: u64) -> Ev {
    let res: Vec<i32> = (0..games).into_par_iter().map(|g| {
        let mut rng = Rng::new(base ^ (g as u64).wrapping_mul(0x9E3779B97F4A7C15));
        let seat = g % 2;
        let me = if search { Agent::Search(mcts::Mcts::new(net, sims.max(8), depth, 0.0)) } else { Agent::Nn(net, 0.5) };
        let ag = if seat == 0 { [me, Agent::Heur(4.0)] } else { [Agent::Heur(4.0), me] };
        let w = play_game(&ag, &mut rng);
        if w < 0 { 0 } else if w as usize == seat { 1 } else { -1 }
    }).collect();
    Ev { w: res.iter().filter(|&&r| r == 1).count() as u32, d: res.iter().filter(|&&r| r == 0).count() as u32, l: res.iter().filter(|&&r| r == -1).count() as u32 }
}

fn xcheck(path: &str, n: i32) {
    let (net, _) = load_net(path);
    let mut rng = Rng::new(1);
    for seed in 1..=n {
        let mut s = engine::new_game(2, seed);
        let (mut h, mut steps, mut guard) = (0u32, 0u32, 0);
        let (mut sfs, mut asum, mut lsum, mut vsum) = (0f64, 0f64, 0f64, 0f64);
        while !s.over && guard < 6000 {
            guard += 1;
            let pi = s.actor();
            let acts = s.legal_actions();
            let a;
            if acts.len() == 1 { a = acts[0]; } else {
                a = heuristic_act(&s, 0.0, &mut rng);
                let ctx = build_ctx(&s, pi);
                let mut sf = [0f32; SF];
                state_features(&s, &ctx, &mut sf);
                let m = candidate_matrix(&s, &ctx, &acts);
                sfs += sf.iter().map(|x| *x as f64).sum::<f64>();
                asum += m.iter().map(|x| *x as f64).sum::<f64>();
                lsum += net.forward_policy(&sf, &m, acts.len(), false).logits.iter().map(|x| *x as f64).sum::<f64>();
                vsum += net.forward_value(&sf, false).0 as f64;
            }
            h = h.wrapping_mul(31).wrapping_add((acts.len() * 7) as u32 + a.t as u32);
            steps += 1;
            s.apply(a);
        }
        let w = if s.over { s.winner as i32 } else { -1 };
        println!("{} {} {} {} {} {:.2} {:.2} {:.3} {:.3}", seed, w, s.turn_no, steps, h, sfs, asum, lsum, vsum);
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    if argv.get(1).map(|s| s.as_str()) == Some("xcheck") { xcheck(&argv[2], argv[3].parse().unwrap()); return; }

    let ncpu = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let workers = argf("workers", ncpu as f64) as usize;
    let sims = argf("sims", 64.0) as usize;
    let games = argf("games", 24.0) as usize;
    let steps = argf("steps", 150.0) as usize; // upper bound of minibatches per iteration
    let pcoef = argf("pcoef", 1.0) as f32; let vcoef = argf("vcoef", 1.0) as f32; // loss weights (0 freezes that head)
    let ratio = argf("ratio", 4.0); // samples drawn per newly generated sample (limits over-fitting of the replay buffer)
    let min_buffer = argf("min-buffer", 5000.0) as usize; // no updates until the buffer holds this many samples
    mcts::TOPK.store(argf("topk", 12.0) as usize, std::sync::atomic::Ordering::Relaxed);
    mcts::CPUCT_MILLI.store((argf("cpuct", 1.25) * 1000.0) as usize, std::sync::atomic::Ordering::Relaxed);
    let batch = argf("batch", 128.0) as usize;
    let lr = argf("lr", 3e-4);
    let buffer_cap = argf("buffer", 40000.0) as usize;
    let eval_every = argf("eval-every", 5.0) as u64;
    let eval_games = argf("eval-games", 16.0) as usize;
    let opp_mix = argf("opp-mix", 0.2);
    let depth = argf("depth", 18.0) as i32;
    let minutes = argf("minutes", 0.0);
    let iters = argf("iters", 100000.0) as u64;
    let tau_moves = argf("tau-moves", 30.0) as usize;
    let out = arg("out").unwrap_or("models/az.json".into());
    let init = arg("init").or_else(|| if std::path::Path::new("models/pretrained.json").exists() { Some("models/pretrained.json".into()) } else { None });
    let seed = argf("seed", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as f64 % 1e15) as u64;

    rayon::ThreadPoolBuilder::new().num_threads(workers).build_global().unwrap();
    let mut rng = Rng::new(seed);
    let (mut games_total, mut iter0, mut history): (u64, u64, Vec<Value>) = (0, 0, Vec::new());
    let mut net = match &init {
        Some(p) if std::path::Path::new(p).exists() => {
            let (n, j) = load_net(p);
            if arg("resume").is_some() {
                games_total = j["games"].as_u64().unwrap_or(0); iter0 = j["iter"].as_u64().unwrap_or(0);
                history = j["history"].as_array().cloned().unwrap_or_default();
            }
            println!("init from {}", p); n
        }
        _ => { println!("init: random network (consider warm-starting with tools/train-cli.mjs first)"); Net::random(DEFAULT_ARCH, &mut rng) }
    };
    println!("config {}", json!({"workers": workers, "sims": sims, "games": games, "steps": steps, "batch": batch, "lr": lr, "buffer": buffer_cap,
        "evalEvery": eval_every, "evalGames": eval_games, "oppMix": opp_mix, "depth": depth, "minutes": minutes, "iters": iters, "tauMoves": tau_moves, "seed": seed}));

    let t0 = Instant::now();
    let mut buffer: Vec<Sample> = Vec::new();
    let mut best = -1.0f64;
    let cfg = SpCfg { sims, tau_moves, dirichlet: 0.25, depth, qbeta: argf("qbeta", 3.0), ttemp: argf("ttemp", 1.0) };
    for _ in 0..iters {
        if minutes > 0.0 && t0.elapsed().as_secs_f64() / 60.0 > minutes { break; }
        let t1 = Instant::now();
        let base = rng.u64();
        let outs: Vec<(GameOut, bool)> = {
            let netr = &net;
            (0..games).into_par_iter().map(|g| {
                let mut r = Rng::new(base ^ (g as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15));
                let heur = r.f64() < opp_mix;
                (self_play_game(netr, cfg, heur, &mut r), heur)
            }).collect()
        };
        let (mut turns, mut n_games) = (0i64, 0u64);
        let mut new_samples = 0usize;
        let (mut vw, mut vd, mut vl) = (0, 0, 0);
        for (o, heur) in outs {
            turns += o.turns as i64; n_games += 1;
            if heur { if o.winner < 0 { vd += 1 } else if o.winner == o.learner { vw += 1 } else { vl += 1 } }
            new_samples += o.samples.len();
            buffer.extend(o.samples);
        }
        if buffer.len() > buffer_cap { let ex = buffer.len() - buffer_cap; buffer.drain(0..ex); }
        let t_play = t1.elapsed().as_secs_f64();
        let t2 = Instant::now();
        let steps_eff = if buffer.len() < min_buffer { 0 } else { (((new_samples as f64 * ratio) / batch as f64).ceil() as usize).clamp(1, steps) };
        let (ploss, vloss) = if steps_eff > 0 { az_update(&mut net, &buffer, steps_eff, batch, lr, pcoef, vcoef, &mut rng) } else { (0.0, 0.0) };
        let t_train = t2.elapsed().as_secs_f64();
        games_total += n_games; iter0 += 1;
        println!("iter {} games {} buf {} steps {} turns/game {:.0} play {:.0}s train {:.1}s  ploss {:.3} vloss {:.3}  vsHeur(self-play seats) {}-{}-{}",
            iter0, games_total, buffer.len(), steps_eff, turns as f64 / n_games as f64, t_play, t_train, ploss, vloss, vw, vd, vl);
        save(&out, &net, sims, games_total, iter0, &history);
        if eval_every > 0 && iter0 % eval_every == 0 {
            let raw = eval(&net, eval_games, false, sims, depth, rng.u64());
            let srch = eval(&net, eval_games, true, sims, depth, rng.u64());
            history.push(json!({"iter": iter0, "games": games_total, "random": raw.rate(), "heuristic": srch.rate(), "rawHeuristic": raw.rate()}));
            println!("EVAL iter {}: raw net vs heuristic {:.0}% | MCTS({}) vs heuristic {:.0}%", iter0, raw.rate() * 100.0, sims, srch.rate() * 100.0);
            if srch.rate() >= best { best = srch.rate(); save(&format!("{}.best.json", out.trim_end_matches(".json")), &net, sims, games_total, iter0, &history); }
            save(&out, &net, sims, games_total, iter0, &history);
        }
    }
    println!("done {:.1} min", t0.elapsed().as_secs_f64() / 60.0);
}
