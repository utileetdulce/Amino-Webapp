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
use engine::RULES_VERSION;
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

fn civil(z: i64) -> (i64, i64, i64) {
    let z = z + 719468; let era = z.div_euclid(146097); let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1; let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}
fn iso_now() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
    let (y, m, d) = civil(secs.div_euclid(86400)); let r = secs.rem_euclid(86400);
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.000Z", y, m, d, r / 3600, r % 3600 / 60, r % 60)
}
fn git_commit() -> Value {
    std::process::Command::new("git").args(["rev-parse", "--short", "HEAD"]).output().ok()
        .and_then(|o| if o.status.success() { String::from_utf8(o.stdout).ok() } else { None }).map(|s| json!(s.trim())).unwrap_or(Value::Null)
}
/// Model metadata (see js/modelmeta.js): who/when/how this model was trained and under which rules version.
fn meta_now(base: &Value, base_minutes: f64, elapsed_min: f64) -> Value {
    let mut m = base.clone();
    m["updatedAt"] = json!(iso_now());
    m["trainMinutes"] = json!(base_minutes + elapsed_min);
    m["rulesVersion"] = json!(RULES_VERSION);
    m
}

fn fmt6(x: f32) -> String {
    let v: f64 = format!("{:.6}", x).parse().unwrap();
    if v == 0.0 { "0".into() } else { format!("{}", v) }
}

fn save(path: &str, net: &Net, sims: usize, games: u64, iter: u64, history: &[Value], meta: &Value) {
    let mut s = String::with_capacity(net.theta.len() * 10 + 1024);
    s.push_str(&format!("{{\"version\":3,\"name\":{},\"kind\":\"mcts\",\"sims\":{},\"games\":{},\"iter\":{},\"history\":{},\"meta\":{},\"net\":{{\"arch\":{{\"h1\":{},\"h2\":{},\"hv\":{}}},\"theta\":[",
        serde_json::to_string(&meta["name"]).unwrap(), sims, games, iter, serde_json::to_string(history).unwrap(), serde_json::to_string(meta).unwrap(), net.arch.h1, net.arch.h2, net.arch.hv));
    for (i, x) in net.theta.iter().enumerate() { if i > 0 { s.push(','); } s.push_str(&fmt6(*x)); }
    s.push_str("]}}");
    let tmp = format!("{}.tmp", path);
    std::fs::write(&tmp, s).unwrap();
    std::fs::rename(&tmp, path).unwrap();
}

struct Ev { w: u32, d: u32, l: u32 }
impl Ev { fn rate(&self) -> f64 { (self.w as f64 + 0.5 * self.d as f64) / ((self.w + self.d + self.l).max(1) as f64) } }

fn mk_agent<'a>(n: &'a Net, search: bool, sims: usize, depth: i32) -> Agent<'a> {
    if search { Agent::Search(mcts::Mcts::new(n, sims.max(8), depth, 0.0)) } else { Agent::Nn(n, 0.5) }
}

/// Head-to-head (current vs champion); `search` plays both sides with MCTS, otherwise with the raw policy. Returns A's record.
fn eval_h2h(a: &Net, b: &Net, games: usize, search: bool, sims: usize, depth: i32, base: u64) -> Ev {
    let res: Vec<i32> = (0..games).into_par_iter().map(|g| {
        let mut rng = Rng::new(base ^ (g as u64).wrapping_mul(0x9E3779B97F4A7C15));
        let seat = g % 2;
        let (ma, mb) = (mk_agent(a, search, sims, depth), mk_agent(b, search, sims, depth));
        let ag = if seat == 0 { [ma, mb] } else { [mb, ma] };
        let w = play_game(&ag, &mut rng);
        if w < 0 { 0 } else if w as usize == seat { 1 } else { -1 }
    }).collect();
    Ev { w: res.iter().filter(|&&r| r == 1).count() as u32, d: res.iter().filter(|&&r| r == 0).count() as u32, l: res.iter().filter(|&&r| r == -1).count() as u32 }
}

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
    let gate_every = argf("gate-every", 20.0) as u64; // champion gating: every N iterations play current vs champion (0 = off)
    let gate_games = argf("gate-games", 200.0) as usize;
    let gate_win = argf("gate-win", 0.56); // promote the current net to champion at >= this score
    let gate_revert = argf("gate-revert", 0.42); // go back to the champion if the score is below this (stops drift); 0 = never
    let ratio = argf("ratio", 4.0); // samples drawn per newly generated sample (limits over-fitting of the replay buffer)
    let min_buffer = argf("min-buffer", 5000.0) as usize; // no updates until the buffer holds this many samples
    mcts::TOPK.store(argf("topk", 12.0) as usize, std::sync::atomic::Ordering::Relaxed);
    mcts::CPUCT_MILLI.store((argf("cpuct", 1.25) * 1000.0) as usize, std::sync::atomic::Ordering::Relaxed);
    let batch = argf("batch", 128.0) as usize;
    let lr = argf("lr", 5e-5); // long runs at >= 1e-4 drifted and got weaker (see README)
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
    let mut parent_info = Value::Null; let mut base_minutes = 0.0f64;
    let mut net = match &init {
        Some(p) if std::path::Path::new(p).exists() => {
            let (n, j) = load_net(p);
            if let Some(pm) = j.get("meta") {
                parent_info = json!({"id": pm["id"], "name": pm["name"], "updatedAt": pm["updatedAt"], "rulesVersion": pm["rulesVersion"], "games": j["games"]});
                if arg("resume").is_some() { base_minutes = pm["trainMinutes"].as_f64().unwrap_or(0.0); }
            }
            if j["meta"]["rulesVersion"].as_u64().unwrap_or(0) < RULES_VERSION as u64 { println!("WARNING: parent model was trained under rules v{} (current v{}). Warm-starting is usually fine; start fresh only if features/architecture changed.", j["meta"]["rulesVersion"], RULES_VERSION); }
            if j.get("meta").is_none() { parent_info = json!({"id": null, "name": p, "updatedAt": null, "rulesVersion": null, "games": j["games"]}); }
            if arg("resume").is_some() {
                games_total = j["games"].as_u64().unwrap_or(0); iter0 = j["iter"].as_u64().unwrap_or(0);
                history = j["history"].as_array().cloned().unwrap_or_default();
            }
            println!("init from {}", p); n
        }
        _ => { println!("init: random network (consider warm-starting with tools/train-cli.mjs first)"); Net::random(DEFAULT_ARCH, &mut rng) }
    };
    if lr > 1.0e-4 { println!("WARNING: lr {} is high - in our tests runs with lr >= 1e-4 drifted and ended up weaker than the start model (use <= 5e-5 and gating).", lr); }
    println!("config {}", json!({"workers": workers, "sims": sims, "games": games, "steps": steps, "batch": batch, "lr": lr, "buffer": buffer_cap,
        "evalEvery": eval_every, "evalGames": eval_games, "oppMix": opp_mix, "depth": depth, "minutes": minutes, "iters": iters, "tauMoves": tau_moves, "seed": seed}));

    let t0 = Instant::now();
    let stem = std::path::Path::new(&out).file_stem().and_then(|x| x.to_str()).unwrap_or("az").to_string();
    let now0 = iso_now();
    let meta0 = json!({
        "id": format!("m-{:x}-{:04x}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis(), seed & 0xffff),
        "name": arg("name").unwrap_or(stem), "kind": "az", "trainer": "rust", "createdAt": now0, "updatedAt": now0, "rulesVersion": RULES_VERSION,
        "parent": parent_info,
        "hyper": {"sims": sims, "topk": mcts::TOPK.load(std::sync::atomic::Ordering::Relaxed), "cpuct": argf("cpuct", 1.25), "qbeta": argf("qbeta", 3.0), "games": games, "lr": lr, "ratio": ratio,
                  "minBuffer": min_buffer, "batch": batch, "oppMix": opp_mix, "depth": depth, "tauMoves": tau_moves},
        "host": {"os": std::env::consts::OS, "arch": std::env::consts::ARCH, "cpus": ncpu, "workers": workers},
        "git": git_commit(), "trainMinutes": base_minutes, "note": ""
    });
    let mut buffer: Vec<Sample> = Vec::new();
    let mut best = -1.0f64;
    let mut champion = net.clone(); let mut champ_iter = iter0;
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
        save(&out, &net, sims, games_total, iter0, &history, &meta_now(&meta0, base_minutes, t0.elapsed().as_secs_f64() / 60.0));
        if eval_every > 0 && iter0 % eval_every == 0 {
            let raw = eval(&net, eval_games, false, sims, depth, rng.u64());
            let srch = eval(&net, eval_games, true, sims, depth, rng.u64());
            history.push(json!({"iter": iter0, "games": games_total, "random": raw.rate(), "heuristic": srch.rate(), "rawHeuristic": raw.rate()}));
            println!("EVAL iter {}: raw net vs heuristic {:.0}% | MCTS({}) vs heuristic {:.0}%", iter0, raw.rate() * 100.0, sims, srch.rate() * 100.0);
            if srch.rate() >= best { best = srch.rate(); save(&format!("{}.best.json", out.trim_end_matches(".json")), &net, sims, games_total, iter0, &history, &meta_now(&meta0, base_minutes, t0.elapsed().as_secs_f64() / 60.0)); }
            save(&out, &net, sims, games_total, iter0, &history, &meta_now(&meta0, base_minutes, t0.elapsed().as_secs_f64() / 60.0));
        }
        if gate_every > 0 && iter0 % gate_every == 0 {
            let g = eval_h2h(&net, &champion, gate_games, false, sims, depth, rng.u64());
            let rate = g.rate();
            let mut verdict = "kept".to_string();
            if rate >= gate_win {
                champion = net.clone(); champ_iter = iter0; verdict = "PROMOTED".into();
                let mut m = meta_now(&meta0, base_minutes, t0.elapsed().as_secs_f64() / 60.0);
                m["note"] = json!(format!("Champion nach Gating (Iter. {}, {:.0} % gegen den vorigen Champion)", iter0, rate * 100.0));
                save(&format!("{}.champion.json", out.trim_end_matches(".json")), &net, sims, games_total, iter0, &history, &m);
            } else if gate_revert > 0.0 && rate < gate_revert {
                net = champion.clone(); verdict = format!("REVERTED to champion of iter {}", champ_iter);
            }
            println!("GATE iter {}: current vs champion(iter {}) {:.0}% ({}-{}-{}) -> {}", iter0, champ_iter, rate * 100.0, g.w, g.d, g.l, verdict);
        }
    }
    println!("done {:.1} min", t0.elapsed().as_secs_f64() / 60.0);
}
