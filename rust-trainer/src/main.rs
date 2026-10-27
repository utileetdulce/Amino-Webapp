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
    let arch = Arch { h1: a["h1"].as_u64().unwrap() as usize, h2: a["h2"].as_u64().unwrap() as usize, hv: a["hv"].as_u64().unwrap() as usize, hg: a["hg"].as_u64().unwrap_or(0) as usize, gl: a["gl"].as_u64().unwrap_or(0) as usize, fv: a["fv"].as_u64().unwrap_or(if a["hg"].as_u64().unwrap_or(0) > 0 { 1 } else { 0 }) as usize };
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
    s.push_str(&format!("{{\"version\":3,\"name\":{},\"kind\":\"mcts\",\"sims\":{},\"games\":{},\"iter\":{},\"history\":{},\"meta\":{},\"net\":{{\"arch\":{{\"h1\":{},\"h2\":{},\"hv\":{}{}}},\"theta\":[",
        serde_json::to_string(&meta["name"]).unwrap(), sims, games, iter, serde_json::to_string(history).unwrap(), serde_json::to_string(meta).unwrap(), net.arch.h1, net.arch.h2, net.arch.hv, if net.arch.hg > 0 { format!(",\"hg\":{},\"gl\":{},\"fv\":{}", net.arch.hg, net.arch.gl, net.arch.fv) } else { String::new() }));
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
                let o = make_obs(&s, &ctx, &acts, net.gver());
                let e = net.encode(&o, false);
                sfs += o.s.iter().map(|x| *x as f64).sum::<f64>();
                asum += o.a.iter().map(|x| *x as f64).sum::<f64>();
                lsum += net.forward_policy(&e, &o.a, o.n, false).logits.iter().map(|x| *x as f64).sum::<f64>();
                vsum += net.forward_value(&e, false).v as f64;
            }
            h = h.wrapping_mul(31).wrapping_add((acts.len() * 7) as u32 + a.t as u32);
            steps += 1;
            s.apply(a);
        }
        let w = if s.over { s.winner as i32 } else { -1 };
        println!("{} {} {} {} {} {:.2} {:.2} {:.3} {:.3}", seed, w, s.turn_no, steps, h, sfs, asum, lsum, vsum);
    }
}

/// Agent spec for `bench`: heur | raw:<model>:<temp, 0=greedy> | mcts:<model>:<sims>[:topk=12][:cpuct=1.25][:depth=18][:v0][:nodet]
fn parse_agent<'a>(spec: &str, nets: &'a std::collections::HashMap<String, Net>) -> Agent<'a> {
    let p: Vec<&str> = spec.split(':').collect();
    match p[0] {
        "heur" => Agent::Heur(4.0),
        "raw" => Agent::Nn(&nets[p[1]], p.get(2).map_or(0.5, |t| t.parse().unwrap())),
        "mcts" => {
            let mut m = mcts::Mcts::new(&nets[p[1]], p[2].parse().unwrap(), 18, 0.0);
            for kv in &p[3..] {
                let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
                match k {
                    "topk" => m.topk = v.parse().unwrap(), "cpuct" => m.cpuct = v.parse().unwrap(), "depth" => m.max_depth = v.parse().unwrap(),
                    "v0" => m.value_zero = true, "div" => m.type_div = v != "0", "cleaf" => m.chance_leaf = v != "0", "nodet" => m.determinize = false, "roll" => m.roll_turns = v.parse().unwrap(), "rw" => m.roll_w = v.parse().unwrap(), _ => panic!("unknown mcts option {}", k),
                }
            }
            Agent::Search(m)
        }
        _ => panic!("unknown agent {}", spec),
    }
}

/// `bench <games> <agentA> <agentB>`: A vs B with alternating seats; prints A's score with a 95% interval
fn bench(games: usize, sa: &str, sb: &str) {
    let mut nets = std::collections::HashMap::new();
    for spec in [sa, sb] { let p: Vec<&str> = spec.split(':').collect(); if p.len() > 1 && !nets.contains_key(p[1]) { nets.insert(p[1].to_string(), load_net(p[1]).0); } }
    let t0 = Instant::now();
    let base = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64;
    let res: Vec<(i32, i32, i32)> = (0..games).into_par_iter().map(|g| {
        let mut rng = Rng::new(base ^ (g as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15));
        let seat = g % 2;
        let (a, b) = (parse_agent(sa, &nets), parse_agent(sb, &nets));
        let ag = if seat == 0 { [a, b] } else { [b, a] };
        let (w, turns, s0, s1) = play_game_scores(&ag, &mut rng);
        if std::env::var("BENCH_SCORES").is_ok() { println!("game {} seat {} scores {}-{} turns {}", g, seat, s0, s1, turns); }
        let margin = if seat == 0 { s0 - s1 } else { s1 - s0 };
        (if w < 0 { 0 } else if w as usize == seat { 1 } else { -1 }, turns, margin)
    }).collect();
    let mm = res.iter().map(|r| r.2 as f64).sum::<f64>() / games as f64;
    let msd = (res.iter().map(|r| (r.2 as f64 - mm).powi(2)).sum::<f64>() / games as f64).sqrt();
    let (w, d, l) = (res.iter().filter(|r| r.0 == 1).count(), res.iter().filter(|r| r.0 == 0).count(), res.iter().filter(|r| r.0 == -1).count());
    let rate = (w as f64 + 0.5 * d as f64) / games as f64;
    let se = (rate * (1.0 - rate) / games as f64).sqrt();
    println!("{:<44} vs {:<34} {:>5.1}% ±{:.1}  ({}-{}-{})  margin {:+.2} ±{:.2}  turns {:.0}  {:.0}s", sa, sb, rate * 100.0, 196.0 * se, w, d, l, mm, 1.96 * msd / (games as f64).sqrt(),
        res.iter().map(|r| r.1 as f64).sum::<f64>() / games as f64, t0.elapsed().as_secs_f64());
}

const ANAMES: [&str; 24] = ["inc_tiles", "inc_card", "inc_hill", "swap", "place", "move", "load", "unload", "fight", "transport", "tmove", "tstay",
    "bridge", "blockage", "wings", "super", "overlay", "sdone", "sundo", "hide", "guess", "revenge", "decline", "end"];

/// `trace <agentA> <games>`: A (seat 0/1 alternating) vs heuristic; action-type histogram of A, decisions per turn, end scores
fn trace(sa: &str, games: usize) {
    let mut nets = std::collections::HashMap::new();
    let p: Vec<&str> = sa.split(':').collect(); if p.len() > 1 { nets.insert(p[1].to_string(), load_net(p[1]).0); }
    let mut hist = [0usize; 24]; let mut turns_a = 0usize; let mut dec_a = 0usize; let (mut sc_a, mut sc_h) = (0i32, 0i32);
    let mut rng = Rng::new(42);
    for g in 0..games {
        let seat = g % 2;
        let a = parse_agent(sa, &nets);
        let ag = if seat == 0 { [a, Agent::Heur(4.0)] } else { [Agent::Heur(4.0), a] };
        let mut s = engine::new_game(2, (rng.u64() >> 33) as i32);
        let mut guard = 0;
        while !s.over && guard < 6000 {
            guard += 1;
            let pi = s.actor();
            let la = s.legal_actions();
            let act = ag[pi].act(&s, &mut rng);
            if pi == seat && la.len() > 1 { hist[act.t as usize] += 1; dec_a += 1; if act.t == engine::A_END { turns_a += 1; } }
            s.apply(act);
        }
        sc_a += s.score(seat); sc_h += s.score(1 - seat);
    }
    let tot: usize = hist.iter().sum();
    let mut v: Vec<(usize, usize)> = hist.iter().copied().enumerate().filter(|x| x.1 > 0).collect(); v.sort_by(|a, b| b.1.cmp(&a.1));
    println!("{}: decisions/turn {:.1}  avg end score A {:.2} vs heur {:.2}", sa, dec_a as f64 / turns_a.max(1) as f64, sc_a as f64 / games as f64, sc_h as f64 / games as f64);
    println!("  {}", v.iter().map(|(i, c)| format!("{} {:.1}%", ANAMES[*i], 100.0 * *c as f64 / tot as f64)).collect::<Vec<_>>().join("  "));
}

/// `diag <model> <sims> <games> [qbeta]`: self-play like training; reports value-head calibration and how the Q-targets shift the policy
fn diag(path: &str, sims: usize, games: usize, qbeta: f64) {
    let (net, _) = load_net(path);
    #[derive(Default, Clone)] struct D { n: f64, v: f64, z: f64, vz: f64, v2: f64, z2: f64, se: f64, se0: f64,
        pe: f64, te: f64, qe: f64, ne: f64, qo: f64, no: f64, kl: f64, start: f64, sv: f64, mv: f64, sn: f64, mn: f64, sr: f64, tdn: f64, tdnn: f64, tde: f64, tden: f64, shift: [f64; 24], pmass: [f64; 24], qsum: [f64; 24], qn: [f64; 24], dty: [f64; 24], dtn: [f64; 24] }
    let rs: Vec<D> = (0..games).into_par_iter().map(|g| {
        let mut rng = Rng::new(1234 + g as u64 * 7919);
        let mut s = engine::new_game(2, (rng.u64() >> 33) as i32);
        let mut m = mcts::Mcts::new(&net, sims, 18, 0.25);
        if std::env::var("DIAG_CLEAF").is_ok() { m.chance_leaf = true; }
        let mut rec: Vec<(usize, f64, bool, f64, f64, f64, Option<f64>, Option<f64>, f64)> = Vec::new();
        let mut prev: Option<(usize, f64, bool, usize)> = None; // (actor, v, was 'end', action type)
        let (mut dty, mut dtn) = ([0f64; 24], [0f64; 24]);
        let mut td = [0f64; 4]; // sum dv after non-end, n, sum dv after end, n
        let (mut tshift, mut tpm, mut tq, mut tqn) = ([0f64; 24], [0f64; 24], [0f64; 24], [0f64; 24]);
        let mut guard = 0; let mut searched = 0;
        while !s.over && guard < 3000 {
            guard += 1;
            let pi = s.actor();
            let la = s.legal_actions();
            if la.len() == 1 { s.apply(la[0]); continue; }
            let ctx = build_ctx(&s, pi);
            let v = net.value(&make_obs(&s, &ctx, &[], net.gver())) as f64;
            if let Some((pa, pv, was_end, pt)) = prev { let vv = if pa == pi { v } else { -v }; dty[pt] += vv - pv; dtn[pt] += 1.0; if was_end { td[2] += vv - pv; td[3] += 1.0; } else { td[0] += vv - pv; td[1] += 1.0; } }
            let r = m.search(&s, sims, true, &mut rng);
            let mut l: Vec<f64> = r.prior.iter().zip(r.q.iter()).map(|(&p, &q)| (p.max(1e-8) as f64).ln() + qbeta * (q as f64 - r.value)).collect();
            let mx = l.iter().cloned().fold(f64::NEG_INFINITY, f64::max); let mut z = 0.0; for x in l.iter_mut() { *x = (*x - mx).exp(); z += *x; }
            for x in l.iter_mut() { *x /= z; }
            let kl: f64 = l.iter().zip(r.prior.iter()).map(|(t, &p)| if *t > 1e-9 { t * (t / (p as f64).max(1e-9)).ln() } else { 0.0 }).sum();
            for (i, a) in r.acts.iter().enumerate() {
                let t = a.t as usize; tshift[t] += l[i] - r.prior[i] as f64; tpm[t] += r.prior[i] as f64;
                if r.visits[i] > 0 { tq[t] += r.q[i] as f64 - r.value; tqn[t] += 1.0; }
            }
            let ei = r.acts.iter().position(|a| a.t == engine::A_END);
            let (pe, te, qe) = match ei { Some(i) => (r.prior[i] as f64, l[i], if r.visits[i] > 0 { Some(r.q[i] as f64 - r.value) } else { None }), None => (0.0, 0.0, None) };
            let bo = (0..r.acts.len()).filter(|&i| Some(i) != ei && r.visits[i] > 0).map(|i| r.q[i] as f64 - r.value).fold(None, |m: Option<f64>, x| Some(m.map_or(x, |y| y.max(x))));
            rec.push((pi, v, s.actions_this_turn == 0 && s.pending.is_none(), pe, te, kl, qe, bo, r.value));
            let tau = if searched < 30 { 1.0 } else { 0.1 }; searched += 1;
            let idx = mcts::pick_by_visits(&r.visits, tau, &mut rng);
            prev = Some((pi, v, r.acts[idx].t == engine::A_END, r.acts[idx].t as usize));
            s.apply(r.acts[idx]);
        }
        let mut d = D::default();
        d.tdn = td[0]; d.tdnn = td[1]; d.tde = td[2]; d.tden = td[3];
        d.shift = tshift; d.pmass = tpm; d.qsum = tq; d.qn = tqn; d.dty = dty; d.dtn = dtn;
        for (pi, v, st, pe, te, kl, qe, bo, rv) in rec {
            let z = if !s.over || s.winner < 0 { 0.0 } else if s.winner as usize == pi { 1.0 } else { -1.0 };
            d.n += 1.0; d.v += v; d.z += z; d.vz += v * z; d.v2 += v * v; d.z2 += z * z; d.se += (v - z) * (v - z); d.se0 += z * z; d.sr += (rv - z) * (rv - z);
            d.pe += pe; d.te += te; d.kl += kl;
            if let Some(q) = qe { d.qe += q; d.ne += 1.0; }
            if let Some(q) = bo { d.qo += q; d.no += 1.0; }
            if st { d.start += 1.0; d.sv += v; d.sn += 1.0; } else { d.mv += v; d.mn += 1.0; }
        }
        d
    }).collect();
    let mut t = D::default();
    for d in rs { t.n += d.n; t.v += d.v; t.z += d.z; t.vz += d.vz; t.v2 += d.v2; t.z2 += d.z2; t.se += d.se; t.se0 += d.se0; t.sr += d.sr; t.pe += d.pe; t.te += d.te; t.kl += d.kl;
        t.qe += d.qe; t.ne += d.ne; t.qo += d.qo; t.no += d.no; t.sv += d.sv; t.sn += d.sn; t.mv += d.mv; t.mn += d.mn; t.tdn += d.tdn; t.tdnn += d.tdnn; t.tde += d.tde; t.tden += d.tden;
        for k in 0..24 { t.dty[k] += d.dty[k]; t.dtn[k] += d.dtn[k]; t.shift[k] += d.shift[k]; t.pmass[k] += d.pmass[k]; t.qsum[k] += d.qsum[k]; t.qn[k] += d.qn[k]; } }
    let n = t.n;
    let corr = (t.vz / n - t.v / n * t.z / n) / ((t.v2 / n - (t.v / n).powi(2)).sqrt() * (t.z2 / n - (t.z / n).powi(2)).sqrt());
    println!("{} sims {} qbeta {}: {} decisions", path, sims, qbeta, n);
    println!("  value: mean v {:+.3} (turn start {:+.3}, mid-turn {:+.3})  mean z {:+.3}  MSE net {:.3}  MSE search-root {:.3}  MSE(0) {:.3}  corr {:.3}",
        t.v / n, t.sv / t.sn.max(1.0), t.mv / t.mn.max(1.0), t.z / n, t.se / n, t.sr / n, t.se0 / n, corr);
    println!("  'end': prior {:.3} -> target {:.3}   Q(end)-V {:+.3} (n {})   best other Q-V {:+.3} (n {})   KL(target||prior) {:.3}",
        t.pe / n, t.te / n, t.qe / t.ne.max(1.0), t.ne, t.qo / t.no.max(1.0), t.no, t.kl / n);
    let mut ty: Vec<usize> = (0..24).filter(|&k| t.pmass[k] / n > 0.005).collect();
    ty.sort_by(|a, b| t.pmass[*b].partial_cmp(&t.pmass[*a]).unwrap());
    println!("  per type: prior mass -> target shift (relative) | mean Q-V of visited | value drift after playing it (should be ~0)");
    for k in ty { println!("    {:<10} {:.3} {:+.4} ({:+.1}%) | {:+.3} (n {}) | {:+.3} (n {})", ANAMES[k], t.pmass[k] / n, t.shift[k] / n, 100.0 * t.shift[k] / t.pmass[k], t.qsum[k] / t.qn[k].max(1.0), t.qn[k], t.dty[k] / t.dtn[k].max(1.0), t.dtn[k]); }
    println!("  value drift per step (should be ~0): after a non-end action {:+.4} (n {})   after 'end' {:+.4} (n {})", t.tdn / t.tdnn.max(1.0), t.tdnn, t.tde / t.tden.max(1.0), t.tden);
}

/// `gradcheck`: compares the hand-written backward pass with finite differences on a random graph net
fn gradcheck() {
    let mut rng = Rng::new(7);
    let arch = Arch { h1: 12, h2: 8, hv: 10, hg: 6, gl: 2, fv: if std::env::var("FV2").is_ok() { 2 } else { 1 } };
    let mut net = Net::random(arch, &mut rng);
    for x in net.theta.iter_mut() { *x += (rng.f64() as f32 - 0.5) * 0.1; } // make zero-initialised parts non-trivial
    let mut obs: Vec<(Obs, Vec<f32>, f32, f32)> = Vec::new();
    for g in 0..4 {
        let mut s = engine::new_game(2, 100 + g);
        for step in 0..(30 + g * 25) {
            let acts = s.legal_actions();
            if s.over { break; }
            let a = if acts.len() > 1 && step % 3 != 0 { heuristic_act(&s, 4.0, &mut rng) } else { acts[0] };
            s.apply(a);
        }
        let pi = s.actor(); let acts = s.legal_actions();
        if acts.len() < 2 { continue; }
        let ctx = build_ctx(&s, pi);
        let o = make_obs(&s, &ctx, &acts, arch.fv);
        let mut t: Vec<f32> = (0..acts.len()).map(|_| rng.f64() as f32).collect(); let z: f32 = t.iter().sum(); for x in t.iter_mut() { *x /= z; }
        println!("sample {}: {} nodes, {} candidates", g, o.g.as_ref().unwrap().n, acts.len());
        obs.push((o, t, 0.3 - 0.2 * g as f32, 0.1 * g as f32));
    }
    let loss = |net: &Net| -> f64 {
        let mut l = 0f64;
        for (o, t, z, z2) in &obs {
            let e = net.encode(o, false); let fp = net.forward_policy(&e, &o.a, o.n, false); let p = nn::softmax(&fp.logits, 1.0);
            for c in 0..o.n { l -= t[c] as f64 * (p[c].max(1e-12) as f64).ln(); }
            let fv = net.forward_value(&e, false);
            l += ((fv.v - z) as f64).powi(2) + 0.5 * ((fv.aux - z2) as f64).powi(2);
        }
        l
    };
    let mut grad = vec![0f32; net.size()];
    for (o, t, z, z2) in &obs {
        let e = net.encode(o, true); let fp = net.forward_policy(&e, &o.a, o.n, true); let p = nn::softmax(&fp.logits, 1.0);
        let dlog: Vec<f32> = (0..o.n).map(|c| p[c] - t[c]).collect();
        let fv = net.forward_value(&e, true);
        net.backward(&mut grad, o, &e, &fp, &dlog, &fv, 2.0 * (fv.v - z), 0.5 * 2.0 * (fv.aux - z2));
    }
    let (mut maxrel, mut checked) = (0f64, 0);
    let eps = 3e-3f32;
    let gmax = grad.iter().fold(0f64, |m, &x| m.max(x.abs() as f64));
    for i in (0..net.size()).step_by(7) {
        let mut a = net.clone(); a.theta[i] += eps; let mut b = net.clone(); b.theta[i] -= eps;
        let num = (loss(&a) - loss(&b)) / (2.0 * eps as f64);
        let an = grad[i] as f64;
        let rel = (num - an).abs() / gmax;
        if rel > maxrel { maxrel = rel; }
        if rel > 0.02 { println!("  param {} analytic {:.5} numeric {:.5}", i, an, num); }
        checked += 1;
    }
    println!("gradcheck: {} parameters checked, max |analytic-numeric| / max|grad| = {:.4} (max |grad| {:.3})", checked, maxrel, gmax);
}

/// `abtest <teacher.json> <games> <sims> <rounds> [lr] [hg,gl]`: does the board graph carry information the plain MLP lacks?
/// Generates teacher self-play data once, then trains (a) the teacher's MLP and (b) the same MLP grown by a graph network on it
/// (function-preserving start), and prints held-out policy cross-entropy / top-1 agreement / value MSE after every round.
fn abtest(teacher: &str, games: usize, sims: usize, rounds: usize, lr: f64, hg: usize, gl: usize) {
    let (tnet, _) = load_net(teacher);
    let cfg = SpCfg { sims, tau_moves: 30, dirichlet: 0.25, depth: 18, qbeta: argf("qbeta", 0.0), ttemp: 1.0, vbalance: false, vtd: 1.0, draw_z: 0.0, graph: true };
    let t0 = Instant::now();
    let outs: Vec<GameOut> = (0..games).into_par_iter().map(|g| {
        let mut r = Rng::new(0xABCD ^ (g as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15));
        self_play_game(&tnet, cfg, g % 10 < 3, &mut r)
    }).collect();
    let split = games * 9 / 10;
    let vraw = std::env::var("VRAW").is_ok();
    let pcoef_ab = std::env::var("PCOEF").ok().and_then(|x| x.parse().ok()).unwrap_or(1.0f32);
    let (mut train, mut hold): (Vec<Sample>, Vec<Sample>) = (Vec::new(), Vec::new());
    for (i, mut o) in outs.into_iter().enumerate() { if vraw { for x in o.samples.iter_mut() { x.z = x.zr; } } if i < split { train.extend(o.samples) } else { hold.extend(o.samples) } }
    println!("data: {} train / {} held-out samples in {:.0}s", train.len(), hold.len(), t0.elapsed().as_secs_f64());
    let mut rng = Rng::new(5);
    let mut nets: Vec<(&str, Net)> = if std::env::var("SCRATCH").is_ok() {
        vec![("mlp", Net::random(DEFAULT_ARCH, &mut rng)), ("graph", Net::random(Arch { hg, gl, fv: 1, ..DEFAULT_ARCH }, &mut rng))]
    } else { vec![("mlp", tnet.clone()), ("graph", Net::grow_from(&tnet, hg, gl, &mut rng))] };
    println!("params: mlp {}  graph {}", nets[0].1.size(), nets[1].1.size());
    let holdout = |net: &Net| -> (f64, f64, f64, f64) {
        let r: Vec<(f64, f64, f64, f64)> = hold.par_iter().map(|d| {
            let e = net.encode(&d.obs, false);
            let fp = net.forward_policy(&e, &d.obs.a, d.obs.n, false);
            let p = nn::softmax(&fp.logits, 1.0);
            let mut ce = 0f64; for c in 0..d.obs.n { if d.tgt[c] > 1e-8 { ce -= d.tgt[c] as f64 * (p[c].max(1e-8) as f64).ln(); } }
            let am = |v: &[f32]| (0..v.len()).max_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap()).unwrap();
            let top1 = if am(&p) == am(&d.tgt) { 1.0 } else { 0.0 };
            let fv = net.forward_value(&e, false);
            (ce, top1, ((fv.v - d.z) as f64).powi(2), 1.0)
        }).collect();
        let n = r.len() as f64;
        (r.iter().map(|x| x.0).sum::<f64>() / n, r.iter().map(|x| x.1).sum::<f64>() / n, r.iter().map(|x| x.2).sum::<f64>() / n, n)
    };
    for (name, net) in nets.iter() { let h = holdout(net); println!("{:<6} round 0: policy CE {:.4}  top1 {:.3}  value MSE {:.4}", name, h.0, h.1, h.2); }
    let steps = argf("steps", 400.0) as usize;
    for round in 1..=rounds {
        for (name, net) in nets.iter_mut() {
            let t = Instant::now();
            let (pl, vl) = az_update(net, &train, steps, 128, lr, pcoef_ab, 1.0, 0.5, &mut rng);
            let h = holdout(net);
            println!("{:<6} round {}: train ploss {:.4} vloss {:.4} | held-out policy CE {:.4}  top1 {:.3}  value MSE {:.4}   ({:.0}s)", name, round, pl, vl, h.0, h.1, h.2, t.elapsed().as_secs_f64());
        }
    }
    let _ = nets.iter().map(|(n, net)| save(&format!("/tmp/abtest-{}.json", n), net, sims, 0, 0, &[], &json!({"name": n}))).count();
}

/// `vtest <teacher.json> <games> <K> <rounds> [lr] [hg,gl]`: value learning from low-noise labels.
/// Label of a turn-start position = mean outcome of K noisy heuristic-vs-heuristic playouts (hidden information re-dealt every time),
/// so a label is ~sqrt(K) less noisy than a single game result. Trains a fresh MLP and a fresh graph net on them and reports
/// held-out MSE against the labels (plus the teacher's value head, which was trained on game outcomes, as a reference).
fn vtest(teacher: &str, games: usize, k: usize, rounds: usize, lr: f64, hg: usize, gl: usize) {
    let (tnet, _) = load_net(teacher);
    let t0 = Instant::now();
    let outs: Vec<Vec<Sample>> = (0..games).into_par_iter().map(|g| {
        let mut rng = Rng::new(0x5151 ^ (g as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15));
        let mut s = engine::new_game(2, (rng.u64() >> 33) as i32);
        let mut out: Vec<Sample> = Vec::new();
        let mut guard = 0;
        let search = mcts::Mcts::new(&tnet, 32, 18, 0.25);
        let heur_seat = if g % 3 == 0 { rng.below(2) as i32 } else { -1 };
        while !s.over && guard < 3000 {
            guard += 1;
            let pi = s.actor();
            let acts = s.legal_actions();
            if acts.len() == 1 { s.apply(acts[0]); continue; }
            if s.actions_this_turn == 0 && s.pending.is_none() && rng.f64() < 0.35 {
                let mut tot = 0f64;
                for _ in 0..k {
                    let mut r = s;
                    mcts::determinize(&mut r, pi, &mut rng);
                    let mut gg = 0;
                    while !r.over && gg < 3000 { gg += 1; let a = heuristic_act(&r, 4.0, &mut rng); r.apply(a); }
                    tot += if !r.over || r.winner < 0 { 0.0 } else if r.winner as usize == pi { 1.0 } else { -1.0 };
                }
                let ctx = build_ctx(&s, pi);
                let obs = make_obs(&s, &ctx, &[], 1);
                out.push(Sample { obs, tgt: Vec::new(), z: (tot / k as f64) as f32, z2: 0.0, zr: 0.0, vw: 1.0 });
            }
            let a = if pi as i32 == heur_seat { heuristic_act(&s, 4.0, &mut rng) } else {
                let r = search.search(&s, 32, true, &mut rng);
                r.acts[mcts::pick_by_visits(&r.visits, if s.turn_no < 20 { 1.0 } else { 0.1 }, &mut rng)]
            };
            s.apply(a);
        }
        out
    }).collect();
    let split = games * 9 / 10;
    let (mut train, mut hold): (Vec<Sample>, Vec<Sample>) = (Vec::new(), Vec::new());
    for (i, o) in outs.into_iter().enumerate() { if i < split { train.extend(o) } else { hold.extend(o) } }
    println!("labels: {} train / {} held-out in {:.0}s", train.len(), hold.len(), t0.elapsed().as_secs_f64());
    let var0 = hold.iter().map(|d| (d.z as f64).powi(2)).sum::<f64>() / hold.len() as f64;
    let mse = |net: &Net| -> f64 { hold.par_iter().map(|d| ((net.value(&d.obs) - d.z) as f64).powi(2)).sum::<f64>() / hold.len() as f64 };
    println!("label variance (MSE of predicting 0): {:.4}   teacher value head: MSE {:.4}", var0, mse(&tnet));
    let mut rng = Rng::new(9);
    let mut nets: Vec<(&str, Net)> = vec![("mlp", Net::random(DEFAULT_ARCH, &mut rng)), ("graph", Net::random(Arch { hg, gl, fv: 1, ..DEFAULT_ARCH }, &mut rng))];
    let steps = argf("steps", 400.0) as usize;
    for round in 1..=rounds {
        for (name, net) in nets.iter_mut() {
            let (_, vl) = az_update(net, &train, steps, 128, lr, 0.0, 1.0, 0.0, &mut rng);
            if round % 5 == 0 || round == rounds { println!("{:<6} round {:>3}: train MSE {:.4}  held-out MSE {:.4}  (R2 {:.3})", name, round, vl, mse(net), 1.0 - mse(net) / var0); }
        }
    }
    for (n, net) in nets.iter() { save(&format!("/tmp/vtest-{}.json", n), net, 0, 0, 0, &[], &json!({"name": n})); }
}

/// `stall <games> <agentA> <agentB>` (env AMINO_MAX_ROUNDS): how games end - by 7 points, all 12 resources delivered, or the
/// round cap - and, for capped games, where the resources are and when the score last changed.
fn stall(games: usize, sa: &str, sb: &str) {
    let mut nets = std::collections::HashMap::new();
    for spec in [sa, sb] { let p: Vec<&str> = spec.split(':').collect(); if p.len() > 1 && !nets.contains_key(p[1]) { nets.insert(p[1].to_string(), load_net(p[1]).0); } }
    let cap = engine::MAX_ROUNDS_RT.load(std::sync::atomic::Ordering::Relaxed);
    let res: Vec<String> = (0..games).into_par_iter().map(|g| {
        let mut rng = Rng::new(777 + g as u64 * 7919);
        let seat = g % 2;
        let (a, b) = (parse_agent(sa, &nets), parse_agent(sb, &nets));
        let ag = if seat == 0 { [a, b] } else { [b, a] };
        let mut s = engine::new_game(2, (rng.u64() >> 33) as i32);
        let (mut last_sc, mut last_round) = ((0, 0), 0);
        let mut guard = 0;
        while !s.over && guard < 200000 {
            guard += 1;
            let pi = s.actor();
            let act = ag[pi].act(&s, &mut rng);
            s.apply(act);
            let sc = (s.score(0), s.score(1));
            if sc != last_sc { last_sc = sc; last_round = s.turn_no / 2; }
        }
        let owned = s.resources.iter().filter(|r| r.owner >= 0).count();
        let board = s.resources.iter().filter(|r| r.owner < 0 && r.carrier < 0 && r.cell >= 0).count();
        let carried = s.resources.iter().filter(|r| r.carrier >= 0).count();
        let why = if !s.over { "guard" } else if s.score(0) >= 7 || s.score(1) >= 7 { "7pts" } else if owned >= 12 { "all12" } else { "cap" };
        format!("{} {} {}-{} rounds {} lastScore {} owned {} board {} carried {} resStack {} tiles {} hands {}/{} winner {}",
            why, g, s.score(seat), s.score(1 - seat), s.turn_no / 2, last_round, owned, board, carried, s.res_stack.len(), s.tile_stack.len(),
            s.players[0].hand.len(), s.players[1].hand.len(), s.winner)
    }).collect();
    let n = res.len() as f64;
    for k in ["7pts", "all12", "cap", "guard"] { println!("{:<6} {:>5.1}%", k, 100.0 * res.iter().filter(|r| r.starts_with(k)).count() as f64 / n); }
    let capped: Vec<&String> = res.iter().filter(|r| r.starts_with("cap")).collect();
    println!("cap {} - capped games (first 12):", cap);
    for r in capped.iter().take(12) { println!("  {}", r); }
    if std::env::var("STALL_ALL").is_ok() { for r in &res { println!("{}", r); } }
}

/// `probe <model> <sims> <games> <round>`: self-play (MCTS) up to `round`, then at the next turn start of a game that is still 0-0
/// with nothing on the board, print every candidate: prior, visits, Q, and the value net right after it (mover's view).
fn probe(path: &str, sims: usize, games: usize, round: i32) {
    let (net, _) = load_net(path);
    let m = mcts::Mcts::new(&net, sims, 18, 0.0);
    let mut shown = 0;
    for g in 0..games {
        let mut rng = Rng::new(4242 + g as u64);
        let mut s = engine::new_game(2, (rng.u64() >> 33) as i32);
        let mut guard = 0;
        while !s.over && guard < 20000 && !(s.turn_no / 2 >= round && s.actions_this_turn == 0 && s.pending.is_none()) {
            guard += 1;
            let la = s.legal_actions();
            let a = if la.len() == 1 { la[0] } else { let r = m.search(&s, sims, false, &mut rng); r.acts[mcts::pick_by_visits(&r.visits, 0.0, &mut rng)] };
            s.apply(a);
        }
        if s.over || s.resources.len() > 0 { continue; }
        let pi = s.actor();
        let p = &s.players[pi];
        let (v0, _) = m.evaluate(&s, &[], false);
        println!("game {} round {} player {}: hand {} cards {} hills_in_hand {} ant {} | tiles left {} | v(now) {:+.3}", g, s.turn_no / 2, pi, p.hand.len(), p.cards.len(), p.hills_in_hand, p.ant.is_some(), s.tile_stack.len(), v0);
        let r = m.search(&s, sims, false, &mut rng);
        for (i, a) in r.acts.iter().enumerate() {
            // value right after the action (and any forced follow-ups), from pi's view, averaged over 16 determinizations
            let mut acc = 0.0;
            for _ in 0..16 {
                let mut t = s; mcts::determinize(&mut t, pi, &mut rng);
                t.apply(*a);
                let mut k = 0; while !t.over && k < 50 { let la = t.legal_actions(); if la.len() != 1 { break; } t.apply(la[0]); k += 1; }
                let (v, _) = m.evaluate(&t, &[], false);
                acc += if t.over { if t.winner < 0 { 0.0 } else if t.winner as usize == pi { 1.0 } else { -1.0 } } else if t.actor() == pi { v as f64 } else { -(v as f64) };
            }
            // ground truth under the current net: play out to the end (both sides MCTS) after this action
            let mut np = std::env::var("PROBE_PLAYOUTS").ok().and_then(|x| x.parse().ok()).unwrap_or(0usize);
            if std::env::var("PROBE_REP").is_ok() && r.acts[..i].iter().any(|b| b.t == a.t && b.c == a.c) { np = 0; }
            let opp_heur = std::env::var("PROBE_OPP").map_or(false, |v| v == "heur");
            let outs: Vec<(f64, f64)> = (0..np).into_par_iter().map(|k| {
                let mut rr = Rng::new(99 + k as u64 * 31);
                let mm = mcts::Mcts::new(&net, sims, 18, 0.0);
                let mut t = s; mcts::determinize(&mut t, pi, &mut rr);
                t.apply(*a);
                let mut gd = 0;
                while !t.over && gd < 20000 { gd += 1; let la = t.legal_actions(); let b = if la.len() == 1 { la[0] } else if opp_heur && t.actor() != pi { heuristic_act(&t, 4.0, &mut rr) } else { let r = mm.search(&t, sims, false, &mut rr); r.acts[mcts::pick_by_visits(&r.visits, 0.0, &mut rr)] }; t.apply(b); }
                (if t.winner < 0 { 0.0 } else if t.winner as usize == pi { 1.0 } else { -1.0 }, (t.turn_no / 2) as f64)
            }).collect();
            let gt = if np > 0 { format!("   playouts: outcome {:+.2}, rounds {:.0}", outs.iter().map(|o| o.0).sum::<f64>() / np as f64, outs.iter().map(|o| o.1).sum::<f64>() / np as f64) } else { String::new() };
            if std::env::var("PROBE_REP").is_ok() && np == 0 { continue; }
            println!("   {:<10}{} prior {:.3} visits {:>4} Q {:+.3}   v(after) {:+.3}{}", ANAMES[a.t as usize], if a.t == engine::A_PLACE { ["", " tile", " +hill", " +res"][a.c.min(2) as usize + 1] } else { "" }, r.prior[i], r.visits[i], r.q[i], acc / 16.0, gt);
        }
        if std::env::var("PROBE_AFTER_TILES").is_ok() {
            // one level deeper: after drawing tiles, what does the net want? (prior mass per action type, top candidates)
            let mut t = s;
            if let Some(a) = t.legal_actions().iter().find(|a| a.t == engine::A_INC_TILES).copied() { t.apply(a); }
            let r = m.search(&t, sims, false, &mut rng);
            let mut mass = [0f64; 24]; for (i, a) in r.acts.iter().enumerate() { mass[a.t as usize] += r.prior[i] as f64; }
            let mut vm = [0f64; 24]; for (i, a) in r.acts.iter().enumerate() { vm[a.t as usize] += r.visits[i] as f64; }
            let hills = r.acts.iter().filter(|a| a.t == engine::A_PLACE && a.c == 1).count();
            println!("   after inc_tiles (hand {}): {} candidates, {} hill placements", t.players[pi].hand.len(), r.acts.len(), hills);
            for k in 0..24 { if mass[k] > 0.0 || vm[k] > 0.0 { println!("     {:<10} prior mass {:.3}  visits {}", ANAMES[k], mass[k], vm[k]); } }
            let mut idx: Vec<usize> = (0..r.acts.len()).collect(); idx.sort_by(|&x, &y| r.visits[y].cmp(&r.visits[x]));
            for &i in idx.iter().take(5) { let a = r.acts[i]; println!("     top: {:<8} piece {} prior {:.3} visits {} Q {:+.3}", ANAMES[a.t as usize], a.c, r.prior[i], r.visits[i], r.q[i]); }
            let hp: f64 = r.acts.iter().enumerate().filter(|(_, a)| a.t == engine::A_PLACE && a.c == 1).map(|(i, _)| r.prior[i] as f64).sum();
            let hq: Vec<String> = r.acts.iter().enumerate().filter(|(_, a)| a.t == engine::A_PLACE && a.c == 1).take(4).map(|(i, _)| format!("{:+.3}/{}", r.q[i], r.visits[i])).collect();
            println!("     hill placements: prior mass {:.3}, Q/visits of first: {}", hp, hq.join(" "));
        }
        shown += 1; if shown >= 2 { break; }
    }
}

/// `hills <games> <agentA> <agentB>`: hill placement and getting cut off. Counts hills A or B places while its ant is out,
/// how many the ant cannot reach afterwards (paths, bridges, tunnels), hills laid while carrying (and of these within 5 steps
/// without tunnels); episodes in which A's carrying ant has no transport path to an own hill (cut off) and how they end
/// (reconnected while still carrying / resource dropped or lost / still cut off at the game end), their mean length in rounds;
/// and how often each side cuts off an enemy carrier (overbuilding, blockage, ...).
fn hills(games: usize, sa: &str, sb: &str) {
    let mut nets = std::collections::HashMap::new();
    for spec in [sa, sb] { let p: Vec<&str> = spec.split(':').collect(); if p.len() > 1 && !nets.contains_key(p[1]) { nets.insert(p[1].to_string(), load_net(p[1]).0); } }
    let res: Vec<[u32; 11]> = (0..games).into_par_iter().map(|g| {
        let mut rng = Rng::new(4242 + g as u64 * 7919);
        let seat = g % 2;
        let (a, b) = (parse_agent(sa, &nets), parse_agent(sb, &nets));
        let ag = if seat == 0 { [a, b] } else { [b, a] };
        let mut s = engine::new_game(2, (rng.u64() >> 33) as i32);
        // [hills with ant out, unreachable, laid while carrying, carrying and within 5, episodes, reconnected, dropped, stuck at end, rounds, cuts by A, cuts by B]
        let mut c = [0u32; 11];
        let (mut in_ep, mut t0) = (false, 0);
        let mut guard = 0;
        while !s.over && guard < 200000 {
            guard += 1;
            let pi = s.actor();
            let act = ag[pi].act(&s, &mut rng);
            let had_ant = s.players[pi].ant.is_some();
            let carrying = s.players[pi].carry >= 0;
            let home0: Vec<i32> = (0..s.n).map(|q| if q != pi && s.players[q].carry >= 0 { s.home_dist(q) } else { 99 }).collect();
            s.apply(act);
            if (0..s.n).any(|q| home0[q] < 99 && s.players[q].carry >= 0 && s.home_dist(q) >= 99) { c[if pi == seat { 9 } else { 10 }] += 1; }
            if act.t == engine::A_PLACE && act.c == 1 && had_ant {
                c[0] += 1;
                let slot = s.slot_at(engine::gx(act.g as i32), engine::gy(act.g as i32)).unwrap();
                let st = s.ant_starts(pi);
                if s.bfs(pi, st.as_slice(), 999, true).dist[slot * 3] < 0 { c[1] += 1; }
                if carrying { c[2] += 1; let d = s.bfs(pi, st.as_slice(), 5, false).dist[slot * 3]; if d >= 0 { c[3] += 1; } }
            }
            let cut = s.players[seat].carry >= 0 && s.home_dist(seat) >= 99;
            if cut && !in_ep { in_ep = true; t0 = s.turn_no; c[4] += 1; }
            else if in_ep && !cut { in_ep = false; c[if s.players[seat].carry >= 0 { 5 } else { 6 }] += 1; c[8] += ((s.turn_no - t0) / 2) as u32; }
        }
        if in_ep { c[7] += 1; c[8] += ((s.turn_no - t0) / 2) as u32; }
        c
    }).collect();
    let t: Vec<u32> = (0..11).map(|k| res.iter().map(|r| r[k]).sum()).collect();
    println!("{} vs {}: hills placed with ant out {}, unreachable {} ({:.1}%), laid while carrying {}, of these within transport range {}", sa, sb, t[0], t[1], 100.0 * t[1] as f64 / t[0].max(1) as f64, t[2], t[3]);
    println!("  A cut off with a resource: {} episodes -> reconnected {}, dropped/lost {}, stuck at game end {}; mean {:.1} rounds", t[4], t[5], t[6], t[7], t[8] as f64 / t[4].max(1) as f64);
    println!("  enemy carriers cut off by A {}, by B {}", t[9], t[10]);
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    if argv.get(1).map(|s| s.as_str()) == Some("abtest") {
        let n = |i: usize, d: f64| argv.get(i).map_or(d, |x| x.parse().unwrap());
        let (hg, gl) = argv.get(7).map_or((16, 3), |x| { let v: Vec<usize> = x.split(',').map(|y| y.parse().unwrap()).collect(); (v[0], v[1]) });
        abtest(&argv[2], n(3, 1000.0) as usize, n(4, 128.0) as usize, n(5, 10.0) as usize, n(6, 3e-4), hg, gl); return;
    }
    if argv.get(1).map(|s| s.as_str()) == Some("vtest") {
        let n = |i: usize, d: f64| argv.get(i).map_or(d, |x| x.parse().unwrap());
        let (hg, gl) = argv.get(7).map_or((16, 3), |x| { let v: Vec<usize> = x.split(',').map(|y| y.parse().unwrap()).collect(); (v[0], v[1]) });
        vtest(&argv[2], n(3, 600.0) as usize, n(4, 8.0) as usize, n(5, 30.0) as usize, n(6, 1e-3), hg, gl); return;
    }
    if argv.get(1).map(|s| s.as_str()) == Some("gradcheck") { gradcheck(); return; }
    if argv.get(1).map(|s| s.as_str()) == Some("xcheck") { xcheck(&argv[2], argv[3].parse().unwrap()); return; }
    if argv.get(1).map(|s| s.as_str()) == Some("diag") { diag(&argv[2], argv[3].parse().unwrap(), argv[4].parse().unwrap(), argv.get(5).map_or(3.0, |x| x.parse().unwrap())); return; }
    if argv.get(1).map(|s| s.as_str()) == Some("trace") { trace(&argv[2], argv[3].parse().unwrap()); return; }
    if std::env::var("AMINO_FORCE_INCOME").is_ok() { engine::FORCE_INCOME.store(true, std::sync::atomic::Ordering::Relaxed); }
    if let Ok(v) = std::env::var("AMINO_MAX_ROUNDS") { engine::MAX_ROUNDS_RT.store(v.parse().unwrap(), std::sync::atomic::Ordering::Relaxed); }
    if argv.get(1).map(|s| s.as_str()) == Some("probe") { probe(&argv[2], argv[3].parse().unwrap(), argv[4].parse().unwrap(), argv[5].parse().unwrap()); return; }
    if argv.get(1).map(|s| s.as_str()) == Some("stall") { stall(argv[2].parse().unwrap(), &argv[3], &argv[4]); return; }
    if argv.get(1).map(|s| s.as_str()) == Some("hills") { hills(argv[2].parse().unwrap(), &argv[3], &argv[4]); return; }
    if argv.get(1).map(|s| s.as_str()) == Some("bench") { bench(argv[2].parse().unwrap(), &argv[3], &argv[4]); return; }

    let ncpu = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let workers = argf("workers", ncpu as f64) as usize;
    let sims = argf("sims", 64.0) as usize;
    let games = argf("games", 24.0) as usize;
    let steps = argf("steps", 150.0) as usize; // upper bound of minibatches per iteration
    let pcoef = argf("pcoef", 1.0) as f32; let vcoef = argf("vcoef", 1.0) as f32; // loss weights (0 freezes that head)
    let auxcoef = argf("auxcoef", 0.5) as f32; // weight of the auxiliary score-margin head (graph nets only)
    let gate_every = argf("gate-every", 20.0) as u64; // champion gating: every N iterations play current vs champion (0 = off)
    let gate_games = argf("gate-games", 200.0) as usize;
    let gate_sims = argf("gate-sims", 64.0) as usize; // gating plays with search (so value-head gains count); 0 = raw policy
    let gate_win = argf("gate-win", 0.56); // promote the current net to champion at >= this score
    let gate_revert = argf("gate-revert", 0.42); // go back to the champion if the score is below this (stops drift); 0 = never
    let sp_champ = argf("sp-champion", 0.0) != 0.0;
    let ratio = argf("ratio", 4.0); // samples drawn per newly generated sample (limits over-fitting of the replay buffer)
    let min_buffer = argf("min-buffer", 5000.0) as usize; // no updates until the buffer holds this many samples
    mcts::TOPK.store(argf("topk", 12.0) as usize, std::sync::atomic::Ordering::Relaxed);
    mcts::ROLL_TURNS.store(argf("roll", 0.0) as usize, std::sync::atomic::Ordering::Relaxed);
    mcts::ROLL_W_MILLI.store((argf("roll-w", 0.5) * 1000.0) as usize, std::sync::atomic::Ordering::Relaxed);
    mcts::CHANCELEAF.store(argf("chance-leaf", 0.0) as usize, std::sync::atomic::Ordering::Relaxed);
    mcts::TYPEDIV.store(argf("type-div", 0.0) as usize, std::sync::atomic::Ordering::Relaxed);
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
    if let Some(spec) = arg("grow-graph") {
        let v: Vec<usize> = spec.split(',').map(|x| x.parse().unwrap()).collect();
        if net.arch.hg == 0 { net = Net::grow_from(&net, v[0], v[1], &mut rng); println!("grown to graph net hg={} gl={} ({} params)", v[0], v[1], net.size()); }
    }
    if arg("grow-features").is_some() && net.arch.hg > 0 && net.arch.fv < 3 { net = Net::grow_features(&net, &mut rng); println!("grown to graph feature version {} ({} params)", net.arch.fv, net.size()); }
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
        "hyper": {"sims": sims, "topk": mcts::TOPK.load(std::sync::atomic::Ordering::Relaxed), "cpuct": argf("cpuct", 1.25), "qbeta": argf("qbeta", 3.0), "vbalance": argf("vbalance", 0.0) != 0.0, "vtd": argf("vtd", 1.0), "typeDiv": argf("type-div", 0.0) != 0.0, "drawZ": argf("draw-z", 0.0), "chanceLeaf": argf("chance-leaf", 0.0) != 0.0, "gateSims": gate_sims, "games": games, "lr": lr, "ratio": ratio,
                  "minBuffer": min_buffer, "batch": batch, "oppMix": opp_mix, "depth": depth, "tauMoves": tau_moves},
        "host": {"os": std::env::consts::OS, "arch": std::env::consts::ARCH, "cpus": ncpu, "workers": workers},
        "git": git_commit(), "trainMinutes": base_minutes, "note": ""
    });
    let mut buffer: Vec<Sample> = Vec::new();
    let mut best = -1.0f64;
    let mut champion = net.clone(); let mut champ_iter = iter0;
    let cfg = SpCfg { sims, tau_moves, dirichlet: 0.25, depth, qbeta: argf("qbeta", 3.0), ttemp: argf("ttemp", 1.0), vbalance: argf("vbalance", 0.0) != 0.0, vtd: argf("vtd", 1.0), draw_z: argf("draw-z", 0.0) as f32, graph: false };
    for _ in 0..iters {
        if minutes > 0.0 && t0.elapsed().as_secs_f64() / 60.0 > minutes { break; }
        let t1 = Instant::now();
        let base = rng.u64();
        let outs: Vec<(GameOut, bool)> = {
            let netr = if sp_champ { &champion } else { &net }; // --sp-champion: data comes from the frozen champion (no feedback loop)
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
        let (ploss, vloss) = if steps_eff > 0 { az_update(&mut net, &buffer, steps_eff, batch, lr, pcoef, vcoef, auxcoef, &mut rng) } else { (0.0, 0.0) };
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
            let g = eval_h2h(&net, &champion, gate_games, gate_sims > 0, gate_sims, depth, rng.u64());
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
