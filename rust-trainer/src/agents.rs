// Port of js/agents.js (heuristic, NN, MCTS agents) + js/selfplay.js playGame
use crate::engine::*;
use crate::features::*;
use crate::mcts::{pick_by_visits, Mcts};
use crate::nn::{sample_idx, softmax, Net};
use crate::rng::Rng;

pub const MAX_CANDS: usize = 40;

pub fn cap_candidates(acts: Vec<Act>, max: usize, rng: &mut Rng) -> Vec<Act> {
    if acts.len() <= max { return acts; }
    let mut keep: Vec<Act> = acts.iter().copied().filter(|a| a.t != A_PLACE && a.t != A_OVERLAY).collect();
    let mut rest: Vec<Act> = acts.iter().copied().filter(|a| a.t == A_PLACE || a.t == A_OVERLAY).collect();
    let room = std::cmp::max(4, max.saturating_sub(keep.len()));
    rng.shuffle(&mut rest);
    rest.truncate(room);
    keep.extend(rest);
    keep
}

pub fn heuristic_scores(s: &State, ctx: &Ctx, acts: &[Act]) -> Vec<f32> {
    let p = &s.players[ctx.pi];
    let hand_n = p.hand.len();
    let hills_placed = 2 - p.hills_in_hand as i32;
    let enemy_carrier = !ctx.carriers.is_empty();
    let may_attack = !s.used[2] && p.carry < 0;
    let mut out = vec![0f32; acts.len()];
    let mut f = [0f32; 16];
    for (i, a) in acts.iter().enumerate() {
        let v: f64 = match a.t {
            A_TMOVE => if s.cells[a.a as usize / 3].hill == ctx.pi as i8 { 100.0 } else { 50.0 },
            A_TSTAY => 1.0,
            A_TRANSPORT => if a.a == 1 { if ctx.cur_hill_dist > 6 { 60.0 } else { -50.0 } } else { 70.0 },
            A_FIGHT => 68.0,
            A_LOAD => 80.0,
            A_UNLOAD => 0.0,
            A_HIDE | A_GUESS => 0.0,
            A_REVENGE => 90.0,
            A_DECLINE => 0.0,
            A_MOVE => {
                let cid = a.a as usize / 3;
                let fr = ctx.free_res.iter().find(|r| r.node == a.a);
                let ca = ctx.carriers.iter().find(|c| c.cell as usize == cid);
                if p.carry < 0 && fr.is_some() { 72.0 + fr.unwrap().val as f64 }
                else if may_attack && ca.is_some() { 55.0 + 3.0 * ca.unwrap().val as f64 }
                else if p.carry >= 0 {
                    let dh = if ctx.dist_hill[cid] >= 0 { ctx.dist_hill[cid] as i32 } else { 99 };
                    let gain = ctx.cur_hill_dist.min(30) - dh.min(30);
                    if s.used[3] { 2.0 } else if gain > 0 { 18.0 + gain as f64 * 3.0 } else { 0.0 }
                } else { 1.0 }
            }
            A_PLACE => {
                f.fill(0.0);
                action_numeric(s, ctx, *a, &mut f, 0);
                let (reach, d_ant, to_res) = (f[10] as f64, f[2] as f64, f[11] as f64);
                if a.c == 1 { if hills_placed == 0 { 90.0 } else if !ctx.free_res.is_empty() { 30.0 + 6.0 * (1.0 - to_res) } else { 22.0 } }
                else if a.c >= 2 { if p.ant.is_some() { 52.0 + 14.0 * reach - 10.0 * d_ant } else { 0.0 } }
                else { (if hand_n > 2 { 22.0 } else { 12.0 }) + 6.0 * reach + 6.0 * (1.0 - to_res) }
            }
            A_INC_TILES => if hand_n <= 1 { 68.0 } else if hand_n <= 3 { 40.0 } else { 28.0 },
            A_INC_CARD => if p.cards.is_empty() { 48.0 } else { 30.0 },
            A_INC_HILL => if hills_placed == 2 && p.ant.is_some() && ctx.free_res.is_empty() { 8.0 } else { -5.0 },
            A_SWAP => 4.0,
            A_WINGS => {
                let cid = a.a as usize / 3;
                let fr = ctx.free_res.iter().find(|r| r.node == a.a);
                let ca = ctx.carriers.iter().find(|c| c.cell as usize == cid);
                let in_reach = ctx.dist_ant[cid] >= 0;
                if p.carry < 0 && fr.is_some() && !in_reach { 58.0 }
                else if may_attack && ca.is_some() && !in_reach { 56.0 } else { -10.0 }
            }
            A_BLOCKAGE => if enemy_carrier { 21.0 } else { 6.0 },
            A_BRIDGE => 24.0,
            A_SUPER => 9.0,
            A_OVERLAY => 12.0,
            A_SDONE => 14.0,
            A_SUNDO => 0.0,
            A_END => 16.0,
            _ => 0.0,
        };
        out[i] = v as f32;
    }
    out
}

pub fn heuristic_act(s: &State, noise: f64, rng: &mut Rng) -> Act {
    let acts = s.legal_actions();
    if acts.len() == 1 { return acts[0]; }
    let ctx = build_ctx(s, s.actor());
    let sc = heuristic_scores(s, &ctx, &acts);
    let (mut best, mut bv) = (0, -1e9f64);
    for i in 0..acts.len() {
        let v = sc[i] as f64 + (if noise == 0.0 { 0.0 } else { (rng.f64() - 0.5) * noise });
        if v > bv { bv = v; best = i; }
    }
    acts[best]
}

pub fn nn_act(net: &Net, s: &State, temp: f32, rng: &mut Rng) -> Act {
    let acts = s.legal_actions();
    if acts.len() == 1 { return acts[0]; }
    let acts = cap_candidates(acts, MAX_CANDS, rng);
    let pi = s.actor();
    let ctx = build_ctx(s, pi);
    let o = make_obs(s, &ctx, &acts, net.gver());
    let e = net.encode(&o, false);
    let fp = net.forward_policy(&e, &o.a, o.n, false);
    let p = softmax(&fp.logits, temp);
    acts[sample_idx(&p, rng)]
}

pub enum Agent<'a> { Heur(f64), Nn(&'a Net, f32), Search(Mcts<'a>) }

impl<'a> Agent<'a> {
    pub fn act(&self, s: &State, rng: &mut Rng) -> Act {
        match self {
            Agent::Heur(noise) => heuristic_act(s, *noise, rng),
            Agent::Nn(net, t) => nn_act(net, s, *t, rng),
            Agent::Search(m) => {
                let acts = s.legal_actions();
                if acts.len() == 1 { return acts[0]; }
                let r = m.search(s, m.sims, false, rng);
                let idx = pick_by_visits(&r.visits, 0.0, rng);
                r.acts[idx]
            }
        }
    }
}

/// returns (winner, turns)
pub fn play_game_turns(agents: &[Agent], rng: &mut Rng) -> (i32, i32) {
    let mut s = new_game(agents.len(), (rng.u64() >> 33) as i32);
    let mut guard = 0;
    while !s.over && guard < 6000 {
        guard += 1;
        let pi = s.actor();
        let a = agents[pi].act(&s, rng);
        s.apply(a);
    }
    (if !s.over { -1 } else { s.winner as i32 }, s.turn_no)
}

/// returns (winner, turns, final scores of seats 0 and 1)
pub fn play_game_scores(agents: &[Agent], rng: &mut Rng) -> (i32, i32, i32, i32) {
    let mut s = new_game(agents.len(), (rng.u64() >> 33) as i32);
    let mut guard = 0;
    while !s.over && guard < 6000 {
        guard += 1;
        let pi = s.actor();
        let a = agents[pi].act(&s, rng);
        s.apply(a);
    }
    (if !s.over { -1 } else { s.winner as i32 }, s.turn_no, s.score(0), s.score(1))
}

/// returns winner (-1 = draw)
pub fn play_game(agents: &[Agent], rng: &mut Rng) -> i32 {
    let mut s = new_game(agents.len(), (rng.u64() >> 33) as i32);
    let mut guard = 0;
    while !s.over && guard < 6000 {
        guard += 1;
        let pi = s.actor();
        let a = agents[pi].act(&s, rng);
        s.apply(a);
    }
    if !s.over { -1 } else { s.winner as i32 }
}
