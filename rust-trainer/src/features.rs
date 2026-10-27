// Port of js/features.js
use crate::engine::*;
use crate::fv::FV;

pub const NT: usize = 24;
pub const NUM_AF: usize = 16;
pub const AF: usize = NT + NUM_AF;
pub const SF: usize = 41;

#[derive(Clone, Copy, Default)]
pub struct Fr { pub cell: u8, pub val: u8, pub node: u8 }

pub struct Ctx {
    pub pi: usize, pub ant_cell: i32,
    pub dist_ant: [i16; 64], pub dist_hill: [i16; 64],
    pub free_res: FV<Fr, 12>, pub carriers: FV<Fr, 6>, pub enemy_pos: FV<u8, 16>,
    pub cur_hill_dist: i32, pub hs: FV<u8, 12>,
}

fn cell_min(b: &Bfs) -> [i16; 64] {
    let mut m = [-1i16; 64];
    for (n, d) in b.order.iter() {
        let c = *n as usize / 3;
        if m[c] < 0 || m[c] > *d { m[c] = *d; }
    }
    m
}

fn center(x: i32, y: i32) -> (f64, f64) {
    let h = 3f64.sqrt() / 2.0;
    let x0 = x as f64 * 0.5;
    let yf = y as f64;
    let v = if is_up(x, y) { [(x0 + 0.5, yf * h), (x0 + 1.0, (yf + 1.0) * h), (x0, (yf + 1.0) * h)] }
            else { [(x0, yf * h), (x0 + 1.0, yf * h), (x0 + 0.5, (yf + 1.0) * h)] };
    ((v[0].0 + v[1].0 + v[2].0) / 3.0, (v[0].1 + v[1].1 + v[2].1) / 3.0)
}
fn dist2(a: (i32, i32), b: (i32, i32)) -> f64 {
    let (p, q) = (center(a.0, a.1), center(b.0, b.1));
    ((p.0 - q.0).powi(2) + (p.1 - q.1).powi(2)).sqrt()
}

pub fn build_ctx(s: &State, pi: usize) -> Ctx {
    let p = &s.players[pi];
    let node = s.ant_node(pi);
    let dist_ant = if node >= 0 { let st = s.ant_starts(pi); cell_min(&s.bfs(pi, st.as_slice(), 999, true)) } else { [-1; 64] };
    let hs = s.hill_cells(pi);
    let dist_hill = if !hs.is_empty() {
        let st: Vec<u8> = hs.iter().map(|h| h * 3).collect();
        cell_min(&s.bfs(pi, &st, 999, false))
    } else { [-1; 64] };
    let mut free_res: FV<Fr, 12> = FV::new();
    for r in s.resources.iter() {
        if r.owner < 0 && r.carrier < 0 && r.cell >= 0 { free_res.push(Fr { cell: r.cell as u8, val: p.values[r.kind as usize], node: s.res_node(r.cell as usize) as u8 }); }
    }
    let mut carriers: FV<Fr, 6> = FV::new();
    let mut enemy_pos: FV<u8, 16> = FV::new();
    for q in 0..s.n {
        if q == pi { continue; }
        if let Some(qa) = &s.players[q].ant {
            enemy_pos.push(qa.cell);
            if s.players[q].carry >= 0 {
                let kind = s.resources[s.players[q].carry as usize].kind;
                carriers.push(Fr { cell: qa.cell, val: p.values[kind as usize], node: 0 });
            }
        }
        for h in s.hill_cells(q).iter() { enemy_pos.push(*h); }
    }
    let ant_cell = if node >= 0 { node / 3 } else { -1 };
    let cur_hill_dist = if ant_cell >= 0 && dist_hill[ant_cell as usize] >= 0 { dist_hill[ant_cell as usize] as i32 } else { 99 };
    Ctx { pi, ant_cell, dist_ant, dist_hill, free_res, carriers, enemy_pos, cur_hill_dist, hs }
}

#[inline] fn mn(a: i32, b: i32) -> i32 { if a < b { a } else { b } }

pub fn state_features(s: &State, ctx: &Ctx, out: &mut [f32; SF]) {
    let pi = ctx.pi;
    let p = &s.players[pi];
    out.fill(0.0);
    let mut best = 0;
    for q in 0..s.n { if q != pi { best = best.max(s.score(q)); } }
    let my = s.score(pi);
    let mut o = [0f64; SF];
    o[0] = my as f64 / 7.0; o[1] = best as f64 / 7.0;
    o[2] = mn(p.hand.len() as i32, 12) as f64 / 8.0;
    let mut uniq = 0; let mut seen = [false; 6];
    for &t in p.hand.iter() { if !seen[t as usize] { seen[t as usize] = true; uniq += 1; } }
    o[3] = uniq as f64 / 6.0; o[4] = p.cards.len() as f64 / 6.0;
    o[5] = p.hills_in_hand as f64 / 2.0; o[6] = (2 - p.hills_in_hand) as f64 / 2.0;
    o[7] = if p.ant.is_some() { 1.0 } else { 0.0 }; o[8] = if p.carry >= 0 { 1.0 } else { 0.0 };
    if p.carry >= 0 {
        let r = &s.resources[p.carry as usize];
        o[9] = p.values[r.kind as usize] as f64 / 3.0;
        o[10] = if r.laid == s.turn_no { 1.0 } else { 0.0 };
    }
    o[11] = ctx.carriers.len() as f64 / 3.0;
    o[12] = ctx.carriers.iter().fold(0, |m, c| m.max(c.val as i32)) as f64 / 3.0;
    o[13] = ctx.free_res.len() as f64 / 4.0;
    o[14] = s.res_stack.len() as f64 / 12.0; o[15] = s.tile_stack.len() as f64 / 50.0;
    for i in 0..4 { o[16 + i] = if s.used[i] { 1.0 } else { 0.0 }; }
    if let Some(pd) = &s.pending { o[20 + pd.kind as usize] = 1.0; }
    let d_free = ctx.free_res.iter().fold(99, |m, r| { let d = ctx.dist_ant[r.cell as usize]; if d >= 0 { mn(m, d as i32) } else { m } });
    o[25] = mn(d_free, 14) as f64 / 14.0;
    o[26] = mn(ctx.cur_hill_dist, 14) as f64 / 14.0;
    let d_car = ctx.carriers.iter().fold(99, |m, c| { let d = ctx.dist_ant[c.cell as usize]; if d >= 0 { mn(m, d as i32) } else { m } });
    o[27] = mn(d_car, 14) as f64 / 14.0;
    o[28] = p.values[0] as f64 / 3.0; o[29] = p.values[1] as f64 / 3.0; o[30] = p.values[2] as f64 / 3.0;
    o[31] = (s.turn_no as f64 / (60.0 * s.n as f64)).min(2.0);
    o[32] = s.n as f64 / 6.0;
    let b = |c: u8| if p.has_card(c) { 1.0 } else { 0.0 };
    o[33] = b(C_BRIDGE); o[34] = b(C_BLOCKAGE); o[35] = b(C_WINGS); o[36] = b(C_REVENGE); o[37] = b(C_TRANSPORT);
    o[38] = s.resources.iter().filter(|r| r.owner == pi as i8).count() as f64 / 6.0;
    o[39] = s.resources.iter().filter(|r| r.owner >= 0).count() as f64 / 12.0;
    o[40] = 1.0;
    for i in 0..SF { out[i] = o[i] as f32; }
}

fn free_res_at(ctx: &Ctx, cid: u8) -> Option<Fr> { ctx.free_res.iter().find(|r| r.cell == cid).copied() }
fn carrier_at(ctx: &Ctx, cid: u8) -> Option<Fr> { ctx.carriers.iter().find(|c| c.cell == cid).copied() }
fn min_dist_to(s: &State, xy: (i32, i32), list: &[u8]) -> f64 {
    let mut m = 99.0f64;
    for &t in list { m = m.min(dist2(xy, s.cells[t as usize].xy())); }
    m
}
#[inline] fn dget(d: &[i16; 64], c: usize, def: i32) -> i32 { if d[c] >= 0 { d[c] as i32 } else { def } }

/// numeric part of action features, written into out[off..off+NUM_AF)
pub fn action_numeric(s: &State, ctx: &Ctx, a: Act, out: &mut [f32], off: usize) {
    let p = &s.players[ctx.pi];
    let set_dest = |out: &mut [f32], cid: usize| {
        let fr = free_res_at(ctx, cid as u8);
        let ca = carrier_at(ctx, cid as u8);
        out[off] = (fr.map_or(ca.map_or(0, |c| c.val), |f| f.val)) as f32 / 3.0;
        let dh = dget(&ctx.dist_hill, cid, 14);
        out[off + 1] = (mn(dh, 14) as f64 / 14.0) as f32;
        out[off + 2] = (mn(dget(&ctx.dist_ant, cid, 14), 14) as f64 / 14.0) as f32;
        out[off + 3] = if fr.is_some() { 1.0 } else { 0.0 };
        out[off + 4] = if ca.is_some() { 1.0 } else { 0.0 };
        out[off + 5] = if s.cells[cid].hill == ctx.pi as i8 { 1.0 } else { 0.0 };
        out[off + 6] = ((mn(ctx.cur_hill_dist, 14) - mn(dh, 14)) as f64 / 6.0).clamp(-1.0, 1.0) as f32;
    };
    let open_sum = |t: usize| TILE_OPEN[t].iter().map(|&x| x as i32).sum::<i32>() as f64 / 3.0;
    match a.t {
        A_MOVE | A_TMOVE | A_WINGS => set_dest(out, a.a as usize / 3),
        A_LOAD => {
            set_dest(out, ctx.ant_cell as usize);
            out[off] = if !ctx.free_res.is_empty() { (free_res_at(ctx, ctx.ant_cell as u8).map_or(0, |f| f.val)) as f32 / 3.0 } else { 0.0 };
        }
        A_FIGHT => { out[off] = (carrier_at(ctx, ctx.ant_cell as u8).map_or(0, |c| c.val)) as f32 / 3.0; }
        A_TRANSPORT => { out[off + 1] = (mn(ctx.cur_hill_dist, 14) as f64 / 14.0) as f32; out[off + 15] = a.a as f32; }
        A_INC_HILL => { out[off + 15] = 0.0; }
        A_INC_TILES => { out[off + 15] = (mn(p.hand.len() as i32, 10) as f64 / 10.0) as f32; }
        A_INC_CARD => { out[off + 15] = (p.cards.len() as f64 / 6.0) as f32; }
        A_SWAP => {
            out[off + 15] = (p.hand.iter().filter(|&&t| t == a.b).count() as f64 / 4.0) as f32;
            out[off + 13] = open_sum(a.c as usize) as f32;
        }
        A_PLACE => {
            let (x, y) = (gx(a.g as i32), gy(a.g as i32));
            out[off + 7] = if a.c == 1 { 1.0 } else { 0.0 };
            out[off + 8] = if a.c >= 2 { 1.0 } else { 0.0 };
            out[off + 13] = open_sum(a.a as usize) as f32;
            out[off + 14] = (TILE_GROUPS_LEN[a.a as usize] as f64 / 3.0) as f32;
            let mut conn = 0; let mut reach = 0; let mut best_d = 99;
            for e in 0..3 {
                let (nx, ny, ne) = neighbor(x, y, e);
                if let Some(ns) = s.slot_at(nx, ny) {
                    if s.cells[ns].open(ne) == 1 {
                        conn += 1;
                        if ctx.dist_ant[ns] >= 0 { reach = 1; best_d = mn(best_d, ctx.dist_ant[ns] as i32 + 1); }
                    }
                }
            }
            out[off + 9] = (conn as f64 / 3.0) as f32;
            out[off + 10] = reach as f32;
            out[off + 2] = (mn(best_d, 14) as f64 / 14.0) as f32;
            let free_cells: Vec<u8> = ctx.free_res.iter().map(|r| r.cell).collect();
            out[off + 11] = if !free_cells.is_empty() { (min_dist_to(s, (x, y), &free_cells).min(10.0) / 10.0) as f32 } else { 1.0 };
            out[off + 12] = if !ctx.enemy_pos.is_empty() { (min_dist_to(s, (x, y), ctx.enemy_pos.as_slice()).min(10.0) / 10.0) as f32 } else { 1.0 };
            out[off + 1] = if !ctx.hs.is_empty() { (min_dist_to(s, (x, y), ctx.hs.as_slice()).min(10.0) / 10.0) as f32 } else { 1.0 };
        }
        A_BRIDGE => {
            let (ca, cb) = (a.a as usize, a.b as usize);
            let d = if ctx.dist_ant[ca] >= 0 { ctx.dist_ant[ca] as i32 } else if ctx.dist_ant[cb] >= 0 { ctx.dist_ant[cb] as i32 } else { 14 };
            out[off + 2] = (mn(d, 14) as f64 / 14.0) as f32;
            out[off + 9] = if (ctx.dist_ant[ca] >= 0) != (ctx.dist_ant[cb] >= 0) { 1.0 } else { 0.0 };
        }
        A_BLOCKAGE => {
            set_dest(out, a.a as usize);
            out[off + 12] = if !ctx.enemy_pos.is_empty() { (min_dist_to(s, s.cells[a.a as usize].xy(), ctx.enemy_pos.as_slice()).min(10.0) / 10.0) as f32 } else { 1.0 };
        }
        A_OVERLAY => {
            out[off + 13] = open_sum(a.b as usize) as f32;
            out[off + 2] = (mn(dget(&ctx.dist_ant, a.a as usize, 14), 14) as f64 / 14.0) as f32;
        }
        A_HIDE | A_GUESS => { out[off + 15] = a.a as f32; }
        A_REVENGE | A_DECLINE => {
            let pd = s.pending.as_ref().unwrap();
            let r = &s.resources[pd.res as usize];
            out[off] = (s.players[ctx.pi].values[r.kind as usize] as f64 / 3.0) as f32;
        }
        _ => {}
    }
}

pub const TILE_GROUPS_LEN: [usize; 6] = [1, 1, 1, 2, 2, 3];

pub fn action_features(s: &State, ctx: &Ctx, a: Act, out: &mut [f32], off: usize) {
    for i in 0..AF { out[off + i] = 0.0; }
    out[off + a.t as usize] = 1.0;
    action_numeric(s, ctx, a, out, off + NT);
}

pub fn candidate_matrix(s: &State, ctx: &Ctx, acts: &[Act]) -> Vec<f32> {
    let mut m = vec![0f32; acts.len() * AF];
    for (i, a) in acts.iter().enumerate() { action_features(s, ctx, *a, &mut m, i * AF); }
    m
}

// ---------- board graph for the graph network (see nn.rs) ----------
/// One node per path component (cell, comp); edges follow open tile edges (the same walkable graph the ants use).
/// Feature version 1: NF node features. Version 2 adds node features (enemy reach, regions, frontier), global extras (GXN) and
/// candidate extras (CXN: what a placement connects / opens, transport odds).
pub const NF: usize = 20;
pub const NF2: usize = 32;
pub const GXN: usize = 14;
pub const CXN: usize = 17;
/// Feature version 3 adds "is my ant cut off?" global extras and a one-action lookahead per candidate (see reach_info).
pub const GXN3: usize = 24;
pub const CXN3: usize = 21;

const R_ANT: u8 = 1; const R_FOE: u8 = 2; const R_MYH: u8 = 4; const R_EH: u8 = 8; const R_FREE: u8 = 16;

#[derive(Clone, Default)]
pub struct Graph {
    pub n: usize, pub nf: usize,
    pub x: Vec<f32>,            // n * nf node features (current player's view)
    pub nbr: Vec<[i16; 3]>,     // neighbour node across tile edge 0/1/2 (-1: none)
    pub base: Vec<u16>,         // first node of each cell slot (ncells + 1 entries)
    pub ant: i32, pub foe: i32, // node of own ant / first enemy ant (-1: none)
    pub cand: Vec<Vec<u16>>,    // per candidate action: nodes whose embedding describes the target (empty: none)
    pub gx: Vec<f32>,           // v2: global extras
    pub cx: Vec<f32>,           // v2: n_candidates * CXN
    pub region: Vec<u16>, pub rflags: Vec<u8>, // v2: connected component of every node, flag bits per component
    pub fv: u8, pub reg_front: Vec<u32>,        // feature version; v3: open frontier edges per component
}

/// Where player pi's ant can get to: any own hill (paths, bridges, tunnels), transport distance to the nearest own hill
/// (no tunnels; 99 = none), any free resource; plus the reach map (for unreachable hills).
pub struct Reach { pub any: bool, pub td: i32, pub free: bool, pub a: Option<Bfs> }
pub fn reach_info(s: &State, pi: usize) -> Reach {
    if s.players[pi].ant.is_none() { return Reach { any: false, td: 99, free: false, a: None }; }
    let st = s.ant_starts(pi);
    let a = s.bfs(pi, st.as_slice(), 999, true);
    let t = s.bfs(pi, st.as_slice(), 999, false);
    let (mut any, mut td) = (false, 99);
    for &h in s.hill_cells(pi).iter() { let n = h as usize * 3; if a.dist[n] >= 0 { any = true; } if t.dist[n] >= 0 { td = td.min(t.dist[n] as i32); } }
    let free = s.resources.iter().any(|r| r.owner < 0 && r.carrier < 0 && r.cell >= 0 && a.dist[s.res_node(r.cell as usize)] >= 0);
    Reach { any, td, free, a: Some(a) }
}
/// actions whose effect on the ant's connections is looked ahead (fv 3)
fn looked_ahead(t: u8) -> bool { matches!(t, A_PLACE | A_BRIDGE | A_OVERLAY | A_INC_HILL | A_WINGS | A_MOVE | A_TMOVE) }

fn merge_min(a: &mut [i16; 64], b: &[i16; 64]) { for i in 0..64 { if b[i] >= 0 && (a[i] < 0 || b[i] < a[i]) { a[i] = b[i]; } } }

pub fn build_graph(s: &State, ctx: &Ctx, fv: usize) -> Graph {
    let v2 = fv >= 2;
    let pi = ctx.pi;
    let p = &s.players[pi];
    let nc = s.ncells;
    let nf = if v2 { NF2 } else { NF };
    let mut base = vec![0u16; nc + 1];
    for cid in 0..nc { base[cid + 1] = base[cid] + s.cells[cid].ncomps() as u16; }
    let n = base[nc] as usize;
    let mut x = vec![0f32; n * nf];
    let mut nbr = vec![[-1i16; 3]; n];
    let ant_cell = p.ant.map(|a| a.cell as usize);
    let mut ant_node = -1i32;
    if let Some(a) = &p.ant { ant_node = base[a.cell as usize] as i32 + s.cells[a.cell as usize].comp_index(a.edge as usize) as i32; }
    let mut foe = -1i32;
    let mut foe_cells: FV<u8, 6> = FV::new();
    let mut foe_nodes: FV<u16, 6> = FV::new();
    for q in 0..s.n { if q == pi { continue; } if let Some(a) = &s.players[q].ant {
        foe_cells.push(a.cell);
        let nd = base[a.cell as usize] as i32 + s.cells[a.cell as usize].comp_index(a.edge as usize) as i32;
        foe_nodes.push(nd as u16);
        if foe < 0 { foe = nd; }
    } }
    let carry_val = if p.carry >= 0 { p.values[s.resources[p.carry as usize].kind as usize] as f32 / 3.0 } else { 0.0 };
    for cid in 0..nc {
        let c = &s.cells[cid];
        let fr_res = ctx.free_res.iter().find(|r| r.cell as usize == cid);
        let ca = ctx.carriers.iter().find(|r| r.cell as usize == cid).map_or(0.0, |r| r.val as f32 / 3.0);
        let da = ctx.dist_ant[cid]; let dh = ctx.dist_hill[cid];
        let foe_here = foe_cells.iter().any(|&f| f as usize == cid);
        for ci in 0..c.ncomps() {
            let node = base[cid] as usize + ci;
            let mask = c.comp(ci);
            let f = &mut x[node * nf..(node + 1) * nf];
            f[0] = if is_up(c.x as i32, c.y as i32) { 1.0 } else { 0.0 };
            for e in 0..3 { f[1 + e] = ((mask >> e) & 1) as f32; }
            f[4] = c.ncomps() as f32 / 3.0;
            f[5] = if c.hill == pi as i8 { 1.0 } else { 0.0 };
            f[6] = if c.hill >= 0 && c.hill != pi as i8 { 1.0 } else { 0.0 };
            f[7] = fr_res.map_or(0.0, |r| if r.node as usize == cid * 3 + ci { r.val as f32 / 3.0 } else { 0.0 });
            f[8] = if c.block >= 0 { 1.0 } else { 0.0 };
            f[9] = if c.bridge > 0 { 1.0 } else { 0.0 };
            f[10] = if node as i32 == ant_node { 1.0 } else { 0.0 };
            f[11] = if ant_cell == Some(cid) { 1.0 } else { 0.0 };
            f[12] = if foe_here { 1.0 } else { 0.0 };
            f[13] = ca;
            f[14] = if da >= 0 { mn(da as i32, 14) as f32 / 14.0 } else { 1.0 };
            f[15] = if da >= 0 { 1.0 } else { 0.0 };
            f[16] = if dh >= 0 { mn(dh as i32, 14) as f32 / 14.0 } else { 1.0 };
            f[17] = if dh >= 0 { 1.0 } else { 0.0 };
            f[18] = if c.overlaid { 1.0 } else { 0.0 };
            f[19] = if ant_cell == Some(cid) { carry_val } else { 0.0 };
            for e in 0..3 {
                if (mask >> e) & 1 == 0 { continue; }
                let (nid, ne) = s.nb(cid, e);
                if let Some(nid) = nid {
                    let nc2 = &s.cells[nid];
                    if nc2.open(ne) == 1 { nbr[node][e] = (base[nid] as usize + nc2.comp_index(ne)) as i16; }
                }
            }
        }
    }
    let mut g = Graph { n, nf, x, nbr, base, ant: ant_node, foe, cand: Vec::new(), gx: Vec::new(), cx: Vec::new(), region: Vec::new(), rflags: Vec::new(), fv: fv as u8, reg_front: Vec::new() };
    if !v2 { return g; }

    // --- v2: regions (connected components of the static path graph) and their contents
    let mut region = vec![u16::MAX; n];
    let mut nreg = 0u16;
    let mut size: Vec<u32> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    for st in 0..n {
        if region[st] != u16::MAX { continue; }
        region[st] = nreg; stack.push(st); let mut cnt = 0u32;
        while let Some(v) = stack.pop() {
            cnt += 1;
            for e in 0..3 { let m = g.nbr[v][e]; if m >= 0 && region[m as usize] == u16::MAX { region[m as usize] = nreg; stack.push(m as usize); } }
        }
        size.push(cnt); nreg += 1;
    }
    let mut rflags = vec![0u8; nreg as usize];
    if ant_node >= 0 { rflags[region[ant_node as usize] as usize] |= R_ANT; }
    for &fnode in foe_nodes.iter() { rflags[region[fnode as usize] as usize] |= R_FOE; }
    for cid in 0..nc {
        let c = &s.cells[cid];
        let fr = ctx.free_res.iter().find(|r| r.cell as usize == cid);
        for ci in 0..c.ncomps() {
            let r = region[g.base[cid] as usize + ci] as usize;
            if c.hill == pi as i8 { rflags[r] |= R_MYH; }
            if c.hill >= 0 && c.hill != pi as i8 { rflags[r] |= R_EH; }
            if fr.map_or(false, |f| f.node as usize == cid * 3 + ci) { rflags[r] |= R_FREE; }
        }
    }
    // --- v2: enemy reach / enemy hill distances (cell minimum over all enemies, each with its own bridges)
    let mut dist_foe = [-1i16; 64]; let mut dist_eh = [-1i16; 64];
    for q in 0..s.n {
        if q == pi { continue; }
        if s.players[q].ant.is_some() {
            let st = s.ant_starts(q);
            merge_min(&mut dist_foe, &cell_min(&s.bfs(q, st.as_slice(), 999, true)));
        }
        let hs = s.hill_cells(q);
        if !hs.is_empty() {
            let st: Vec<u8> = hs.iter().map(|h| h * 3).collect();
            merge_min(&mut dist_eh, &cell_min(&s.bfs(q, &st, 999, false)));
        }
    }
    let mut reg_front = vec![0u32; nreg as usize];
    let laid_now: Vec<usize> = s.resources.iter().filter(|r| r.owner < 0 && r.carrier < 0 && r.cell >= 0 && r.laid == s.turn_no).map(|r| r.cell as usize).collect();
    for cid in 0..nc {
        let c = &s.cells[cid];
        let (dfo, deh) = (dist_foe[cid], dist_eh[cid]);
        for ci in 0..c.ncomps() {
            let node = g.base[cid] as usize + ci;
            let mask = c.comp(ci);
            let r = region[node] as usize; let fl = rflags[r];
            let mut fr_edges = 0;
            for e in 0..3 { if (mask >> e) & 1 == 1 { let (nid, _) = s.nb(cid, e); if nid.is_none() { fr_edges += 1; } } }
            reg_front[r] += fr_edges;
            let f = &mut g.x[node * nf..(node + 1) * nf];
            f[20] = if dfo >= 0 { mn(dfo as i32, 14) as f32 / 14.0 } else { 1.0 };
            f[21] = if dfo >= 0 { 1.0 } else { 0.0 };
            f[22] = if deh >= 0 { mn(deh as i32, 14) as f32 / 14.0 } else { 1.0 };
            f[23] = if deh >= 0 { 1.0 } else { 0.0 };
            f[24] = if fl & R_ANT != 0 { 1.0 } else { 0.0 };
            f[25] = if fl & R_FOE != 0 { 1.0 } else { 0.0 };
            f[26] = if fl & R_MYH != 0 { 1.0 } else { 0.0 };
            f[27] = if fl & R_EH != 0 { 1.0 } else { 0.0 };
            f[28] = if fl & R_FREE != 0 { 1.0 } else { 0.0 };
            f[29] = size[r] as f32 / n as f32;
            f[30] = fr_edges as f32 / 3.0;
            f[31] = if laid_now.contains(&cid) { 1.0 } else { 0.0 };
        }
    }
    // --- v2: global extras
    let fo = (0..s.n).find(|&q| q != pi);
    let mut gx = vec![0f32; if fv >= 3 { GXN3 } else { GXN }];
    if let Some(q) = fo {
        let pq = &s.players[q];
        gx[0] = mn(pq.hand.len() as i32, 12) as f32 / 8.0;
        gx[1] = pq.cards.len() as f32 / 6.0;
        gx[2] = pq.hills_in_hand as f32 / 2.0;
        gx[3] = if pq.ant.is_some() { 1.0 } else { 0.0 };
        gx[4] = if pq.carry >= 0 { 1.0 } else { 0.0 };
        if pq.carry >= 0 { gx[5] = p.values[s.resources[pq.carry as usize].kind as usize] as f32 / 3.0; }
        gx[6] = 1.0;
        if pq.carry >= 0 { if let Some(a) = &pq.ant { let d = dist_eh[a.cell as usize]; if d >= 0 { gx[6] = mn(d as i32, 14) as f32 / 14.0; } } }
        gx[12] = s.resources.iter().filter(|r| r.owner == q as i8).count() as f32 / 6.0;
    } else { gx[6] = 1.0; }
    gx[7] = if ant_cell.map_or(false, |c| dist_foe[c] >= 0) { 1.0 } else { 0.0 };
    let (mut dm, mut df, mut nm, mut nfo, mut nreg_free) = (14i32, 14i32, 0, 0, 0);
    for r in ctx.free_res.iter() {
        let cid = r.cell as usize;
        if ctx.dist_ant[cid] >= 0 { dm = dm.min(ctx.dist_ant[cid] as i32); nm += 1; }
        if dist_foe[cid] >= 0 { df = df.min(dist_foe[cid] as i32); nfo += 1; }
        if ant_node >= 0 && region[g.base[cid] as usize + (r.node as usize - cid * 3)] == region[ant_node as usize] { nreg_free += 1; }
    }
    gx[8] = ((df - dm) as f32 / 14.0).clamp(-1.0, 1.0);
    gx[9] = nm as f32 / 4.0; gx[10] = nfo as f32 / 4.0;
    let mut dh_my = 14i32;
    for h in ctx.hs.iter() { let d = dist_foe[*h as usize]; if d >= 0 { dh_my = dh_my.min(d as i32); } }
    gx[11] = dh_my as f32 / 14.0;
    gx[13] = nreg_free as f32 / 4.0;
    g.gx = gx; g.region = region; g.rflags = rflags; g.reg_front = reg_front;
    g
}

/// v3 global extras: is the ant cut off, and what could change that
fn add_v3(g: &mut Graph, s: &State, pi: usize, r: &Reach) {
    let p = &s.players[pi];
    let b = |x: bool| if x { 1.0 } else { 0.0 };
    let gx = &mut g.gx;
    gx[14] = b(r.any);
    gx[15] = b(r.td < 99);
    gx[16] = if r.td < 99 { mn(r.td, 14) as f32 / 14.0 } else { 1.0 };
    gx[17] = if g.ant >= 0 { (g.reg_front[g.region[g.ant as usize] as usize] as f32 / 6.0).min(1.0) } else { 0.0 };
    gx[18] = b(r.free);
    gx[19] = b(p.carry >= 0 && r.td >= 99);
    gx[20] = b(p.ant.is_some() && !r.any && !r.free);
    gx[21] = match &r.a { Some(a) => s.hill_cells(pi).iter().filter(|&&h| a.dist[h as usize * 3] < 0).count() as f32 / 2.0, None => 0.0 };
    gx[22] = b(p.has_card(C_SUPER));
    gx[23] = b(p.cards.len() >= 2);
}

fn cell_nodes(g: &Graph, s: &State, cid: usize, out: &mut Vec<u16>) {
    if cid >= s.ncells { return; }
    for k in g.base[cid]..g.base[cid + 1] { out.push(k); }
}

/// P(roll >= d) and E[min(roll, d)] for one die (0..5) or two dice
fn roll_odds(two: bool, d: i32) -> (f32, f32) {
    let mut p = 0.0; let mut e = 0.0; let mut tot = 0.0;
    for a in 0..6 { for b in 0..(if two { 6 } else { 1 }) {
        let r = a + b; tot += 1.0;
        if r >= d { p += 1.0; }
        e += r.min(d) as f32;
    } }
    (p / tot, e / tot / 6.0)
}

/// nodes describing the target of every candidate (average of their embeddings is fed to the policy head); v2 also fills cx
pub fn attach_candidates(g: &mut Graph, s: &State, ctx: &Ctx, acts: &[Act]) {
    g.cand.clear();
    let v2 = g.nf == NF2;
    let v3 = g.fv >= 3;
    let cxn = if v3 { CXN3 } else { CXN };
    g.cx = if v2 { vec![0f32; acts.len() * cxn] } else { Vec::new() };
    let r0 = if v3 { Some(reach_info(s, ctx.pi)) } else { None };
    if let Some(r) = &r0 { add_v3(g, s, ctx.pi, r); }
    for (ci, a) in acts.iter().enumerate() {
        let mut v: Vec<u16> = Vec::new();
        match a.t {
            A_MOVE | A_TMOVE | A_WINGS => {
                let cid = a.a as usize / 3; let k = a.a as usize % 3;
                if cid < s.ncells { let k = k.min(s.cells[cid].ncomps().saturating_sub(1)); v.push(g.base[cid] + k as u16); }
            }
            A_PLACE => {
                let (x, y) = (gx(a.g as i32), gy(a.g as i32));
                let mut regs: Vec<u16> = Vec::new();
                let mut frontier = 0;
                let sh = make_shape(x, y, a.a as usize, a.b as usize);
                for e in 0..3 {
                    let (nx, ny, ne) = neighbor(x, y, e);
                    match s.slot_at(nx, ny) {
                        Some(ns) => if s.cells[ns].open(ne) == 1 {
                            let node = g.base[ns] + s.cells[ns].comp_index(ne) as u16;
                            v.push(node);
                            if v2 { let r = g.region[node as usize]; if !regs.contains(&r) { regs.push(r); } }
                        },
                        None => if sh.open[e] == 1 { frontier += 1; },
                    }
                }
                if v2 {
                    let cx = &mut g.cx[ci * cxn..(ci + 1) * cxn];
                    for e in 0..3 { cx[e] = sh.open[e] as f32; }
                    let same = |i: usize, j: usize| (0..sh.nc as usize).any(|k| (sh.comps[k] >> i) & 1 == 1 && (sh.comps[k] >> j) & 1 == 1);
                    cx[3] = if same(0, 1) { 1.0 } else { 0.0 }; cx[4] = if same(0, 2) { 1.0 } else { 0.0 }; cx[5] = if same(1, 2) { 1.0 } else { 0.0 };
                    cx[6] = frontier as f32 / 3.0;
                    let mut un = 0u8; let mut single_ant_free = false; let mut single_ant_foe = false;
                    for &r in regs.iter() { let f = g.rflags[r as usize]; un |= f; if f & R_ANT != 0 && f & R_FREE != 0 { single_ant_free = true; } if f & R_ANT != 0 && f & R_FOE != 0 { single_ant_foe = true; } }
                    cx[7] = if un & R_ANT != 0 { 1.0 } else { 0.0 }; cx[8] = if un & R_FOE != 0 { 1.0 } else { 0.0 }; cx[9] = if un & R_FREE != 0 { 1.0 } else { 0.0 };
                    cx[10] = if un & R_MYH != 0 { 1.0 } else { 0.0 }; cx[11] = if un & R_EH != 0 { 1.0 } else { 0.0 };
                    cx[12] = if regs.len() >= 2 { 1.0 } else { 0.0 };
                    cx[13] = if regs.len() >= 2 && un & R_ANT != 0 && un & R_FREE != 0 && !single_ant_free { 1.0 } else { 0.0 };
                    cx[14] = if regs.len() >= 2 && un & R_ANT != 0 && un & R_FOE != 0 && !single_ant_foe { 1.0 } else { 0.0 };
                }
            }
            A_BRIDGE => { cell_nodes(g, s, a.a as usize, &mut v); cell_nodes(g, s, a.b as usize, &mut v); }
            A_BLOCKAGE | A_OVERLAY | A_INC_HILL => cell_nodes(g, s, a.a as usize, &mut v),
            A_TRANSPORT => {
                if v2 {
                    let d = if ctx.cur_hill_dist >= 99 { 99 } else { ctx.cur_hill_dist };
                    let (pd, ea) = roll_odds(a.a == 1, d);
                    g.cx[ci * cxn + 15] = pd; g.cx[ci * cxn + 16] = ea;
                }
            }
            _ => {}
        }
        if let Some(r0) = &r0 { // lookahead: the ant's connections after this action
            let r1 = if looked_ahead(a.t) { let mut s1 = *s; s1.apply(*a); reach_info(&s1, ctx.pi) } else { Reach { any: r0.any, td: r0.td, free: r0.free, a: None } };
            let o = ci * cxn;
            g.cx[o + 17] = if r1.td < 99 { mn(r1.td, 14) as f32 / 14.0 } else { 1.0 };
            g.cx[o + 18] = if r1.any { 1.0 } else { 0.0 };
            let carrying = s.players[ctx.pi].carry >= 0;
            g.cx[o + 19] = if (carrying && r0.td >= 99 && r1.td < 99) || (!carrying && !r0.any && r1.any) { 1.0 } else { 0.0 };
            g.cx[o + 20] = if r1.free { 1.0 } else { 0.0 };
        }
        g.cand.push(v);
    }
}

/// Everything the network needs for one decision. `gv`: 0 = plain MLP input, 1/2 = graph feature version.
pub struct Obs { pub s: [f32; SF], pub a: Vec<f32>, pub n: usize, pub g: Option<Graph> }

pub fn make_obs(s: &State, ctx: &Ctx, acts: &[Act], gv: usize) -> Obs {
    let mut sf = [0f32; SF];
    state_features(s, ctx, &mut sf);
    let a = candidate_matrix(s, ctx, acts);
    let g = if gv > 0 { let mut g = build_graph(s, ctx, gv); attach_candidates(&mut g, s, ctx, acts); Some(g) } else { None };
    Obs { s: sf, a, n: acts.len(), g }
}
