// Port of js/engine.js (non-"full" legal-action paths only; that is all training/search/eval use).
// Cells are addressed by dense `slot` (insertion order); iteration over cells follows ascending grid id
// exactly like JS integer-keyed objects, so action order matches the JS engine.
use crate::fv::FV;

pub const G: i32 = 128;
const GOFF: i32 = 64;
pub const MAX_ACTIONS: i32 = 70;
pub const MAX_ROUNDS: i32 = 90;
pub const WIN_POINTS: i32 = 7;

pub const TILE_OPEN: [[u8; 3]; 6] = [[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1]];
const TILE_GROUPS: [&[u8]; 6] = [&[0b111], &[0b011], &[0b001], &[0b011, 0b100], &[0b001, 0b010], &[0b001, 0b010, 0b100]];
const TILE_COUNT: [usize; 6] = [8, 12, 6, 10, 8, 6];

// card types
pub const C_BRIDGE: u8 = 0;
pub const C_SUPER: u8 = 1;
pub const C_BLOCKAGE: u8 = 2;
pub const C_WINGS: u8 = 3;
pub const C_REVENGE: u8 = 4;
pub const C_TRANSPORT: u8 = 5;
const DROP_RANK: [u8; 6] = [1, 2, 0, 5, 4, 3];

// action types (index = ACTION_TYPES in features.js)
pub const A_INC_TILES: u8 = 0;
pub const A_INC_CARD: u8 = 1;
pub const A_INC_HILL: u8 = 2;
pub const A_SWAP: u8 = 3;
pub const A_PLACE: u8 = 4;
/// Keep in sync with RULES_VERSION in js/engine.js (bumped whenever rules / action generation change).
pub const RULES_VERSION: u32 = 4;
pub const A_MOVE: u8 = 5;
pub const A_LOAD: u8 = 6;
pub const A_UNLOAD: u8 = 7;
pub const A_FIGHT: u8 = 8;
pub const A_TRANSPORT: u8 = 9;
pub const A_TMOVE: u8 = 10;
pub const A_TSTAY: u8 = 11;
pub const A_BRIDGE: u8 = 12;
pub const A_BLOCKAGE: u8 = 13;
pub const A_WINGS: u8 = 14;
pub const A_SUPER: u8 = 15;
pub const A_OVERLAY: u8 = 16;
pub const A_SDONE: u8 = 17;
pub const A_SUNDO: u8 = 18;
pub const A_HIDE: u8 = 19;
pub const A_GUESS: u8 = 20;
pub const A_REVENGE: u8 = 21;
pub const A_DECLINE: u8 = 22;
pub const A_END: u8 = 23;

// pending types
pub const P_HIDE: u8 = 0;
pub const P_GUESS: u8 = 1;
pub const P_REVENGE: u8 = 2;
pub const P_TMOVE: u8 = 3;
pub const P_SUPER: u8 = 4;

/// Compact action. Field use per type:
/// inc_hill a=cell | swap a=q b=give c=take | place g=gid a=tile b=rot c=piece | move/tmove a=node | fight a=q
/// transport a=card | bridge a,b=cells | blockage a=cell | wings a=node b,c=via cards(255=none)
/// overlay a=cell b=tile c=rot | hide/guess a=v
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
pub struct Act { pub t: u8, pub a: u8, pub b: u8, pub c: u8, pub g: u16 }
impl Act {
    #[inline] pub fn new(t: u8) -> Act { Act { t, a: 0, b: 0, c: 0, g: 0 } }
    #[inline] pub fn a1(t: u8, a: usize) -> Act { Act { t, a: a as u8, b: 0, c: 0, g: 0 } }
    #[inline] pub fn a3(t: u8, a: usize, b: usize, c: usize) -> Act { Act { t, a: a as u8, b: b as u8, c: c as u8, g: 0 } }
}

// ---------- geometry ----------
#[inline] pub fn gid(x: i32, y: i32) -> i32 { (y + GOFF) * G + (x + GOFF) }
#[inline] pub fn gx(id: i32) -> i32 { id % G - GOFF }
#[inline] pub fn gy(id: i32) -> i32 { id / G - GOFF }
#[inline] pub fn is_up(x: i32, y: i32) -> bool { ((x + y) & 1) == 0 }
#[inline]
pub fn neighbor(x: i32, y: i32, e: usize) -> (i32, i32, usize) {
    match e {
        0 => (x - 1, y, 1),
        1 => (x + 1, y, 0),
        _ => if is_up(x, y) { (x, y + 1, 2) } else { (x, y - 1, 2) },
    }
}
#[inline]
fn edge_pos(x: i32, y: i32, e: usize) -> usize {
    let ord: [usize; 3] = if is_up(x, y) { [1, 2, 0] } else { [2, 1, 0] };
    ord.iter().position(|&o| o == e).unwrap()
}

// ---------- cells ----------
#[derive(Clone, Copy, Default, PartialEq)]
pub struct Shape { pub open: [u8; 3], pub nc: u8, pub comps: [u8; 3], pub all_open: u8 }

pub fn make_shape(x: i32, y: i32, tile: usize, rot: usize) -> Shape {
    let mut open = [0u8; 3];
    let mut slot = [0usize; 3];
    for e in 0..3 {
        let t = (edge_pos(x, y, e) + rot) % 3;
        slot[e] = t;
        open[e] = TILE_OPEN[tile][t];
    }
    let mut comps = [0u8; 3];
    let mut nc = 0;
    for &g in TILE_GROUPS[tile] {
        let mut m = 0u8;
        for e in 0..3 { if (g >> slot[e]) & 1 == 1 && open[e] == 1 { m |= 1 << e; } }
        if m != 0 { comps[nc] = m; nc += 1; }
    }
    let mut all_open = 0u8;
    for e in 0..3 { if open[e] == 1 { all_open |= 1 << e; } }
    Shape { open, nc: nc as u8, comps, all_open }
}

fn shape_eq(a: &Shape, b: &Shape) -> bool {
    if a.open != b.open || a.nc != b.nc { return false; }
    let mut x = a.comps; let mut y = b.comps;
    x[..a.nc as usize].sort_unstable(); y[..b.nc as usize].sort_unstable();
    x == y
}

#[derive(Clone, Copy, Default)]
pub struct Cell {
    pub gid: i32, pub x: i8, pub y: i8, pub tile: u8, pub rot: u8,
    pub sh: Shape,
    pub hill: i8, pub res: i8, pub block: i8, pub bridge: u8, pub overlaid: bool,
}
impl Cell {
    #[inline] pub fn xy(&self) -> (i32, i32) { (self.x as i32, self.y as i32) }
    #[inline] pub fn open(&self, e: usize) -> u8 { self.sh.open[e] }
    #[inline] pub fn ncomps(&self) -> usize { if self.hill >= 0 || self.res >= 0 { 1 } else { self.sh.nc as usize } }
    #[inline] pub fn comp(&self, ci: usize) -> u8 {
        if self.hill >= 0 || self.res >= 0 { if ci == 0 { self.sh.all_open } else { 0 } }
        else if ci < self.sh.nc as usize { self.sh.comps[ci] } else { 0 }
    }
    #[inline] pub fn comp_index(&self, edge: usize) -> usize {
        if self.hill >= 0 || self.res >= 0 { return 0; }
        for i in 0..self.sh.nc as usize { if (self.sh.comps[i] >> edge) & 1 == 1 { return i; } }
        0
    }
}

// ---------- state ----------
#[derive(Clone, Copy, Default)]
pub struct Ant { pub cell: u8, pub edge: u8, pub free: bool }

#[derive(Clone, Copy, Default)]
pub struct Res { pub kind: u8, pub cell: i8, pub carrier: i8, pub owner: i8, pub laid: i32 }

#[derive(Clone, Copy, Default)]
pub struct Player {
    pub hand: FV<u8, 56>, pub cards: FV<u8, 18>,
    pub hills_in_hand: i8, pub bridges: i8, pub blockages: i8,
    pub ant: Option<Ant>, pub carry: i8, pub values: [u8; 3], pub own_turns: i32,
}
impl Player {
    pub fn has_card(&self, t: u8) -> bool { self.cards.contains(t) }
}

#[derive(Clone, Copy, Default)]
pub struct Ov { pub cell: u8, pub tile: u8, pub rot: u8, pub new_tile: u8 }

#[derive(Clone, Copy, Default)]
pub struct Pending {
    pub kind: u8, pub actor: u8, pub att: u8, pub def: u8, pub res: i8, pub hide: i8,
    pub att_win: bool, pub guess_v: i8, pub n: u8, pub overlays: FV<Ov, 56>,
}

#[derive(Clone, Copy, Default)]
pub struct Bridge { pub owner: u8, pub a: u8, pub b: u8, pub placed: i32 }

#[derive(Clone, Copy)]
pub struct State {
    pub seed: i32, pub n: usize, pub players: [Player; 6],
    pub grid: [u8; (G * G) as usize], pub cells: [Cell; 64], pub ncells: usize, pub order: FV<u8, 64>,
    pub tile_stack: FV<u8, 56>, pub res_stack: FV<u8, 12>, pub card_stack: FV<u8, 18>,
    pub resources: FV<Res, 12>, pub hills: FV<(u8, u8), 12>, pub blocks: FV<(u8, u8), 12>, pub bridges: FV<Bridge, 12>,
    pub cur: usize, pub turn_no: i32, pub used: [bool; 4], pub actions_this_turn: i32,
    pub pending: Option<Pending>, pub winner: i8, pub over: bool,
    // anti-loop bookkeeping for AI candidate generation (mirrors JS moveTrail / lastT)
    pub move_trail: FV<u8, 80>, pub last_t: u8,
}

// used[]: 0 income, 1 swap, 2 fight, 3 transport

// ---------- rng (mulberry32 embedded in state; identical to JS) ----------
pub fn rnd(s: &mut State) -> f64 {
    s.seed = s.seed.wrapping_add(0x6d2b79f5u32 as i32);
    let seed = s.seed;
    let t = (seed ^ ((seed as u32) >> 15) as i32).wrapping_mul(1 | seed);
    let t = t.wrapping_add((t ^ ((t as u32) >> 7) as i32).wrapping_mul(61 | t)) ^ t;
    (((t ^ ((t as u32) >> 14) as i32) as u32) as f64) / 4294967296.0
}
#[inline] fn rint(s: &mut State, n: usize) -> usize { (rnd(s) * n as f64).floor() as usize }
fn shuffle_u8(s: &mut State, a: &mut [u8]) {
    let mut i = a.len();
    while i > 1 {
        i -= 1;
        let j = rint(s, i + 1);
        a.swap(i, j);
    }
}

impl State {
    #[inline]
    pub fn slot_at(&self, x: i32, y: i32) -> Option<usize> {
        if x < -GOFF || x >= GOFF || y < -GOFF || y >= GOFF { return None; }
        let v = self.grid[gid(x, y) as usize];
        if v == 255 { None } else { Some(v as usize) }
    }
    /// neighbour slot across edge e of cell `slot`, plus the neighbour's edge index
    #[inline]
    pub fn nb(&self, slot: usize, e: usize) -> (Option<usize>, usize) {
        let c = &self.cells[slot];
        let (nx, ny, ne) = neighbor(c.x as i32, c.y as i32, e);
        (self.slot_at(nx, ny), ne)
    }

    pub fn put_cell(&mut self, x: i32, y: i32, tile: usize, rot: usize) -> usize {
        let sh = make_shape(x, y, tile, rot);
        let id = gid(x, y);
        let slot = self.ncells;
        self.ncells += 1;
        self.cells[slot] = Cell { gid: id, x: x as i8, y: y as i8, tile: tile as u8, rot: rot as u8, sh, hill: -1, res: -1, block: -1, bridge: 0, overlaid: false };
        self.grid[id as usize] = slot as u8;
        let mut pos = self.order.len();
        for (k, &o) in self.order.iter().enumerate() { if self.cells[o as usize].gid > id { pos = k; break; } }
        self.order.insert(pos, slot as u8);
        slot
    }

    pub fn actor(&self) -> usize { match &self.pending { Some(p) => p.actor as usize, None => self.cur } }

    pub fn score(&self, pi: usize) -> i32 {
        let v = &self.players[pi].values;
        let mut t = 0;
        for r in self.resources.iter() { if r.owner as usize == pi && r.owner >= 0 { t += v[r.kind as usize] as i32; } }
        t
    }
    fn finish_by_score(&mut self) {
        let mut best = -1; let mut bi = -1i32; let mut tie = false;
        for i in 0..self.n {
            let sc = self.score(i);
            if sc > best { best = sc; bi = i as i32; tie = false; } else if sc == best { tie = true; }
        }
        self.over = true;
        self.winner = if tie { -1 } else { bi as i8 };
    }
    fn check_win(&mut self) {
        for i in 0..self.n { if self.score(i) >= WIN_POINTS { self.over = true; self.winner = i as i8; return; } }
        if self.resources.iter().filter(|r| r.owner >= 0).count() >= 12 { self.finish_by_score(); }
    }

    fn start_turn(&mut self) {
        let cur = self.cur;
        self.turn_no += 1;
        self.players[cur].own_turns += 1;
        let mut cnt = 0;
        for b in self.blocks.iter() { if b.0 as usize == cur { self.cells[b.1 as usize].block = -1; cnt += 1; } }
        self.players[cur].blockages += cnt;
        let mut keep: FV<(u8, u8), 12> = FV::new();
        for b in self.blocks.iter() { if b.0 as usize != cur { keep.push(*b); } }
        self.blocks = keep;
        self.used = [false; 4];
        self.actions_this_turn = 0;
        self.move_trail = FV::new(); self.last_t = 255;
    }
    fn end_turn(&mut self) {
        let cur = self.cur;
        let own = self.players[cur].own_turns;
        let mut keep: FV<Bridge, 12> = FV::new();
        let bl: Vec<Bridge> = self.bridges.iter().copied().collect();
        for b in bl {
            if b.owner as usize == cur && b.placed < own {
                self.cells[b.a as usize].bridge -= 1; self.cells[b.b as usize].bridge -= 1; self.players[cur].bridges += 1;
            } else { keep.push(b); }
        }
        self.bridges = keep;
        self.cur = (self.cur + 1) % self.n;
        if self.turn_no / self.n as i32 >= MAX_ROUNDS { return self.finish_by_score(); }
        self.start_turn();
    }

    // ---------- path graph ----------
    pub fn ant_node(&self, pi: usize) -> i32 {
        match &self.players[pi].ant {
            None => -1,
            Some(a) => a.cell as i32 * 3 + self.cells[a.cell as usize].comp_index(a.edge as usize) as i32,
        }
    }
    pub fn ant_starts(&self, pi: usize) -> FV<u8, 3> {
        let mut out = FV::new();
        if let Some(a) = &self.players[pi].ant {
            if a.free { for i in 0..self.cells[a.cell as usize].ncomps() { out.push(a.cell * 3 + i as u8); } }
            else { out.push(self.ant_node(pi) as u8); }
        }
        out
    }
    pub fn hill_cells(&self, pi: usize) -> FV<u8, 12> {
        let mut o = FV::new();
        for h in self.hills.iter() { if h.0 as usize == pi { o.push(h.1); } }
        o
    }

    fn expand(&self, pi: usize, node: usize, underground: bool, f: &mut impl FnMut(usize)) {
        let cid = node / 3; let ci = node - cid * 3;
        let c = &self.cells[cid];
        let comp = c.comp(ci);
        for e in 0..3 {
            if (comp >> e) & 1 == 0 { continue; }
            let (nid, ne) = self.nb(cid, e);
            if let Some(nid) = nid {
                let nc = &self.cells[nid];
                if nc.block < 0 && nc.open(ne) == 1 { f(nid * 3 + nc.comp_index(ne)); }
            }
        }
        for b in self.bridges.iter() {
            if b.owner as usize != pi { continue; }
            let other = if b.a as usize == cid { b.b as i32 } else if b.b as usize == cid { b.a as i32 } else { -1 };
            if other >= 0 && self.cells[other as usize].block < 0 {
                for i in 0..self.cells[other as usize].ncomps() { f(other as usize * 3 + i); }
            }
        }
        if underground && c.hill as i32 == pi as i32 {
            for h in self.hills.iter() {
                if h.0 as usize == pi && h.1 as usize != cid && self.cells[h.1 as usize].block < 0 { f(h.1 as usize * 3); }
            }
        }
    }

    pub fn bfs(&self, pi: usize, starts: &[u8], max_depth: i32, underground: bool) -> Bfs {
        let mut r = Bfs { dist: [-1; 192], order: FV::new() };
        let mut frontier: FV<u8, 192> = FV::new();
        for &st in starts { if r.dist[st as usize] < 0 { r.dist[st as usize] = 0; r.order.push((st, 0)); frontier.push(st); } }
        let mut d = 1;
        while d <= max_depth && !frontier.is_empty() {
            let mut next: FV<u8, 192> = FV::new();
            for &n in frontier.iter() {
                let dist = &mut r.dist; let order = &mut r.order;
                self.expand(pi, n as usize, underground, &mut |m| {
                    if dist[m] < 0 { dist[m] = d as i16; order.push((m as u8, d as i16)); next.push(m as u8); }
                });
            }
            frontier = next;
            d += 1;
        }
        r
    }

    fn any_card_in_grid_dist2(&self, x: i32, y: i32, pred: impl Fn(&Cell) -> bool) -> bool {
        let test = |x: i32, y: i32| self.slot_at(x, y).map_or(false, |s| pred(&self.cells[s]));
        if test(x, y) { return true; }
        let mut seen: [(i32, i32); 24] = [(0, 0); 24];
        let mut ns = 1; seen[0] = (x, y);
        let mut frontier: [(i32, i32); 12] = [(0, 0); 12];
        let mut nf = 1; frontier[0] = (x, y);
        for _d in 1..=2 {
            let mut next: [(i32, i32); 12] = [(0, 0); 12];
            let mut nn = 0;
            for k in 0..nf {
                let (cx, cy) = frontier[k];
                for e in 0..3 {
                    let (nx, ny, _) = neighbor(cx, cy, e);
                    if seen[..ns].contains(&(nx, ny)) { continue; }
                    seen[ns] = (nx, ny); ns += 1;
                    if test(nx, ny) { return true; }
                    next[nn] = (nx, ny); nn += 1;
                }
            }
            frontier = next; nf = nn;
        }
        false
    }
    fn has_own_hill_near(&self, x: i32, y: i32, pi: usize) -> bool { self.any_card_in_grid_dist2(x, y, |c| c.hill == pi as i8) }
    fn res_on_board_near(&self, x: i32, y: i32) -> bool { self.any_card_in_grid_dist2(x, y, |c| c.res >= 0) }

    fn frontier_cells(&self) -> Vec<(i32, i32)> {
        let mut seen = [0u64; (G * G / 64) as usize];
        let mut out = Vec::with_capacity(64);
        for &sl in self.order.iter() {
            let c = &self.cells[sl as usize];
            for e in 0..3 {
                let (nx, ny, _) = neighbor(c.x as i32, c.y as i32, e);
                if self.slot_at(nx, ny).is_some() { continue; }
                let id = gid(nx, ny) as usize;
                if (seen[id >> 6] >> (id & 63)) & 1 == 1 { continue; }
                seen[id >> 6] |= 1 << (id & 63);
                out.push((nx, ny));
            }
        }
        out
    }

    fn placement_ok(&self, x: i32, y: i32, sh: &Shape) -> bool {
        let mut conn = 0;
        for e in 0..3 {
            let (nx, ny, ne) = neighbor(x, y, e);
            if let Some(s) = self.slot_at(nx, ny) {
                if self.cells[s].open(ne) != sh.open[e] { return false; }
                if sh.open[e] == 1 { conn += 1; }
            }
        }
        conn > 0
    }

    fn has_piece_or_ant(&self, slot: usize) -> bool {
        let c = &self.cells[slot];
        if c.hill >= 0 || c.res >= 0 || c.block >= 0 || c.bridge > 0 { return true; }
        for q in 0..self.n { if let Some(a) = &self.players[q].ant { if a.cell as usize == slot { return true; } } }
        false
    }

    pub fn consistent(&self) -> bool {
        for &sl in self.order.iter() {
            let c = &self.cells[sl as usize];
            for e in 0..3 {
                let (nid, ne) = self.nb(sl as usize, e);
                if let Some(n) = nid { if self.cells[n].open(ne) != c.open(e) { return false; } }
            }
        }
        true
    }

    fn removed_card(&mut self, pi: usize, t: u8) {
        let i = self.players[pi].cards.index_of(t).unwrap();
        self.players[pi].cards.remove(i);
        self.card_stack.insert(0, t);
    }

    pub fn interesting_nodes(&self, pi: usize) -> FV<u8, 96> {
        let mut out: FV<u8, 96> = FV::new();
        let add = |out: &mut FV<u8, 96>, n: u8| { if !out.contains(n) { out.push(n); } };
        for r in self.resources.iter() {
            if r.owner < 0 && r.carrier < 0 && r.cell >= 0 {
                for i in 0..self.cells[r.cell as usize].ncomps() { add(&mut out, r.cell as u8 * 3 + i as u8); }
            }
        }
        for q in 0..self.n {
            if q == pi { continue; }
            if let Some(a) = &self.players[q].ant {
                if self.players[q].carry >= 0 { for i in 0..self.cells[a.cell as usize].ncomps() { add(&mut out, a.cell * 3 + i as u8); } }
            }
        }
        for h in self.hill_cells(pi).iter() { add(&mut out, h * 3); }
        out
    }

    // ---------- legal actions ----------
    fn rots_unique(x: i32, y: i32, tile: usize) -> ([(u8, Shape); 3], usize) {
        let mut out = [(0u8, Shape::default()); 3];
        let mut n = 0;
        for r in 0..3 {
            let sh = make_shape(x, y, tile, r);
            let mut dup = false;
            for k in 0..n { if shape_eq(&out[k].1, &sh) { dup = true; break; } }
            if !dup { out[n] = (r as u8, sh); n += 1; }
        }
        (out, n)
    }

    fn uniq(v: &[u8]) -> FV<u8, 56> {
        let mut o = FV::new();
        for &x in v { if !o.contains(x) { o.push(x); } }
        o
    }

    pub fn placement_actions(&self, pi: usize, acts: &mut Vec<Act>) {
        let p = &self.players[pi];
        let tiles = Self::uniq(p.hand.as_slice());
        if tiles.is_empty() { return; }
        for (x, y) in self.frontier_cells() {
            let id = gid(x, y) as u16;
            let mut hill_ok: Option<bool> = None;
            let mut res_ok: Option<bool> = None;
            for &t in tiles.iter() {
                let (rs, nr) = Self::rots_unique(x, y, t as usize);
                for k in 0..nr {
                    let (r, sh) = &rs[k];
                    if !self.placement_ok(x, y, sh) { continue; }
                    acts.push(Act { t: A_PLACE, a: t, b: *r, c: 0, g: id });
                    if p.hills_in_hand > 0 {
                        if hill_ok.is_none() { hill_ok = Some(!self.res_on_board_near(x, y)); }
                        if hill_ok.unwrap() { acts.push(Act { t: A_PLACE, a: t, b: *r, c: 1, g: id }); }
                    }
                    if !self.res_stack.is_empty() {
                        if res_ok.is_none() { res_ok = Some(!self.has_own_hill_near(x, y, pi)); }
                        if res_ok.unwrap() { acts.push(Act { t: A_PLACE, a: t, b: *r, c: 2, g: id }); }
                    }
                }
            }
        }
    }

    fn overlay_actions(&self, pi: usize, acts: &mut Vec<Act>, first_only: bool) {
        let p = &self.players[pi];
        let tiles = Self::uniq(p.hand.as_slice());
        for &sl in self.order.iter() {
            let c = &self.cells[sl as usize];
            if c.overlaid || self.has_piece_or_ant(sl as usize) { continue; }
            for &t in tiles.iter() {
                let (rs, nr) = Self::rots_unique(c.x as i32, c.y as i32, t as usize);
                for k in 0..nr {
                    let (r, sh) = &rs[k];
                    if shape_eq(sh, &c.sh) { continue; }
                    let mut ok = true;
                    for e in 0..3 {
                        let (nid, ne) = self.nb(sl as usize, e);
                        if let Some(n) = nid { if self.cells[n].open(ne) != sh.open[e] { ok = false; break; } }
                    }
                    if !ok { continue; }
                    acts.push(Act::a3(A_OVERLAY, sl as usize, t as usize, *r as usize));
                    if first_only { return; }
                }
            }
        }
    }

    pub fn legal_actions(&self) -> Vec<Act> {
        if self.over { return Vec::new(); }
        if self.pending.is_some() { return self.pending_actions(); }
        let pi = self.cur;
        let p = &self.players[pi];
        if self.actions_this_turn >= MAX_ACTIONS { return vec![Act::new(A_END)]; }
        let mut acts: Vec<Act> = Vec::with_capacity(96);
        let u = &self.used;

        if !u[0] {
            if !self.tile_stack.is_empty() { acts.push(Act::new(A_INC_TILES)); }
            if !self.card_stack.is_empty() { acts.push(Act::new(A_INC_CARD)); }
            for h in self.hill_cells(pi).iter() { acts.push(Act::a1(A_INC_HILL, *h as usize)); }
        }
        if !u[1] && !p.hand.is_empty() {
            for q in 0..self.n {
                if q == pi || self.players[q].hand.is_empty() { continue; }
                let mut cnt = 0;
                for &give in Self::uniq(p.hand.as_slice()).iter() {
                    for &take in Self::uniq(self.players[q].hand.as_slice()).iter() {
                        if give == take { continue; }
                        let c = cnt; cnt += 1;
                        if c >= 10 { continue; }
                        acts.push(Act::a3(A_SWAP, q, give as usize, take as usize));
                    }
                }
            }
        }
        self.placement_actions(pi, &mut acts);

        let node = self.ant_node(pi);
        let cs_slot = if node >= 0 { (node / 3) as usize } else { 0 };
        if node >= 0 {
            let cs = &self.cells[cs_slot];
            let starts = self.ant_starts(pi);
            let bf = self.bfs(pi, starts.as_slice(), 999, true);
            if p.carry >= 0 {
                // house rule: carrying ants move only via transport
            } else {
                let interesting = self.interesting_nodes(pi);
                let mut is_int = [false; 192];
                for &n in interesting.iter() { is_int[n as usize] = true; }
                let dests: Vec<u8> = bf.order.iter().filter(|(n, d)| *d > 0 && !self.move_trail.iter().any(|t| t == n)).map(|(n, _)| *n).collect();
                let mut chosen: Vec<u8> = Vec::with_capacity(16);
                let mut mark = [false; 192];
                for &n in &dests { if is_int[n as usize] && !mark[n as usize] { mark[n as usize] = true; chosen.push(n); } }
                let stride = std::cmp::max(1, dests.len() / 6);
                let mut i = 0;
                while i < dests.len() { let n = dests[i]; if !mark[n as usize] { mark[n as usize] = true; chosen.push(n); } i += stride; }
                for n in chosen { acts.push(Act::a1(A_MOVE, n as usize)); }
            }
            if p.carry < 0 && cs.res >= 0 && self.last_t != A_UNLOAD { acts.push(Act::new(A_LOAD)); }
            if p.carry >= 0 && cs.res < 0 && self.last_t != A_LOAD { acts.push(Act::new(A_UNLOAD)); }
            if !u[2] && p.carry < 0 {
                for q in 0..self.n {
                    if q != pi {
                        if let Some(qa) = &self.players[q].ant {
                            if qa.cell as usize == cs_slot && self.players[q].carry >= 0 { acts.push(Act::a1(A_FIGHT, q)); }
                        }
                    }
                }
            }
            if !u[3] && p.carry >= 0 && self.resources[p.carry as usize].laid != self.turn_no {
                acts.push(Act::a1(A_TRANSPORT, 0));
                if p.has_card(C_TRANSPORT) { acts.push(Act::a1(A_TRANSPORT, 1)); }
            }
        }

        // ant cards
        if p.has_card(C_BRIDGE) && p.bridges > 0 {
            let mut reach_cells = [false; 64];
            if node >= 0 {
                let starts = self.ant_starts(pi);
                let reach = self.bfs(pi, starts.as_slice(), 999, true);
                for (n, _) in reach.order.iter() { reach_cells[*n as usize / 3] = true; }
            }
            let mut cnt = 0;
            for &sl in self.order.iter() {
                let c = &self.cells[sl as usize];
                for e in 0..3 {
                    let (nid, ne) = self.nb(sl as usize, e);
                    let Some(nid) = nid else { continue };
                    if self.cells[nid].gid < c.gid { continue; }
                    let nc = &self.cells[nid];
                    if c.open(e) == 1 && nc.open(ne) == 1 { continue; }
                    if c.block >= 0 || nc.block >= 0 { continue; }
                    if self.bridges.iter().any(|b| (b.a == sl && b.b as usize == nid) || (b.a as usize == nid && b.b == sl)) { continue; }
                    if reach_cells[sl as usize] == reach_cells[nid] { continue; }
                    let cc = cnt; cnt += 1;
                    if cc >= 12 { continue; }
                    acts.push(Act::a3(A_BRIDGE, sl as usize, nid, 0));
                }
            }
        }
        if p.has_card(C_BLOCKAGE) && p.blockages > 0 {
            let mut to_block: Vec<u8> = Vec::with_capacity(8);
            for q in 0..self.n {
                if q == pi { continue; }
                if self.players[q].ant.is_none() { continue; }
                let hs = self.hill_cells(q);
                if hs.is_empty() { continue; }
                let hstarts: Vec<u8> = hs.iter().map(|h| h * 3).collect();
                let hd = self.bfs(q, &hstarts, 999, false);
                let mut n = self.ant_node(q);
                let mut guard = 0;
                while n >= 0 && hd.dist[n as usize] > 0 && { let g = guard; guard += 1; g < 20 } {
                    let mut best = -1i32; let mut bd = 1_000_000_000i32;
                    self.expand(q, n as usize, false, &mut |m| {
                        let dm = hd.dist[m];
                        if dm >= 0 && (dm as i32) < bd { bd = dm as i32; best = m as i32; }
                    });
                    if best < 0 { break; }
                    let cell = (best / 3) as u8;
                    if !to_block.contains(&cell) { to_block.push(cell); }
                    n = best;
                }
            }
            for id in to_block {
                let c = &self.cells[id as usize];
                if c.block >= 0 { continue; }
                if (0..self.n).any(|q| self.players[q].ant.map_or(false, |a| a.cell == id)) { continue; }
                acts.push(Act::a1(A_BLOCKAGE, id as usize));
            }
        }
        if node >= 0 && !(p.carry >= 0 && self.cells[cs_slot].res >= 0) {
            let has_wings = p.has_card(C_WINGS);
            let mut via: Option<(u8, u8)> = None;
            if !has_wings && p.cards.len() >= 2 {
                let mut sorted: Vec<u8> = p.cards.iter().copied().collect();
                sorted.sort_by_key(|c| DROP_RANK[*c as usize]);
                via = Some((sorted[0], sorted[1]));
            }
            if has_wings || via.is_some() {
                for &n in self.interesting_nodes(pi).iter() {
                    if n as i32 == node { continue; }
                    if self.cells[n as usize / 3].block >= 0 { continue; }
                    let (v0, v1) = if has_wings { (255, 255) } else { via.unwrap() };
                    acts.push(Act { t: A_WINGS, a: n, b: v0, c: v1, g: 0 });
                }
            }
        }
        if p.has_card(C_SUPER) && !p.hand.is_empty() {
            let mut tmp = Vec::new();
            self.overlay_actions(pi, &mut tmp, true);
            if !tmp.is_empty() { acts.push(Act::new(A_SUPER)); }
        }
        acts.push(Act::new(A_END));
        acts
    }

    fn pending_actions(&self) -> Vec<Act> {
        let pd = self.pending.as_ref().unwrap();
        let pi = pd.actor as usize;
        match pd.kind {
            P_HIDE => vec![Act::a1(A_HIDE, 0), Act::a1(A_HIDE, 1)],
            P_GUESS => vec![Act::a1(A_GUESS, 0), Act::a1(A_GUESS, 1)],
            P_REVENGE => vec![Act::new(A_REVENGE), Act::new(A_DECLINE)],
            P_TMOVE => {
                let node = self.ant_node(pi);
                let starts = self.ant_starts(pi);
                let dist = self.bfs(pi, starts.as_slice(), pd.n as i32, false);
                let mut acts = vec![Act::new(A_TSTAY)];
                let hs: Vec<u8> = self.hill_cells(pi).iter().map(|h| h * 3).collect();
                let hd = self.bfs(pi, &hs, 999, false);
                let cands: Vec<u8> = dist.order.iter().filter(|(_, d)| *d > 0).map(|(n, _)| *n).collect();
                if let Some(&hn) = cands.iter().find(|&&n| self.cells[n as usize / 3].hill == pi as i8) {
                    acts.push(Act::a1(A_TMOVE, hn as usize));
                    return acts;
                }
                let mut sorted: Vec<u8> = cands.iter().copied().filter(|&n| hd.dist[n as usize] >= 0).collect();
                sorted.sort_by_key(|&n| hd.dist[n as usize]);
                let cur_d = if node >= 0 && hd.dist[node as usize] >= 0 { hd.dist[node as usize] as i32 } else { 1_000_000_000 };
                for &n in sorted.iter().take(2) { if (hd.dist[n as usize] as i32) < cur_d { acts.push(Act::a1(A_TMOVE, n as usize)); } }
                acts
            }
            P_SUPER => {
                let mut acts = Vec::new();
                self.overlay_actions(pi, &mut acts, false);
                if self.consistent() { acts.push(Act::new(A_SDONE)); }
                if acts.is_empty() { acts.push(Act::new(A_SUNDO)); }
                acts
            }
            _ => Vec::new(),
        }
    }

    // ---------- apply ----------
    fn move_ant(&mut self, pi: usize, node: usize) {
        let cid = node / 3;
        let comp = self.cells[cid].comp(node - cid * 3);
        let edge = if comp != 0 { comp.trailing_zeros() as u8 } else { 0 };
        self.players[pi].ant = Some(Ant { cell: cid as u8, edge, free: false });
    }
    fn free_ants_at(&mut self, cell: usize) {
        for q in 0..self.n { if let Some(a) = &mut self.players[q].ant { if a.cell as usize == cell { a.free = true; } } }
    }
    fn deliver(&mut self, pi: usize) {
        let ri = self.players[pi].carry as usize;
        let r = &mut self.resources[ri];
        r.owner = pi as i8; r.carrier = -1; r.cell = -1;
        self.players[pi].carry = -1;
        self.check_win();
    }
    fn finalize_fight(&mut self) {
        let pd = self.pending.unwrap();
        if pd.att_win {
            self.players[pd.def as usize].carry = -1;
            self.players[pd.att as usize].carry = pd.res;
            self.resources[pd.res as usize].carrier = pd.att as i8;
        }
        self.pending = None;
    }

    pub fn apply(&mut self, a: Act) {
        if self.over { return; }
        let pi = self.actor();
        if a.t == A_MOVE {
            if self.move_trail.is_empty() { let n = self.ant_node(pi); if n >= 0 { self.move_trail.push(n as u8); } }
        } else if a.t != A_END { self.move_trail = FV::new(); }
        match a.t {
            A_INC_TILES => {
                for _ in 0..4 { if let Some(t) = self.tile_stack.pop() { self.players[pi].hand.push(t); } }
                self.used[0] = true;
            }
            A_INC_CARD => { let c = self.card_stack.pop().unwrap(); self.players[pi].cards.push(c); self.used[0] = true; }
            A_INC_HILL => {
                let cell = a.a;
                let i = self.hills.iter().position(|h| h.0 as usize == pi && h.1 == cell).unwrap();
                self.hills.remove(i);
                self.cells[cell as usize].hill = -1;
                self.free_ants_at(cell as usize);
                self.players[pi].hills_in_hand += 1;
                self.used[0] = true;
            }
            A_SWAP => {
                let q = a.a as usize;
                let i = self.players[pi].hand.index_of(a.b).unwrap(); self.players[pi].hand.remove(i);
                let j = self.players[q].hand.index_of(a.c).unwrap(); self.players[q].hand.remove(j);
                self.players[pi].hand.push(a.c); self.players[q].hand.push(a.b);
                self.used[1] = true;
            }
            A_PLACE => {
                let i = self.players[pi].hand.index_of(a.a).unwrap(); self.players[pi].hand.remove(i);
                let (x, y) = (gx(a.g as i32), gy(a.g as i32));
                let slot = self.put_cell(x, y, a.a as usize, a.b as usize);
                if a.c == 1 {
                    self.players[pi].hills_in_hand -= 1;
                    self.hills.push((pi as u8, slot as u8));
                    self.cells[slot].hill = pi as i8;
                    if self.players[pi].ant.is_none() {
                        let ao = self.cells[slot].sh.all_open;
                        self.players[pi].ant = Some(Ant { cell: slot as u8, edge: ao.trailing_zeros() as u8, free: false });
                    }
                } else if a.c == 2 {
                    let kind = self.res_stack.pop().unwrap();
                    self.resources.push(Res { kind, cell: slot as i8, carrier: -1, owner: -1, laid: self.turn_no });
                    self.cells[slot].res = self.resources.len() as i8 - 1;
                }
            }
            A_MOVE => { self.move_ant(pi, a.a as usize); if self.move_trail.len() < 80 { self.move_trail.push(a.a); } }
            A_LOAD => {
                let cell = self.players[pi].ant.unwrap().cell as usize;
                let ri = self.cells[cell].res;
                let r = &mut self.resources[ri as usize];
                r.carrier = pi as i8; r.cell = -1;
                self.cells[cell].res = -1;
                self.players[pi].carry = ri;
                self.free_ants_at(cell);
            }
            A_UNLOAD => {
                let cell = self.players[pi].ant.unwrap().cell as usize;
                let ri = self.players[pi].carry;
                let r = &mut self.resources[ri as usize];
                r.carrier = -1; r.cell = cell as i8;
                self.cells[cell].res = ri;
                self.players[pi].carry = -1;
                self.free_ants_at(cell);
            }
            A_FIGHT => {
                self.used[2] = true;
                let q = a.a as usize;
                self.pending = Some(Pending { kind: P_HIDE, actor: q as u8, att: pi as u8, def: q as u8, res: self.players[q].carry, hide: -1, ..Default::default() });
            }
            A_HIDE => { let pd = self.pending.as_mut().unwrap(); pd.hide = a.a as i8; pd.kind = P_GUESS; pd.actor = pd.att; }
            A_GUESS => {
                let loser = {
                    let pd = self.pending.as_mut().unwrap();
                    pd.att_win = a.a as i8 == pd.hide;
                    pd.guess_v = a.a as i8;
                    if pd.att_win { pd.def } else { pd.att }
                };
                if self.players[loser as usize].has_card(C_REVENGE) {
                    let pd = self.pending.as_mut().unwrap();
                    pd.kind = P_REVENGE; pd.actor = loser;
                } else { self.finalize_fight(); }
            }
            A_REVENGE => {
                let actor = self.pending.as_ref().unwrap().actor as usize;
                self.removed_card(actor, C_REVENGE);
                let pd = self.pending.as_mut().unwrap();
                pd.kind = P_HIDE; pd.actor = pd.def; pd.hide = -1;
            }
            A_DECLINE => self.finalize_fight(),
            A_TRANSPORT => {
                if a.a == 1 { self.removed_card(pi, C_TRANSPORT); }
                let d1 = rint(self, 6);
                let d2 = if a.a == 1 { rint(self, 6) } else { 0 };
                self.used[3] = true;
                let cell = self.players[pi].ant.unwrap().cell as usize;
                if self.cells[cell].hill == pi as i8 { self.deliver(pi); }
                else if d1 + d2 > 0 { self.pending = Some(Pending { kind: P_TMOVE, actor: pi as u8, n: (d1 + d2) as u8, res: -1, hide: -1, ..Default::default() }); }
            }
            A_TMOVE => {
                self.move_ant(pi, a.a as usize);
                if self.cells[a.a as usize / 3].hill == pi as i8 { self.deliver(pi); }
                self.pending = None;
            }
            A_TSTAY => self.pending = None,
            A_BRIDGE => {
                self.removed_card(pi, C_BRIDGE);
                self.players[pi].bridges -= 1;
                let placed = self.players[pi].own_turns;
                self.bridges.push(Bridge { owner: pi as u8, a: a.a, b: a.b, placed });
                self.cells[a.a as usize].bridge += 1; self.cells[a.b as usize].bridge += 1;
            }
            A_BLOCKAGE => {
                self.removed_card(pi, C_BLOCKAGE);
                self.players[pi].blockages -= 1;
                self.blocks.push((pi as u8, a.a));
                self.cells[a.a as usize].block = pi as i8;
            }
            A_WINGS => {
                if a.b != 255 { self.removed_card(pi, a.b); self.removed_card(pi, a.c); } else { self.removed_card(pi, C_WINGS); }
                if self.players[pi].carry >= 0 {
                    let cell = self.players[pi].ant.unwrap().cell as usize;
                    let ri = self.players[pi].carry;
                    let r = &mut self.resources[ri as usize];
                    r.carrier = -1; r.cell = cell as i8;
                    self.cells[cell].res = ri;
                    self.players[pi].carry = -1;
                }
                self.move_ant(pi, a.a as usize);
            }
            A_SUPER => {
                self.removed_card(pi, C_SUPER);
                self.pending = Some(Pending { kind: P_SUPER, actor: pi as u8, res: -1, hide: -1, ..Default::default() });
            }
            A_OVERLAY => {
                let slot = a.a as usize;
                let c = self.cells[slot];
                self.pending.as_mut().unwrap().overlays.push(Ov { cell: slot as u8, tile: c.tile, rot: c.rot, new_tile: a.b });
                let i = self.players[pi].hand.index_of(a.b).unwrap(); self.players[pi].hand.remove(i);
                let sh = make_shape(c.x as i32, c.y as i32, a.b as usize, a.c as usize);
                let cm = &mut self.cells[slot];
                cm.sh = sh; cm.tile = a.b; cm.rot = a.c; cm.overlaid = true;
            }
            A_SUNDO => {
                let ov = self.pending.as_ref().unwrap().overlays;
                for i in (0..ov.len()).rev() {
                    let o = ov[i];
                    let cm = &mut self.cells[o.cell as usize];
                    cm.sh = make_shape(cm.x as i32, cm.y as i32, o.tile as usize, o.rot as usize);
                    cm.tile = o.tile; cm.rot = o.rot; cm.overlaid = false;
                    self.players[pi].hand.push(o.new_tile);
                }
                self.pending = None;
            }
            A_SDONE => self.pending = None,
            A_END => { self.end_turn(); return; }
            _ => {}
        }
        self.last_t = a.t;
        if self.pending.map_or(true, |p| p.kind == P_SUPER) { self.actions_this_turn += 1; }
    }
}

pub struct Bfs { pub dist: [i16; 192], pub order: FV<(u8, i16), 192> }

pub fn new_game(num_players: usize, seed: i32) -> State {
    let mut s = State {
        seed, n: num_players, players: [Player::default(); 6],
        grid: [255u8; (G * G) as usize], cells: [Cell::default(); 64], ncells: 0, order: FV::new(),
        tile_stack: FV::new(), res_stack: FV::new(), card_stack: FV::new(),
        resources: FV::new(), hills: FV::new(), blocks: FV::new(), bridges: FV::new(),
        cur: 0, turn_no: 0, used: [false; 4], actions_this_turn: 0, pending: None, winner: -1, over: false,
        move_trail: FV::new(), last_t: 255,
    };
    let mut buf = [0u8; 56];
    let mut k = 0;
    for (i, &c) in TILE_COUNT.iter().enumerate() { for _ in 0..c { buf[k] = i as u8; k += 1; } }
    shuffle_u8(&mut s, &mut buf[..k]);
    for t in &buf[..k] { s.tile_stack.push(*t); }
    let mut rb = [0u8; 12]; let mut k = 0;
    for kind in 0..3 { for _ in 0..4 { rb[k] = kind; k += 1; } }
    shuffle_u8(&mut s, &mut rb);
    for t in &rb { s.res_stack.push(*t); }
    let mut cb = [0u8; 18]; let mut k = 0;
    for c in 0..6 { for _ in 0..3 { cb[k] = c; k += 1; } }
    shuffle_u8(&mut s, &mut cb);
    for t in &cb { s.card_stack.push(*t); }
    for i in 0..num_players {
        let mut values = [1u8, 2, 3];
        shuffle_u8(&mut s, &mut values);
        let mut p = Player { hills_in_hand: 2, bridges: 2, blockages: 2, carry: -1, values, ..Default::default() };
        p.hand.push(s.tile_stack.pop().unwrap());
        s.players[i] = p;
    }
    let t = s.tile_stack.pop().unwrap() as usize;
    let r = rint(&mut s, 3);
    s.put_cell(0, 0, t, r);
    s.cur = rint(&mut s, num_players);
    s.start_turn();
    s
}
