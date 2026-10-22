// Amino game engine. Pure JSON-serialisable state, no DOM.
// A turn is a sequence of atomic actions; `actor(state)` says who must decide next.

import { cellId, cellX, cellY, isUp, neighbor, neighborId, OPP, edgePos, gridDistWithin } from './geometry.js';

export const TILE_TYPES = [
  { name: 'Y', open: [1, 1, 1], groups: [[0, 1, 2]], count: 8 },
  { name: 'Kurve', open: [1, 1, 0], groups: [[0, 1]], count: 12 },
  { name: 'Sackgasse', open: [1, 0, 0], groups: [[0]], count: 6 },
  { name: 'Gabel', open: [1, 1, 1], groups: [[0, 1], [2]], count: 10 },
  { name: 'Stummel2', open: [1, 1, 0], groups: [[0], [1]], count: 8 },
  { name: 'Stummel3', open: [1, 1, 1], groups: [[0], [1], [2]], count: 6 },
];
export const CARD_TYPES = ['bridge', 'superimpose', 'blockage', 'wings', 'revenge', 'transport'];
export const KINDS = ['Blume', 'Pilz', 'Raupe'];
export const WIN_POINTS = 7;
export const MAX_ACTIONS = 70;
export const MAX_ROUNDS = 90;
const CARD_DROP_ORDER = ['blockage', 'bridge', 'superimpose', 'transport', 'revenge', 'wings'];

// ---------- rng (state-embedded mulberry32) ----------
export function rnd(s) {
  s.seed = (s.seed + 0x6d2b79f5) | 0;
  let t = Math.imul(s.seed ^ (s.seed >>> 15), 1 | s.seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
const rint = (s, n) => Math.floor(rnd(s) * n);
function shuffle(s, a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = rint(s, i + 1);
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// ---------- tiles / cells ----------
export function makeCellShape(x, y, tile, rot) {
  const T = TILE_TYPES[tile];
  const open = [0, 0, 0];
  const slot = [0, 0, 0];
  for (let e = 0; e < 3; e++) {
    const t = (edgePos(x, y, e) + rot) % 3;
    slot[e] = t;
    open[e] = T.open[t];
  }
  const comps = [];
  for (const g of T.groups) {
    const edges = [];
    for (let e = 0; e < 3; e++) if (g.includes(slot[e]) && open[e]) edges.push(e);
    if (edges.length) comps.push(edges);
  }
  const allOpen = [];
  for (let e = 0; e < 3; e++) if (open[e]) allOpen.push(e);
  return { tile, rot, open, comps, allOpen };
}
const shapeSig = (sh) => sh.open.join('') + '|' + sh.comps.map((c) => c.join('')).sort().join(',');

function putCell(state, x, y, tile, rot) {
  const sh = makeCellShape(x, y, tile, rot);
  const id = cellId(x, y);
  const c = { id, x, y, ...sh, hill: -1, res: -1, block: -1, bridge: 0, overlaid: false };
  state.cells[id] = c;
  return c;
}

export function compsOf(c) {
  return c.hill >= 0 || c.res >= 0 ? [c.allOpen] : c.comps;
}
function compIndex(c, edge) {
  const cs = compsOf(c);
  for (let i = 0; i < cs.length; i++) if (cs[i].includes(edge)) return i;
  return 0;
}

// ---------- setup ----------
export const PLAYER_COLORS = ['#d64545', '#3b82d6', '#e0a800', '#2f9e5b', '#8e55c9', '#e07a2f'];
export const PLAYER_COLOR_NAMES = ['Rot', 'Blau', 'Gelb', 'Grün', 'Lila', 'Orange'];

export function newGame({ numPlayers = 2, seed = (Math.random() * 2 ** 31) | 0, names } = {}) {
  const s = {
    seed: seed | 0, n: numPlayers, players: [], cells: {}, tileStack: [], resStack: [], cardStack: [],
    resources: [], hills: [], blocks: [], bridges: [], cur: 0, turnNo: 0, used: { income: 0, swap: 0, fight: 0, transport: 0 },
    actionsThisTurn: 0, pending: null, winner: -1, over: false, log: [], lastRoll: null,
  };
  TILE_TYPES.forEach((T, i) => { for (let k = 0; k < T.count; k++) s.tileStack.push(i); });
  shuffle(s, s.tileStack);
  for (let k = 0; k < 3; k++) for (let i = 0; i < 4; i++) s.resStack.push(k);
  shuffle(s, s.resStack);
  for (const c of CARD_TYPES) for (let i = 0; i < 3; i++) s.cardStack.push(c);
  shuffle(s, s.cardStack);
  for (let i = 0; i < numPlayers; i++) {
    const values = shuffle(s, [1, 2, 3]);
    s.players.push({
      idx: i, name: names?.[i] || PLAYER_COLOR_NAMES[i], color: PLAYER_COLORS[i], hand: [s.tileStack.pop()], cards: [],
      hillsInHand: 2, bridges: 2, blockages: 2, ant: null, carry: -1, values, ownTurns: 0,
    });
  }
  putCell(s, 0, 0, s.tileStack.pop(), rint(s, 3));
  s.cur = rint(s, numPlayers); // "highest roll starts"
  startTurn(s);
  return s;
}

export const cloneState = (s) => JSON.parse(JSON.stringify(s));

function startTurn(s) {
  const p = s.players[s.cur];
  s.turnNo++;
  p.ownTurns++;
  for (const b of s.blocks) if (b.owner === s.cur) s.cells[b.cell].block = -1;
  p.blockages += s.blocks.filter((b) => b.owner === s.cur).length;
  s.blocks = s.blocks.filter((b) => b.owner !== s.cur);
  s.used = { income: 0, swap: 0, fight: 0, transport: 0 };
  s.actionsThisTurn = 0;
}

function endTurn(s) {
  const p = s.players[s.cur];
  const keep = [];
  for (const b of s.bridges) {
    if (b.owner === s.cur && b.placedOwnTurn < p.ownTurns) {
      s.cells[b.a].bridge--; s.cells[b.b].bridge--; p.bridges++;
    } else keep.push(b);
  }
  s.bridges = keep;
  s.cur = (s.cur + 1) % s.n;
  if (Math.floor(s.turnNo / s.n) >= MAX_ROUNDS) return finishByScore(s);
  startTurn(s);
}

// ---------- scoring ----------
export function score(s, pi) {
  let t = 0;
  const v = s.players[pi].values;
  for (const r of s.resources) if (r.owner === pi) t += v[r.kind];
  return t;
}
function finishByScore(s) {
  let best = -1, bi = -1, tie = false;
  for (let i = 0; i < s.n; i++) {
    const sc = score(s, i);
    if (sc > best) { best = sc; bi = i; tie = false; } else if (sc === best) tie = true;
  }
  s.over = true;
  s.winner = tie ? -1 : bi;
}
function checkWin(s) {
  for (let i = 0; i < s.n; i++) if (score(s, i) >= WIN_POINTS) { s.over = true; s.winner = i; return; }
  if (s.resources.filter((r) => r.owner >= 0).length >= 12) finishByScore(s);
}

// ---------- path graph ----------
export function antNode(s, pi) {
  const a = s.players[pi].ant;
  if (!a) return -1;
  return a.cell * 3 + compIndex(s.cells[a.cell], a.edge);
}
export const nodeCell = (n) => Math.floor(n / 3);

// Start nodes for path searches. If the cell's paths changed under the ant (resource loaded, hill removed),
// the ant may leave through any of the cell's path groups.
export function antStarts(s, pi) {
  const a = s.players[pi].ant;
  if (!a) return [];
  if (a.free) return nodesOfCell(s, a.cell);
  return [antNode(s, pi)];
}

function nodesOfCell(s, cid) {
  const c = s.cells[cid];
  const out = [];
  if (!c) return out;
  const k = compsOf(c).length;
  for (let i = 0; i < k; i++) out.push(cid * 3 + i);
  return out;
}

function expand(s, pi, node, opts, out) {
  const cid = nodeCell(node), ci = node - cid * 3;
  const c = s.cells[cid];
  const comp = compsOf(c)[ci] || [];
  for (const e of comp) {
    const [nid, ne] = neighborId(cid, e);
    const nc = s.cells[nid];
    if (nc && nc.block < 0 && nc.open[ne]) out.push(nid * 3 + compIndex(nc, ne));
  }
  if (opts.bridges) {
    for (const b of s.bridges) {
      if (b.owner !== pi) continue;
      let other = -1;
      if (b.a === cid) other = b.b; else if (b.b === cid) other = b.a;
      if (other >= 0 && s.cells[other].block < 0) for (const n of nodesOfCell(s, other)) out.push(n);
    }
  }
  if (opts.underground && c.hill === pi) {
    for (const h of s.hills) if (h.owner === pi && h.cell !== cid && s.cells[h.cell].block < 0) out.push(h.cell * 3);
  }
}

// BFS: Map node -> distance
export function bfs(s, pi, starts, { maxDepth = 999, underground = true, bridges = true } = {}) {
  const dist = new Map();
  let frontier = [];
  for (const st of starts) if (!dist.has(st)) { dist.set(st, 0); frontier.push(st); }
  const opts = { underground, bridges };
  for (let d = 1; d <= maxDepth && frontier.length; d++) {
    const next = [];
    for (const n of frontier) {
      const out = [];
      expand(s, pi, n, opts, out);
      for (const m of out) if (!dist.has(m)) { dist.set(m, d); next.push(m); }
    }
    frontier = next;
  }
  return dist;
}

export function hillCells(s, pi) {
  return s.hills.filter((h) => h.owner === pi).map((h) => h.cell);
}

// ---------- placement helpers ----------
function frontierCells(s) {
  const seen = new Set();
  const out = [];
  for (const k in s.cells) {
    const c = s.cells[k];
    for (let e = 0; e < 3; e++) {
      const [nx, ny] = neighbor(c.x, c.y, e);
      const id = cellId(nx, ny);
      if (!s.cells[id] && !seen.has(id)) { seen.add(id); out.push(id); }
    }
  }
  return out;
}

function placementOk(s, id, sh) {
  const x = cellX(id), y = cellY(id);
  let connections = 0;
  for (let e = 0; e < 3; e++) {
    const [nx, ny, ne] = neighbor(x, y, e);
    const nc = s.cells[cellId(nx, ny)];
    if (!nc) continue;
    if (nc.open[ne] !== sh.open[e]) return false;
    if (sh.open[e]) connections++;
  }
  return connections > 0;
}

function hasOwnHillNear(s, id, pi) {
  return gridDistWithin(id, 2, (c) => { const cc = s.cells[c]; return !!cc && cc.hill === pi; });
}
// only resources lying on the board count for the distance rule
function resOnBoardNear(s, id) {
  return gridDistWithin(id, 2, (c) => { const cc = s.cells[c]; return !!cc && cc.res >= 0; });
}

function uniq(a) { return [...new Set(a)]; }

export function hasCard(p, t) { return p.cards.includes(t); }
function removeCard(s, p, t) {
  const i = p.cards.indexOf(t);
  p.cards.splice(i, 1);
  s.cardStack.unshift(t); // under the pile
}

function hasPieceOrAnt(s, c) {
  if (c.hill >= 0 || c.res >= 0 || c.block >= 0 || c.bridge > 0) return true;
  for (const q of s.players) if (q.ant && q.ant.cell === c.id) return true;
  return false;
}

// ---------- legal actions ----------
export function actor(s) {
  return s.pending ? s.pending.actor : s.cur;
}

export function consistent(s) {
  for (const k in s.cells) {
    const c = s.cells[k];
    for (let e = 0; e < 3; e++) {
      const [nx, ny, ne] = neighbor(c.x, c.y, e);
      const nc = s.cells[cellId(nx, ny)];
      if (nc && nc.open[ne] !== c.open[e]) return false;
    }
  }
  return true;
}

function rotsUnique(x, y, tile) {
  const seen = new Set(), out = [];
  for (let r = 0; r < 3; r++) {
    const sh = makeCellShape(x, y, tile, r);
    const sg = shapeSig(sh);
    if (!seen.has(sg)) { seen.add(sg); out.push([r, sh]); }
  }
  return out;
}

export function placementActions(s, pi) {
  const p = s.players[pi];
  const acts = [];
  const tiles = uniq(p.hand);
  if (!tiles.length) return acts;
  for (const id of frontierCells(s)) {
    const x = cellX(id), y = cellY(id);
    let hillOk = null, resOk = null;
    for (const t of tiles) {
      for (const [r, sh] of rotsUnique(x, y, t)) {
        if (!placementOk(s, id, sh)) continue;
        acts.push({ t: 'place', cell: id, tile: t, rot: r, piece: 0 });
        if (p.hillsInHand > 0) {
          if (hillOk === null) hillOk = !resOnBoardNear(s, id);
          if (hillOk) acts.push({ t: 'place', cell: id, tile: t, rot: r, piece: 1 });
        }
        if (s.resStack.length > 0) {
          if (resOk === null) resOk = !hasOwnHillNear(s, id, pi);
          if (resOk) acts.push({ t: 'place', cell: id, tile: t, rot: r, piece: 2 });
        }
      }
    }
  }
  return acts;
}

function overlayable(s, c) {
  return !c.overlaid && !hasPieceOrAnt(s, c);
}

function overlayActions(s, pi, full) {
  const p = s.players[pi];
  const acts = [];
  const tiles = uniq(p.hand);
  for (const k in s.cells) {
    const c = s.cells[k];
    if (!overlayable(s, c)) continue;
    for (const t of tiles) {
      for (const [r, sh] of rotsUnique(c.x, c.y, t)) {
        if (sh.open.join('') === c.open.join('') && shapeSig(sh) === shapeSig(c)) continue;
        if (!full) {
          // local consistency only
          let ok = true;
          for (let e = 0; e < 3 && ok; e++) {
            const [nx, ny, ne] = neighbor(c.x, c.y, e);
            const nc = s.cells[cellId(nx, ny)];
            if (nc && nc.open[ne] !== sh.open[e]) ok = false;
          }
          if (!ok) continue;
        }
        acts.push({ t: 'overlay', cell: c.id, tile: t, rot: r });
      }
    }
  }
  return acts;
}

export function interestingNodes(s, pi) {
  // nodes that matter for AI: free resources, enemy carriers, own hills
  const out = new Set();
  for (const r of s.resources) {
    if (r.owner < 0 && r.carrier < 0 && r.cell >= 0) for (const n of nodesOfCell(s, r.cell)) out.add(n);
  }
  for (let q = 0; q < s.n; q++) {
    if (q === pi) continue;
    const a = s.players[q].ant;
    if (a && s.players[q].carry >= 0) for (const n of nodesOfCell(s, a.cell)) out.add(n);
  }
  for (const h of hillCells(s, pi)) out.add(h * 3);
  return out;
}

export function legalActions(s, full = false) {
  if (s.over) return [];
  const pd = s.pending;
  if (pd) return pendingActions(s, full);
  const pi = s.cur;
  const p = s.players[pi];
  if (s.actionsThisTurn >= MAX_ACTIONS) return [{ t: 'end' }];
  const acts = [];
  const u = s.used;

  if (!u.income) {
    if (s.tileStack.length) acts.push({ t: 'inc_tiles' });
    if (s.cardStack.length) acts.push({ t: 'inc_card' });
    for (const h of hillCells(s, pi)) acts.push({ t: 'inc_hill', cell: h });
  }
  if (!u.swap && p.hand.length) {
    for (let q = 0; q < s.n; q++) {
      if (q === pi || !s.players[q].hand.length) continue;
      let cnt = 0;
      for (const give of uniq(p.hand)) for (const take of uniq(s.players[q].hand)) {
        if (!full && give === take) continue;
        if (!full && cnt++ >= 10) continue;
        acts.push({ t: 'swap', q, give, take });
      }
    }
  }
  for (const a of placementActions(s, pi)) acts.push(a);

  const node = antNode(s, pi);
  const cs = node >= 0 ? s.cells[nodeCell(node)] : null;
  if (node >= 0) {
    const dist = bfs(s, pi, antStarts(s, pi));
    const interesting = interestingNodes(s, pi);
    const dests = [];
    for (const [n, d] of dist) if (d > 0) dests.push([n, d]);
    if (p.carry >= 0) {
      // house rule: with a resource the ant moves ONLY via the transport action (dice), never by a plain move
    } else if (full) {
      for (const [n] of dests) acts.push({ t: 'move', to: n });
    } else {
      const chosen = new Set();
      for (const [n] of dests) if (interesting.has(n)) chosen.add(n);
      const stride = Math.max(1, Math.floor(dests.length / 6));
      for (let i = 0; i < dests.length; i += stride) chosen.add(dests[i][0]);
      for (const n of chosen) acts.push({ t: 'move', to: n });
    }
    if (p.carry < 0 && cs && cs.res >= 0) acts.push({ t: 'load' });
    if (p.carry >= 0 && cs.res < 0) acts.push({ t: 'unload' });
    if (!u.fight && p.carry < 0) {
      for (let q = 0; q < s.n; q++) {
        const qa = s.players[q].ant;
        if (q !== pi && qa && qa.cell === cs.id && s.players[q].carry >= 0) acts.push({ t: 'fight', q });
      }
    }
    if (!u.transport && p.carry >= 0 && s.resources[p.carry].laidTurn !== s.turnNo) {
      acts.push({ t: 'transport', card: 0 });
      if (hasCard(p, 'transport')) acts.push({ t: 'transport', card: 1 });
    }
  }

  // ant cards
  if (hasCard(p, 'bridge') && p.bridges > 0) {
    const reach = node >= 0 ? bfs(s, pi, antStarts(s, pi)) : new Map();
    const reachCells = new Set([...reach.keys()].map(nodeCell));
    let cnt = 0;
    for (const k in s.cells) {
      const c = s.cells[k];
      for (let e = 0; e < 3; e++) {
        const [nid, ne] = neighborId(c.id, e);
        if (nid < c.id) continue;
        const nc = s.cells[nid];
        if (!nc || (c.open[e] && nc.open[ne])) continue; // already connected
        if (c.block >= 0 || nc.block >= 0) continue;
        if (s.bridges.some((b) => (b.a === c.id && b.b === nid) || (b.a === nid && b.b === c.id))) continue;
        if (!full) {
          if (reachCells.has(c.id) === reachCells.has(nid)) continue;
          if (cnt++ >= 12) continue;
        }
        acts.push({ t: 'bridge', a: c.id, b: nid });
      }
    }
  }
  if (hasCard(p, 'blockage') && p.blockages > 0) {
    const cellsToBlock = new Set();
    if (full) {
      for (const k in s.cells) cellsToBlock.add(+k);
    } else {
      for (let q = 0; q < s.n; q++) {
        if (q === pi) continue;
        const qa = s.players[q].ant;
        if (!qa) continue;
        const hs = hillCells(s, q);
        if (!hs.length) continue;
        const hd = bfs(s, q, hs.map((h) => h * 3), { underground: false });
        let n = antNode(s, q);
        let guard = 0;
        while (hd.has(n) && hd.get(n) > 0 && guard++ < 20) {
          const out = [];
          expand(s, q, n, { bridges: true, underground: false }, out);
          let best = -1, bd = 1e9;
          for (const m of out) if (hd.has(m) && hd.get(m) < bd) { bd = hd.get(m); best = m; }
          if (best < 0) break;
          cellsToBlock.add(nodeCell(best));
          n = best;
        }
      }
    }
    for (const id of cellsToBlock) {
      const c = s.cells[id];
      if (!c || c.block >= 0) continue;
      if (s.players.some((q) => q.ant && q.ant.cell === id)) continue;
      acts.push({ t: 'blockage', cell: id });
    }
  }
  if (node >= 0 && !(p.carry >= 0 && cs.res >= 0)) {
    const hasWings = hasCard(p, 'wings');
    let via = null;
    if (!hasWings && p.cards.length >= 2) {
      const sorted = [...p.cards].sort((a, b) => CARD_DROP_ORDER.indexOf(a) - CARD_DROP_ORDER.indexOf(b));
      via = [sorted[0], sorted[1]];
    }
    if (hasWings || via) {
      const targets = [];
      if (full) {
        for (const k in s.cells) for (const n of nodesOfCell(s, +k)) targets.push(n);
      } else {
        for (const n of interestingNodes(s, pi)) targets.push(n);
      }
      for (const n of targets) {
        if (nodeCell(n) === nodeCell(node) && n === node) continue;
        if (s.cells[nodeCell(n)].block >= 0) continue;
        acts.push({ t: 'wings', to: n, via: hasWings ? null : via });
      }
    }
  }
  if (hasCard(p, 'superimpose') && p.hand.length && overlayActions(s, pi, false).length) acts.push({ t: 'super' });
  acts.push({ t: 'end' });
  return acts;
}

function pendingActions(s, full) {
  const pd = s.pending;
  const pi = pd.actor;
  const p = s.players[pi];
  switch (pd.type) {
    case 'hide': return [{ t: 'hide', v: 0 }, { t: 'hide', v: 1 }];
    case 'guess': return [{ t: 'guess', v: 0 }, { t: 'guess', v: 1 }];
    case 'revenge': return [{ t: 'revenge' }, { t: 'decline' }];
    case 'tmove': {
      const node = antNode(s, pi);
      const dist = bfs(s, pi, antStarts(s, pi), { maxDepth: pd.n, underground: false });
      const acts = [{ t: 'tstay' }];
      const hd = bfs(s, pi, hillCells(s, pi).map((h) => h * 3), { underground: false });
      const cands = [];
      for (const [n, d] of dist) if (d > 0) cands.push(n);
      if (full) { for (const n of cands) acts.push({ t: 'tmove', to: n }); return acts; }
      const hillNodes = cands.filter((n) => s.cells[nodeCell(n)].hill === pi);
      if (hillNodes.length) { acts.push({ t: 'tmove', to: hillNodes[0] }); return acts; }
      const sorted = cands.filter((n) => hd.has(n)).sort((a, b) => hd.get(a) - hd.get(b));
      const curD = hd.has(node) ? hd.get(node) : 1e9;
      for (const n of sorted.slice(0, 2)) if (hd.get(n) < curD) acts.push({ t: 'tmove', to: n });
      return acts;
    }
    case 'super': {
      const acts = overlayActions(s, pi, full);
      if (consistent(s)) acts.push({ t: 'sdone' });
      if (full && pd.overlays.length) acts.push({ t: 'sundo' });
      if (!acts.length) acts.push({ t: 'sundo' });
      return acts;
    }
  }
  return [];
}

export const actionKey = (a) => JSON.stringify(a);

export function isLegal(s, a) {
  const k = actionKey(a);
  return legalActions(s, true).some((x) => actionKey(x) === k);
}

// ---------- apply ----------
function log(s, p, a, extra) {
  s.log.push({ turn: s.turnNo, p, a, ...extra });
  if (s.log.length > 80) s.log.shift();
}

function moveAnt(s, pi, node) {
  const p = s.players[pi];
  const cid = nodeCell(node);
  const c = s.cells[cid];
  const comp = compsOf(c)[node - cid * 3];
  p.ant = { cell: cid, edge: comp ? comp[0] : 0 };
}

function deliver(s, pi) {
  const p = s.players[pi];
  const r = s.resources[p.carry];
  r.owner = pi; r.carrier = -1; r.cell = -1;
  p.carry = -1;
  checkWin(s);
}

export function applyAction(s, a) {
  if (s.over) return;
  const pi = actor(s);
  const p = s.players[pi];
  const info = {};
  switch (a.t) {
    case 'inc_tiles': {
      let k = 0;
      for (let i = 0; i < 4 && s.tileStack.length; i++) { p.hand.push(s.tileStack.pop()); k++; }
      s.used.income = 1; info.n = k; break;
    }
    case 'inc_card':
      p.cards.push(s.cardStack.pop()); s.used.income = 1; break;
    case 'inc_hill': {
      const i = s.hills.findIndex((h) => h.owner === pi && h.cell === a.cell);
      s.hills.splice(i, 1);
      s.cells[a.cell].hill = -1;
      for (const q of s.players) if (q.ant && q.ant.cell === a.cell) q.ant.free = true;
      p.hillsInHand++;
      s.used.income = 1; break;
    }
    case 'swap': {
      const q = s.players[a.q];
      p.hand.splice(p.hand.indexOf(a.give), 1);
      q.hand.splice(q.hand.indexOf(a.take), 1);
      p.hand.push(a.take); q.hand.push(a.give);
      s.used.swap = 1; break;
    }
    case 'place': {
      p.hand.splice(p.hand.indexOf(a.tile), 1);
      const c = putCell(s, cellX(a.cell), cellY(a.cell), a.tile, a.rot);
      if (a.piece === 1) {
        p.hillsInHand--;
        s.hills.push({ owner: pi, cell: c.id });
        c.hill = pi;
        if (!p.ant) p.ant = { cell: c.id, edge: c.allOpen[0] };
      } else if (a.piece === 2) {
        const kind = s.resStack.pop();
        s.resources.push({ kind, cell: c.id, carrier: -1, owner: -1, laidTurn: s.turnNo });
        c.res = s.resources.length - 1;
        info.kind = kind;
      }
      break;
    }
    case 'move': moveAnt(s, pi, a.to); break;
    case 'load': {
      const c = s.cells[p.ant.cell];
      const r = s.resources[c.res];
      r.carrier = pi; r.cell = -1; c.res = -1; p.carry = s.resources.indexOf(r);
      for (const q of s.players) if (q.ant && q.ant.cell === c.id) q.ant.free = true;
      break;
    }
    case 'unload': {
      const c = s.cells[p.ant.cell];
      const r = s.resources[p.carry];
      r.carrier = -1; r.cell = c.id; c.res = p.carry; p.carry = -1;
      for (const q of s.players) if (q.ant && q.ant.cell === c.id) q.ant.free = true;
      break;
    }
    case 'fight': {
      s.used.fight = 1;
      s.pending = { type: 'hide', actor: a.q, att: pi, def: a.q, res: s.players[a.q].carry, hide: null };
      break;
    }
    case 'hide':
      s.pending.hide = a.v; s.pending.type = 'guess'; s.pending.actor = s.pending.att; break;
    case 'guess': {
      const pd = s.pending;
      pd.attWin = a.v === pd.hide;
      pd.guessV = a.v;
      const loser = pd.attWin ? pd.def : pd.att;
      if (hasCard(s.players[loser], 'revenge')) { pd.type = 'revenge'; pd.actor = loser; } else finalizeFight(s);
      break;
    }
    case 'revenge': {
      const pd = s.pending;
      removeCard(s, s.players[pd.actor], 'revenge');
      pd.type = 'hide'; pd.actor = pd.def; pd.hide = null;
      break;
    }
    case 'decline': finalizeFight(s); break;
    case 'transport': {
      if (a.card) removeCard(s, p, 'transport');
      const d1 = rint(s, 6), d2 = a.card ? rint(s, 6) : 0;
      s.lastRoll = [d1, a.card ? d2 : null];
      info.roll = d1 + d2;
      s.used.transport = 1;
      if (s.cells[p.ant.cell].hill === pi) { deliver(s, pi); info.delivered = true; } // already at the hill: store it
      else if (d1 + d2 > 0) s.pending = { type: 'tmove', actor: pi, n: d1 + d2 };
      break;
    }
    case 'tmove': {
      moveAnt(s, pi, a.to);
      if (s.cells[nodeCell(a.to)].hill === pi) deliver(s, pi);
      s.pending = null; break;
    }
    case 'tstay': s.pending = null; break;
    case 'bridge': {
      removeCard(s, p, 'bridge');
      p.bridges--;
      s.bridges.push({ owner: pi, a: a.a, b: a.b, placedOwnTurn: p.ownTurns });
      s.cells[a.a].bridge++; s.cells[a.b].bridge++;
      break;
    }
    case 'blockage': {
      removeCard(s, p, 'blockage');
      p.blockages--;
      s.blocks.push({ owner: pi, cell: a.cell });
      s.cells[a.cell].block = pi;
      break;
    }
    case 'wings': {
      if (a.via) for (const t of a.via) removeCard(s, p, t);
      else removeCard(s, p, 'wings');
      if (p.carry >= 0) { // wings cannot carry resources: drop it here
        const c = s.cells[p.ant.cell];
        const r = s.resources[p.carry];
        r.carrier = -1; r.cell = c.id; c.res = p.carry; p.carry = -1;
      }
      moveAnt(s, pi, a.to);
      break;
    }
    case 'super': {
      removeCard(s, p, 'superimpose');
      s.pending = { type: 'super', actor: pi, overlays: [] };
      break;
    }
    case 'overlay': {
      const c = s.cells[a.cell];
      s.pending.overlays.push({ cell: c.id, tile: c.tile, rot: c.rot, newTile: a.tile });
      p.hand.splice(p.hand.indexOf(a.tile), 1);
      const sh = makeCellShape(c.x, c.y, a.tile, a.rot);
      Object.assign(c, sh, { overlaid: true });
      break;
    }
    case 'sundo': {
      const ov = s.pending.overlays;
      for (let i = ov.length - 1; i >= 0; i--) {
        const o = ov[i];
        const c = s.cells[o.cell];
        Object.assign(c, makeCellShape(c.x, c.y, o.tile, o.rot), { overlaid: false });
        p.hand.push(o.newTile);
      }
      s.pending = null; break;
    }
    case 'sdone': s.pending = null; break;
    case 'end': log(s, pi, a); endTurn(s); return;
  }
  if (!s.pending || s.pending.type === 'super') s.actionsThisTurn++;
  log(s, pi, a, info);
}

function finalizeFight(s) {
  const pd = s.pending;
  if (pd.attWin) {
    const d = s.players[pd.def], at = s.players[pd.att];
    d.carry = -1;
    at.carry = pd.res;
    s.resources[pd.res].carrier = pd.att;
  }
  pd.done = true;
  s.lastFight = { att: pd.att, def: pd.def, attWin: pd.attWin, hide: pd.hide, guess: pd.guessV };
  s.pending = null;
}

// ---------- view for network clients (hide secrets) ----------
export function viewFor(s, seat) {
  const v = cloneState(s);
  v.tileStack = v.tileStack.map(() => -1);
  v.resStack = v.resStack.map(() => -1);
  v.cardStack = v.cardStack.map(() => 'x');
  for (const q of v.players) if (q.idx !== seat) q.cards = q.cards.map(() => 'x');
  if (v.pending && v.pending.type === 'guess' && v.pending.hide != null) v.pending.hide = -1;
  if (v.pending && v.pending.type === 'hide') v.pending.hide = null;
  return v;
}

export function winnerText(s) {
  if (!s.over) return '';
  return s.winner < 0 ? 'Unentschieden' : `${s.players[s.winner].name} gewinnt`;
}
