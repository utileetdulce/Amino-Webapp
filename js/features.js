// Feature extraction for the neural agents and the heuristic bot.
import { antNode, antStarts, bfs, hillCells, nodeCell, score, hasCard, TILE_TYPES } from './engine.js';
import { cellX, cellY, cellCenter, neighborId, neighbor, cellId } from './geometry.js';

export const ACTION_TYPES = ['inc_tiles', 'inc_card', 'inc_hill', 'swap', 'place', 'move', 'load', 'unload', 'fight', 'transport',
  'tmove', 'tstay', 'bridge', 'blockage', 'wings', 'super', 'overlay', 'sdone', 'sundo', 'hide', 'guess', 'revenge', 'decline', 'end'];
const TYPE_IDX = Object.fromEntries(ACTION_TYPES.map((t, i) => [t, i]));
export const NT = ACTION_TYPES.length;
export const NUM_AF = 16;
export const AF = NT + NUM_AF;
export const SF = 41;

function cellMin(map) {
  const m = new Map();
  for (const [n, d] of map) {
    const c = nodeCell(n);
    if (!m.has(c) || m.get(c) > d) m.set(c, d);
  }
  return m;
}

const dist2 = (c1, c2) => {
  const a = cellCenter(cellX(c1), cellY(c1)), b = cellCenter(cellX(c2), cellY(c2));
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
};

export function buildCtx(s, pi) {
  const p = s.players[pi];
  const node = antNode(s, pi);
  const distAnt = node >= 0 ? cellMin(bfs(s, pi, antStarts(s, pi))) : new Map();
  const hs = hillCells(s, pi);
  const distHill = hs.length ? cellMin(bfs(s, pi, hs.map((h) => h * 3), { underground: false })) : new Map();
  const freeRes = [];
  for (let i = 0; i < s.resources.length; i++) {
    const r = s.resources[i];
    if (r.owner < 0 && r.carrier < 0 && r.cell >= 0) freeRes.push({ idx: i, cell: r.cell, val: p.values[r.kind] });
  }
  const carriers = [];
  const enemyPos = [];
  for (let q = 0; q < s.n; q++) {
    if (q === pi) continue;
    const qa = s.players[q].ant;
    if (qa) {
      enemyPos.push(qa.cell);
      if (s.players[q].carry >= 0) carriers.push({ q, cell: qa.cell, val: p.values[s.resources[s.players[q].carry].kind] });
    }
    for (const h of hillCells(s, q)) enemyPos.push(h);
  }
  const antCell = node >= 0 ? nodeCell(node) : -1;
  const curHillDist = antCell >= 0 && distHill.has(antCell) ? distHill.get(antCell) : 99;
  return { s, pi, p, node, antCell, distAnt, distHill, freeRes, carriers, enemyPos, curHillDist, hs };
}

const PEND = { hide: 0, guess: 1, revenge: 2, tmove: 3, super: 4 };

export function stateFeatures(ctx, out = new Float32Array(SF)) {
  const { s, pi, p } = ctx;
  out.fill(0);
  let best = 0;
  for (let q = 0; q < s.n; q++) if (q !== pi) best = Math.max(best, score(s, q));
  const my = score(s, pi);
  out[0] = my / 7; out[1] = best / 7;
  out[2] = Math.min(p.hand.length, 12) / 8; out[3] = new Set(p.hand).size / 6; out[4] = p.cards.length / 6;
  out[5] = p.hillsInHand / 2; out[6] = (2 - p.hillsInHand) / 2; out[7] = p.ant ? 1 : 0; out[8] = p.carry >= 0 ? 1 : 0;
  if (p.carry >= 0) {
    out[9] = p.values[s.resources[p.carry].kind] / 3;
    out[10] = s.resources[p.carry].laidTurn === s.turnNo ? 1 : 0;
  }
  out[11] = ctx.carriers.length / 3;
  out[12] = ctx.carriers.reduce((m, c) => Math.max(m, c.val), 0) / 3;
  out[13] = ctx.freeRes.length / 4;
  out[14] = s.resStack.length / 12; out[15] = s.tileStack.length / 50;
  out[16] = s.used.income; out[17] = s.used.swap; out[18] = s.used.fight; out[19] = s.used.transport;
  if (s.pending) out[20 + PEND[s.pending.type]] = 1;
  const dFree = ctx.freeRes.reduce((m, r) => (ctx.distAnt.has(r.cell) ? Math.min(m, ctx.distAnt.get(r.cell)) : m), 99);
  out[25] = Math.min(dFree, 14) / 14;
  out[26] = Math.min(ctx.curHillDist, 14) / 14;
  const dCar = ctx.carriers.reduce((m, c) => (ctx.distAnt.has(c.cell) ? Math.min(m, ctx.distAnt.get(c.cell)) : m), 99);
  out[27] = Math.min(dCar, 14) / 14;
  out[28] = p.values[0] / 3; out[29] = p.values[1] / 3; out[30] = p.values[2] / 3;
  out[31] = Math.min(s.turnNo / (60 * s.n), 2);
  out[32] = s.n / 6;
  out[33] = hasCard(p, 'bridge') ? 1 : 0; out[34] = hasCard(p, 'blockage') ? 1 : 0; out[35] = hasCard(p, 'wings') ? 1 : 0;
  out[36] = hasCard(p, 'revenge') ? 1 : 0; out[37] = hasCard(p, 'transport') ? 1 : 0;
  out[38] = s.resources.filter((r) => r.owner === pi).length / 6;
  out[39] = s.resources.filter((r) => r.owner >= 0).length / 12;
  out[40] = 1;
  return out;
}

function freeResAt(ctx, cid) {
  for (const r of ctx.freeRes) if (r.cell === cid) return r;
  return null;
}
function carrierAt(ctx, cid) {
  for (const c of ctx.carriers) if (c.cell === cid) return c;
  return null;
}
function minDistTo(ctx, cid, list) {
  let m = 99;
  for (const t of list) m = Math.min(m, dist2(cid, t));
  return m;
}

// numeric part of action features, written into out[off..off+NUM_AF)
export function actionNumeric(ctx, a, out, off) {
  const { s, p } = ctx;
  const setDest = (cid) => {
    const fr = freeResAt(ctx, cid);
    const ca = carrierAt(ctx, cid);
    out[off + 0] = (fr ? fr.val : ca ? ca.val : 0) / 3;
    const dh = ctx.distHill.has(cid) ? ctx.distHill.get(cid) : 14;
    out[off + 1] = Math.min(dh, 14) / 14;
    out[off + 2] = Math.min(ctx.distAnt.has(cid) ? ctx.distAnt.get(cid) : 14, 14) / 14;
    out[off + 3] = fr ? 1 : 0;
    out[off + 4] = ca ? 1 : 0;
    out[off + 5] = s.cells[cid] && s.cells[cid].hill === ctx.pi ? 1 : 0;
    out[off + 6] = Math.max(-1, Math.min(1, (Math.min(ctx.curHillDist, 14) - Math.min(dh, 14)) / 6));
  };
  switch (a.t) {
    case 'move': case 'tmove': case 'wings': setDest(nodeCell(a.to)); break;
    case 'load': setDest(ctx.antCell); out[off + 0] = ctx.freeRes.length ? (freeResAt(ctx, ctx.antCell)?.val || 0) / 3 : 0; break;
    case 'fight': { const ca = carrierAt(ctx, ctx.antCell); out[off + 0] = (ca ? ca.val : 0) / 3; break; }
    case 'transport': out[off + 1] = Math.min(ctx.curHillDist, 14) / 14; out[off + 15] = a.card; break;
    case 'inc_hill': out[off + 15] = 0; break;
    case 'inc_tiles': out[off + 15] = Math.min(p.hand.length, 10) / 10; break;
    case 'inc_card': out[off + 15] = p.cards.length / 6; break;
    case 'swap': out[off + 15] = p.hand.filter((t) => t === a.give).length / 4; out[off + 13] = TILE_TYPES[a.take].open.reduce((x, y) => x + y, 0) / 3; break;
    case 'place': {
      const cid = a.cell;
      const x = cellX(cid), y = cellY(cid);
      out[off + 7] = a.piece === 1 ? 1 : 0;
      out[off + 8] = a.piece === 2 ? 1 : 0;
      const T = TILE_TYPES[a.tile];
      out[off + 13] = T.open.reduce((q, w) => q + w, 0) / 3;
      out[off + 14] = T.groups.length / 3;
      // connections and reachability from my ant
      let conn = 0, reach = 0, bestD = 99;
      for (let e = 0; e < 3; e++) {
        const [nid] = neighborId(cid, e);
        const nc = s.cells[nid];
        if (nc) {
          const [, , ne] = neighbor(x, y, e);
          if (nc.open[ne]) {
            conn++;
            if (ctx.distAnt.has(nid)) { reach = 1; bestD = Math.min(bestD, ctx.distAnt.get(nid) + 1); }
          }
        }
      }
      out[off + 9] = conn / 3;
      out[off + 10] = reach;
      out[off + 2] = Math.min(bestD, 14) / 14;
      const freeCells = ctx.freeRes.map((r) => r.cell);
      out[off + 11] = freeCells.length ? Math.min(minDistTo(ctx, cid, freeCells), 10) / 10 : 1;
      out[off + 12] = ctx.enemyPos.length ? Math.min(minDistTo(ctx, cid, ctx.enemyPos), 10) / 10 : 1;
      out[off + 1] = ctx.hs.length ? Math.min(minDistTo(ctx, cid, ctx.hs), 10) / 10 : 1;
      break;
    }
    case 'bridge': {
      out[off + 2] = Math.min(ctx.distAnt.has(a.a) ? ctx.distAnt.get(a.a) : ctx.distAnt.has(a.b) ? ctx.distAnt.get(a.b) : 14, 14) / 14;
      out[off + 9] = ctx.distAnt.has(a.a) !== ctx.distAnt.has(a.b) ? 1 : 0;
      break;
    }
    case 'blockage': {
      setDest(a.cell);
      out[off + 12] = ctx.enemyPos.length ? Math.min(minDistTo(ctx, a.cell, ctx.enemyPos), 10) / 10 : 1;
      break;
    }
    case 'overlay': {
      out[off + 13] = TILE_TYPES[a.tile].open.reduce((q, w) => q + w, 0) / 3;
      out[off + 2] = Math.min(ctx.distAnt.has(a.cell) ? ctx.distAnt.get(a.cell) : 14, 14) / 14;
      break;
    }
    case 'hide': case 'guess': out[off + 15] = a.v; break;
    case 'revenge': case 'decline': {
      const pd = s.pending;
      const r = s.resources[pd.res];
      out[off + 0] = s.players[ctx.pi].values[r.kind] / 3;
      break;
    }
    default: break;
  }
}

export function actionFeatures(ctx, a, out, off = 0) {
  for (let i = 0; i < AF; i++) out[off + i] = 0;
  out[off + TYPE_IDX[a.t]] = 1;
  actionNumeric(ctx, a, out, off + NT);
}

export function candidateMatrix(ctx, acts) {
  const m = new Float32Array(acts.length * AF);
  for (let i = 0; i < acts.length; i++) actionFeatures(ctx, acts[i], m, i * AF);
  return m;
}

export const SF_NAMES = ['eigene Punkte', 'Gegner Punkte', 'Hand Dreiecke', 'Hand Typen', 'Karten', 'Hügel in Hand', 'Hügel gelegt', 'Ameise auf Brett', 'trägt Ressource', 'Wert getragen',
  'frisch gelegt', 'Gegner tragen', 'bester Gegnerwert', 'freie Ressourcen', 'Ressourcen-Stapel', 'Dreieck-Stapel', 'Einkommen benutzt', 'Tausch benutzt', 'Kampf benutzt', 'Transport benutzt',
  'Phase: verstecken', 'Phase: tippen', 'Phase: Rache', 'Phase: Transport', 'Phase: Überlagern', 'Weg zu freier Res.', 'Weg zum Hügel', 'Weg zu Träger', 'Wert Blume', 'Wert Pilz', 'Wert Raupe',
  'Spielzeit', 'Spielerzahl', 'Karte Brücke', 'Karte Blockade', 'Karte Flügel', 'Karte Rache', 'Karte Transport', 'meine Ressourcen', 'alle Ressourcen', 'Bias'];
export const ACT_NUM_NAMES = ['Ressourcenwert', 'Distanz Hügel', 'Distanz Ameise', 'freie Res. am Ziel', 'Träger am Ziel', 'eigener Hügel', 'Weg-Gewinn', 'ist Hügel', 'ist Ressource',
  'Verbindungen', 'erreichbar', 'Nähe freie Res.', 'Nähe Gegner', 'offene Kanten', 'Wegegruppen', 'sonstiges'];
export const AF_NAMES = [...ACTION_TYPES, ...ACT_NUM_NAMES];
