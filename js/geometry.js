// Triangular lattice. Cell (x, y): upward triangle if (x+y) is even, downward otherwise.
// Edges: 0 = left, 1 = right, 2 = horizontal (bottom for up, top for down).

export const H = Math.sqrt(3) / 2;
const OFF = 200;
const W = 400;

export const cellId = (x, y) => (y + OFF) * W + (x + OFF);
export const cellX = (id) => (id % W) - OFF;
export const cellY = (id) => Math.floor(id / W) - OFF;
export const isUp = (x, y) => ((x + y) & 1) === 0;

export const OPP = [1, 0, 2];

// Returns [nx, ny, neighbourEdge] for the neighbour across edge e.
export function neighbor(x, y, e) {
  if (e === 0) return [x - 1, y, 1];
  if (e === 1) return [x + 1, y, 0];
  return isUp(x, y) ? [x, y + 1, 2] : [x, y - 1, 2];
}

export function neighborId(id, e) {
  const [nx, ny, ne] = neighbor(cellX(id), cellY(id), e);
  return [cellId(nx, ny), ne];
}

// Clockwise order of absolute edges
export const ORDER_UP = [1, 2, 0];
export const ORDER_DOWN = [2, 1, 0];
export function edgePos(x, y, e) {
  const ord = isUp(x, y) ? ORDER_UP : ORDER_DOWN;
  return ord.indexOf(e);
}

// Grid distance (edge-adjacent steps, ignoring tiles), via BFS up to maxD.
export function gridDistWithin(id, maxD, pred) {
  // returns true if any cell with pred(cellId) lies within maxD steps (excluding self distance 0 unless pred true)
  let frontier = [id];
  const seen = new Set([id]);
  if (pred(id)) return true;
  for (let d = 1; d <= maxD; d++) {
    const next = [];
    for (const c of frontier) {
      const x = cellX(c), y = cellY(c);
      for (let e = 0; e < 3; e++) {
        const [nx, ny] = neighbor(x, y, e);
        const nid = cellId(nx, ny);
        if (seen.has(nid)) continue;
        seen.add(nid);
        if (pred(nid)) return true;
        next.push(nid);
      }
    }
    frontier = next;
  }
  return false;
}

// Vertices of a cell in unit-side coordinates.
export function cellVerts(x, y) {
  const x0 = x * 0.5;
  if (isUp(x, y)) return [[x0 + 0.5, y * H], [x0 + 1, (y + 1) * H], [x0, (y + 1) * H]];
  return [[x0, y * H], [x0 + 1, y * H], [x0 + 0.5, (y + 1) * H]];
}
const EV_UP = [[0, 2], [0, 1], [1, 2]];
const EV_DOWN = [[0, 2], [1, 2], [0, 1]];
export function edgeVerts(x, y, e) {
  const v = cellVerts(x, y);
  const m = isUp(x, y) ? EV_UP[e] : EV_DOWN[e];
  return [v[m[0]], v[m[1]]];
}
export function edgeMid(x, y, e) {
  const [a, b] = edgeVerts(x, y, e);
  return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
}
export function cellCenter(x, y) {
  const v = cellVerts(x, y);
  return [(v[0][0] + v[1][0] + v[2][0]) / 3, (v[0][1] + v[1][1] + v[2][1]) / 3];
}
