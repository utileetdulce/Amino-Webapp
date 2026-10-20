// Minimal line chart for training curves.
export class LineChart {
  constructor(canvas, { title = '', series = [], yMin = null, yMax = null, xLabel = 'Spiele', fmt = (v) => v.toFixed(2) } = {}) {
    this.cv = canvas; this.title = title; this.series = series.map((s) => ({ ...s, pts: [] }));
    this.yMin = yMin; this.yMax = yMax; this.xLabel = xLabel; this.fmt = fmt;
    this.draw();
  }
  add(name, x, y) {
    const s = this.series.find((q) => q.name === name);
    if (s && Number.isFinite(y)) { s.pts.push([x, y]); if (s.pts.length > 800) s.pts.splice(1, 1); }
  }
  clear() { for (const s of this.series) s.pts = []; this.draw(); }
  draw() {
    const cv = this.cv, dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth || 300, h = cv.clientHeight || 160;
    if (cv.width !== w * dpr || cv.height !== h * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const css = getComputedStyle(document.documentElement);
    const fg = css.getPropertyValue('--fg').trim() || '#222', grid = css.getPropertyValue('--grid').trim() || '#ddd';
    ctx.clearRect(0, 0, w, h);
    const L = 38, R = 8, T = 36, B = 18;
    let x0 = 1e18, x1 = -1e18, y0 = this.yMin ?? 1e18, y1 = this.yMax ?? -1e18;
    for (const s of this.series) for (const [x, y] of s.pts) {
      x0 = Math.min(x0, x); x1 = Math.max(x1, x);
      if (this.yMin == null) y0 = Math.min(y0, y);
      if (this.yMax == null) y1 = Math.max(y1, y);
    }
    ctx.fillStyle = fg; ctx.font = '600 12px system-ui, sans-serif'; ctx.textBaseline = 'alphabetic';
    ctx.fillText(this.title, L, 14);
    if (x0 > x1) { ctx.font = '12px system-ui'; ctx.fillStyle = '#888'; ctx.fillText('noch keine Daten', L + 6, h / 2); return; }
    if (y1 - y0 < 1e-9) { y1 += 0.5; y0 -= 0.5; }
    if (x1 === x0) x1 = x0 + 1;
    const X = (x) => L + (x - x0) / (x1 - x0) * (w - L - R), Y = (y) => h - B - (y - y0) / (y1 - y0) * (h - T - B);
    ctx.strokeStyle = grid; ctx.lineWidth = 1; ctx.fillStyle = '#888'; ctx.font = '10px system-ui';
    for (let i = 0; i <= 3; i++) {
      const yy = y0 + (y1 - y0) * i / 3;
      ctx.beginPath(); ctx.moveTo(L, Y(yy)); ctx.lineTo(w - R, Y(yy)); ctx.stroke();
      ctx.fillText(this.fmt(yy), 2, Y(yy) + 3);
    }
    ctx.fillText(`${this.xLabel}: ${Math.round(x1)}`, w - 110, h - 4);
    let lx = L;
    for (const s of this.series) {
      ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.beginPath();
      s.pts.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
      ctx.stroke();
      ctx.fillStyle = s.color; ctx.font = '11px system-ui';
      const tw = ctx.measureText(s.name).width + 18;
      ctx.fillRect(lx, 20, 8, 8); ctx.fillStyle = fg; ctx.fillText(s.name, lx + 11, 28); lx += tw;
    }
  }
}
