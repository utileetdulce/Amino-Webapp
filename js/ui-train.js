// Training tab: controls, live charts, network visualisation, watch-a-game inspector.
import { Net, DEFAULT_ARCH } from './nn.js';
import { LineChart } from './charts.js';
import { BoardView } from './render.js';
import { newGame, applyAction, actor, legalActions } from './engine.js';
import { NNAgent, HeuristicAgent } from './agents.js';
import { SF_NAMES, AF_NAMES, SF, AF } from './features.js';
import { candidateText } from './text.js';
import { el } from './ui-play.js';
import { saveModel, deleteModel, downloadJSON, listAllModels, fetchModel } from './store.js';
import { childMeta, touchMeta, modelLabel, modelDetails, RULES_VERSION } from './modelmeta.js';
import { DEFAULT_PPO } from './selfplay.js';

function heat(canvas, arr, rows, cols, { cell = 6, labels = null, labelW = 0 } = {}) {
  const dpr = window.devicePixelRatio || 1;
  const w = labelW + cols * cell, h = rows * cell;
  canvas.width = w * dpr; canvas.height = h * dpr; canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  let mx = 1e-6;
  for (let i = 0; i < arr.length; i++) mx = Math.max(mx, Math.abs(arr[i]));
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = arr[r * cols + c] / mx;
      const a = Math.min(1, Math.abs(v));
      ctx.fillStyle = v >= 0 ? `rgba(214,69,69,${a})` : `rgba(59,130,214,${a})`;
      ctx.fillRect(labelW + c * cell, r * cell, cell, cell);
    }
    if (labels) { ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--fg') || '#222'; ctx.font = `${Math.max(7, cell + 1)}px system-ui`; ctx.textBaseline = 'middle'; ctx.fillText(labels[r], 0, r * cell + cell / 2); }
  }
}

function bars(canvas, arr, { w = 360, h = 56, color = '#d6a745' } = {}) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr; canvas.height = h * dpr; canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  let mx = 1e-6;
  for (const v of arr) mx = Math.max(mx, Math.abs(v));
  const bw = w / arr.length;
  ctx.fillStyle = '#8884'; ctx.fillRect(0, h - 1, w, 1);
  arr.forEach((v, i) => { ctx.fillStyle = color; const bh = (v / mx) * (h - 4); ctx.fillRect(i * bw + 0.5, h - 2 - bh, Math.max(1, bw - 1), bh); });
}

export class TrainUI {
  constructor(root) {
    this.root = root;
    this.worker = null;
    this.net = new Net(DEFAULT_ARCH);
    this.running = false;
    this.stats = null;
    this.watch = { game: null, timer: null, agents: null };
    this.build();
    this.initWorker();
  }

  build() {
    const r = this.root;
    r.innerHTML = '';
    const left = el('div', { class: 'tcol' });
    const right = el('div', { class: 'tcol' });
    r.append(el('div', { class: 'train' }, left, right));

    // controls
    const c = el('div', { class: 'card' });
    c.append(el('h2', {}, 'KI-Training (PPO Self-Play)'));
    c.append(el('p', { class: 'muted' }, 'Actor-Critic-Netz mit Aktions-Scoring (variable Zugmenge), trainiert mit PPO (GAE, Clipping, Entropie-Bonus) im Hintergrund-Thread. Optionaler Warmstart per Imitation der Heuristik (Behavior Cloning).'));
    this.status = el('div', { class: 'statline' }, 'bereit');
    this.btnStart = el('button', { class: 'primary', onclick: () => this.toggle() }, '▶ Training starten');
    c.append(el('div', { class: 'grp' }, this.btnStart,
      el('button', { onclick: () => this.clone() }, '🎓 Warmstart (Imitation)'),
      el('button', { onclick: () => this.send({ cmd: 'eval', games: 40 }) }, '📊 Jetzt evaluieren'),
      el('button', { onclick: () => this.reset() }, '⟲ Neues Netz')));
    const hp = el('div', { class: 'hp' });
    this.inp = {};
    const field = (key, label, val, step = 'any', title = '') => {
      const i = el('input', { type: 'number', value: val, step, title, onchange: () => this.pushCfg() });
      this.inp[key] = i;
      hp.append(el('label', { title }, label, i));
    };
    field('lr', 'Lernrate', DEFAULT_PPO.lr, '0.0001', 'Adam learning rate');
    field('entCoef', 'Entropie-Bonus', DEFAULT_PPO.entCoef, '0.001', 'Exploration');
    field('clip', 'PPO-Clip ε', DEFAULT_PPO.clip, '0.05');
    field('shape', 'Reward-Shaping', DEFAULT_PPO.shape, '0.05', 'Zusatzbelohnung pro eigenem Punkt (0 = nur Sieg/Niederlage)');
    field('gamma', 'γ (Discount)', DEFAULT_PPO.gamma, '0.001');
    field('eps', 'Spiele / Iteration', 12, '1');
    field('evalEvery', 'Eval alle N Iter.', 20, '1');
    this.oppSel = el('select', { onchange: () => this.pushCfg() }, [['mixed', 'Mix (Self-Play + Heuristik + Zufall + frühere Versionen)'], ['self', 'nur Self-Play'], ['heuristic', 'nur gegen Heuristik'], ['random', 'nur gegen Zufall']].map(([v, l]) => el('option', { value: v }, l)));
    hp.append(el('label', {}, 'Gegner', this.oppSel));
    c.append(hp, this.status);
    left.append(c);

    // model management
    const mm = el('div', { class: 'card' });
    mm.append(el('h3', {}, 'Modelle'));
    this.nameInp = el('input', { type: 'text', value: 'Mein Modell', placeholder: 'Modellname' });
    this.modelSel = el('select', { onchange: () => this.showInfo() });
    this.infoBox = el('div', { class: 'scroll' });
    mm.append(el('div', { class: 'grp' }, this.nameInp, el('button', { onclick: () => this.save() }, '💾 Speichern')));
    mm.append(el('div', { class: 'grp' }, this.modelSel,
      el('button', { onclick: () => this.loadSel() }, 'Laden & weiter trainieren'),
      el('button', { onclick: () => this.delSel() }, '🗑')));
    const file = el('input', { type: 'file', accept: 'application/json', hidden: true, onchange: (e) => this.importFile(e.target.files[0]) });
    mm.append(el('div', { class: 'grp' }, el('button', { onclick: () => this.exportCur() }, '⬇ Export (.json)'), el('button', { onclick: () => file.click() }, '⬆ Import'), file));
    mm.append(this.infoBox);
    left.append(mm);
    this.refreshModels();

    // charts
    const ch = el('div', { class: 'card charts' });
    const mk = (title, series, opts = {}) => { const cv = el('canvas', { class: 'chart' }); ch.append(cv); return new LineChart(cv, { title, series, ...opts }); };
    this.chWin = mk('Siegquote der KI (Eval, 0.5 = gleichauf)', [{ name: 'vs Zufall', color: '#2f9e5b' }, { name: 'vs Heuristik', color: '#d64545' }], { yMin: 0, yMax: 1, fmt: (v) => (v * 100).toFixed(0) + '%' });
    this.chRet = mk('Ø Return pro Episode (Training)', [{ name: 'Return', color: '#3b82d6' }, { name: 'geglättet', color: '#e0a800' }]);
    this.chEnt = mk('Policy-Entropie & KL', [{ name: 'Entropie', color: '#8e55c9' }, { name: 'KL ×100', color: '#e07a2f' }]);
    this.chLoss = mk('Verluste', [{ name: 'Value-Loss', color: '#d64545' }, { name: 'Policy-Loss ×10', color: '#3b82d6' }, { name: 'Grad-Norm', color: '#2f9e5b' }]);
    left.append(ch);

    // watch game
    const wg = el('div', { class: 'card' });
    wg.append(el('h3', {}, 'Live: so spielt das aktuelle Netz'));
    const bw = el('div', { class: 'watch-board' });
    const cv = el('canvas', { class: 'board' });
    bw.append(cv);
    wg.append(bw);
    this.watchInfo = el('div', { class: 'inspector' });
    wg.append(this.watchInfo);
    this.oppWatch = el('select', { onchange: () => this.newWatch() }, [['self', 'Netz vs Netz'], ['heuristic', 'Netz vs Heuristik']].map(([v, l]) => el('option', { value: v }, l)));
    this.watchPause = false;
    wg.append(el('div', { class: 'grp' }, this.oppWatch, el('button', { onclick: () => this.newWatch() }, 'Neues Spiel'),
      el('button', { onclick: (e) => { this.watchPause = !this.watchPause; e.target.textContent = this.watchPause ? '▶' : '⏸'; } }, '⏸')));
    right.append(wg);
    this.watchBoard = new BoardView(cv);

    // network visualisation
    const nv = el('div', { class: 'card netviz' });
    nv.append(el('h3', {}, 'Netz-Visualisierung'));
    nv.append(el('p', { class: 'muted' }, 'Gewichte: rot = positiv, blau = negativ. Zeilen = Eingabemerkmale, Spalten = Neuronen der ersten Schicht.'));
    this.cvWa = el('canvas'); this.cvWs = el('canvas'); this.cvW2 = el('canvas');
    nv.append(el('div', {}, el('b', {}, 'Aktions-Merkmale → Schicht 1 (Wa)')), el('div', { class: 'scroll' }, this.cvWa));
    nv.append(el('div', {}, el('b', {}, 'Zustands-Merkmale → Schicht 1 (Ws)')), el('div', { class: 'scroll' }, this.cvWs));
    nv.append(el('div', {}, el('b', {}, 'Schicht 1 → Schicht 2 (W2)')), el('div', { class: 'scroll' }, this.cvW2));
    nv.append(el('div', {}, el('b', {}, 'Aktivierungen der gewählten Aktion (live): Schicht 1 / Schicht 2')));
    this.cvA1 = el('canvas'); this.cvA2 = el('canvas');
    nv.append(this.cvA1, this.cvA2);
    this.netStat = el('div', { class: 'muted' });
    nv.append(this.netStat);
    right.append(nv);
    this.drawNet();
    this.newWatch();
  }

  initWorker() {
    try {
      this.worker = new Worker(new URL('./train-worker.js', import.meta.url), { type: 'module' });
    } catch (e) {
      this.status.textContent = 'Web-Worker nicht verfügbar (bitte über http:// statt file:// öffnen): ' + e.message;
      return;
    }
    this.worker.onerror = (e) => { this.status.textContent = 'Worker-Fehler: ' + (e.message || 'siehe Konsole'); };
    this.worker.onmessage = (e) => this.onWorker(e.data);
    this.initNet();
  }
  send(m) { this.worker?.postMessage(m); }
  cfg() {
    const g = (k) => +this.inp[k].value;
    return { lr: g('lr'), entCoef: g('entCoef'), clip: g('clip'), shape: g('shape'), gamma: g('gamma') };
  }
  pushCfg() { this.send({ cmd: 'cfg', cfg: this.cfg(), opponent: this.oppSel.value, eps: Math.max(1, +this.inp.eps.value | 0), evalEvery: +this.inp.evalEvery.value | 0 }); }
  initNet(netJSON = null, meta = {}) {
    this.send({ cmd: 'init', net: netJSON, arch: DEFAULT_ARCH, cfg: this.cfg(), opponent: this.oppSel.value, eps: Math.max(1, +this.inp.eps.value | 0), ...meta });
    this.send({ cmd: 'cfg', evalEvery: +this.inp.evalEvery.value | 0 });
  }
  toggle() {
    this.running = !this.running;
    if (this.running) this.runStart = Date.now(); else if (this.runStart) { this.runMs = (this.runMs || 0) + Date.now() - this.runStart; this.runStart = null; }
    this.send({ cmd: this.running ? 'start' : 'pause' });
    this.btnStart.textContent = this.running ? '⏸ Pause' : '▶ Training starten';
  }
  clone() { this.status.textContent = 'Imitationslernen von der Heuristik …'; this.send({ cmd: 'clone', games: 12 }); }
  reset() {
    this.running = false; this.send({ cmd: 'pause' }); this.btnStart.textContent = '▶ Training starten';
    for (const c of [this.chWin, this.chRet, this.chEnt, this.chLoss]) c.clear();
    this.retEma = null; this.net = new Net(DEFAULT_ARCH); this.parentModel = null; this.meta = null; this.runMs = 0; this.runStart = null;
    this.initNet();
    this.drawNet(); this.newWatch();
  }

  onWorker(m) {
    if (m.type === 'iter' || m.type === 'ready' || m.type === 'snapshot') {
      this.net = new Net(m.arch, m.theta);
    }
    if (m.type === 'iter') {
      const st = m.stats;
      this.stats = st;
      this.retEma = this.retEma == null ? st.meanReturn : this.retEma * 0.9 + st.meanReturn * 0.1;
      this.chRet.add('Return', st.games, st.meanReturn); this.chRet.add('geglättet', st.games, this.retEma);
      if (st.entropy != null) {
        this.chEnt.add('Entropie', st.games, st.entropy); this.chEnt.add('KL ×100', st.games, st.kl * 100);
        this.chLoss.add('Value-Loss', st.games, st.valueLoss); this.chLoss.add('Policy-Loss ×10', st.games, st.policyLoss * 10); this.chLoss.add('Grad-Norm', st.games, st.gradNorm);
      }
      if (st.iter % 2 === 0) { this.chRet.draw(); this.chEnt.draw(); this.chLoss.draw(); }
      this.status.textContent = `Iteration ${st.iter} · ${st.games} Spiele · ${st.steps} Schritte/Iter · ${m.gamesPerSec.toFixed(1)} Spiele/s · Ø ${st.avgTurns.toFixed(0)} Züge · Entropie ${st.entropy?.toFixed(2)}`;
      if (st.iter % 5 === 0) this.drawNet();
    } else if (m.type === 'eval') {
      this.chWin.add('vs Zufall', m.games, m.random); this.chWin.add('vs Heuristik', m.games, m.heuristic); this.chWin.draw();
    } else if (m.type === 'clone') {
      this.status.textContent = `Imitation fertig (Loss ${m.loss.toFixed(3)}, ${m.demos} Entscheidungen). Jetzt PPO starten.`;
      this.drawNet();
    } else if (m.type === 'ready') { this.drawNet(); }
  }

  drawNet() {
    const n = this.net, a = n.arch;
    heat(this.cvWa, n.Wa, AF, a.h1, { cell: 7, labels: AF_NAMES, labelW: 110 });
    heat(this.cvWs, n.Ws, SF, a.h1, { cell: 7, labels: SF_NAMES, labelW: 110 });
    heat(this.cvW2, n.W2, a.h1, a.h2, { cell: 7 });
    this.netStat.textContent = `${n.size} Parameter · Schichten: [${SF}+${AF}] → ${a.h1} → ${a.h2} → 1 (Policy), ${SF} → ${a.hv} → 1 (Value)`;
  }

  // ----- models -----
  async refreshModels() {
    this.modelList = await listAllModels();
    this.modelSel.innerHTML = '';
    for (const m of this.modelList) this.modelSel.append(el('option', { value: m.key }, modelLabel(m)));
    this.showInfo();
  }
  async showInfo() {
    const key = this.modelSel.value;
    this.infoBox.innerHTML = '';
    if (!key) return;
    const obj = await fetchModel(key);
    if (!obj) return;
    const tbl = el('table', { class: 'mdetails' });
    for (const [k, v] of modelDetails({ ...obj, name: obj.meta?.name || obj.name })) tbl.append(el('tr', {}, el('td', {}, k), el('td', {}, String(v))));
    this.infoBox.append(tbl);
  }
  async loadSel() {
    const key = this.modelSel.value;
    const obj = await fetchModel(key);
    if (!obj) return;
    this.loadObj(obj, this.modelList.find((m) => m.key === key)?.name || obj.name || 'Modell');
  }
  loadObj(obj, nme) {
    for (const c of [this.chWin, this.chRet, this.chEnt, this.chLoss]) c.clear();
    for (const h of obj.history || []) { this.chWin.add('vs Zufall', h.games, h.random); this.chWin.add('vs Heuristik', h.games, h.heuristic); }
    this.chWin.draw();
    this.parentModel = obj; this.meta = null; this.runMs = 0; this.runStart = null;
    this.nameInp.value = (obj.meta?.name || nme) + ' (weiter)';
    this.initNet(obj.net, { games: obj.games || 0, iter: obj.iter || 0 });
    const rv = obj.meta?.rulesVersion;
    this.status.textContent = `Modell „${nme}“ geladen (${obj.games || 0} Spiele).` + (rv != null && rv < RULES_VERSION ? ` ⚠ Es wurde mit Regelversion ${rv} trainiert (aktuell ${RULES_VERSION}). Weitertrainieren ist meist sinnvoll; bei geänderten Merkmalen/Netzgröße neu starten.` : '');
  }
  async save() {
    this.send({ cmd: 'snapshot' });
    await new Promise((r) => setTimeout(r, 120));
    const nme = this.nameInp.value.trim() || 'Modell';
    const ok = saveModel(nme, this.current(nme));
    this.status.textContent = ok ? `Gespeichert als „${nme}“ – jetzt im Spielen-Tab als KI wählbar.` : 'Speichern fehlgeschlagen (Speicher voll?)';
    this.refreshModels();
  }
  runMinutes() { return ((this.runMs || 0) + (this.runStart ? Date.now() - this.runStart : 0)) / 60000; }
  current(name) {
    const hist = this.chWin.series.map((s) => s.pts);
    const history = hist[0].map(([g, r], i) => ({ games: g, random: r, heuristic: hist[1][i]?.[1] ?? 0, iter: 0 }));
    if (!this.meta) {
      this.meta = childMeta(this.parentModel, { name, kind: 'ppo', trainer: 'browser', hyper: { ...this.cfg(), eps: Math.max(1, +this.inp.eps.value | 0), opponent: this.oppSel.value },
        host: { os: navigator.platform || 'browser', arch: '', cpus: navigator.hardwareConcurrency || null }, trainMinutes: this.parentModel?.meta?.trainMinutes || 0 });
      this.baseMinutes = this.meta.trainMinutes;
    }
    this.meta = touchMeta(this.meta, { name, trainMinutes: (this.baseMinutes || 0) + this.runMinutes() });
    return { version: 3, name, games: this.stats?.games || 0, iter: this.stats?.iter || 0, history, meta: this.meta, net: this.net.toJSON() };
  }
  delSel() { const k = this.modelSel.value; if (k && k.startsWith('local:')) { deleteModel(k.slice(6)); this.refreshModels(); } else if (k) this.status.textContent = 'Mitgelieferte Modelle lassen sich nicht löschen.'; }
  exportCur() { const n = this.nameInp.value.trim() || 'amino-modell'; downloadJSON(n + '.json', this.current(n)); }
  async importFile(f) {
    if (!f) return;
    try {
      const obj = JSON.parse(await f.text());
      if (!obj.net) throw new Error('keine gültige Modelldatei');
      const nme = obj.name || f.name.replace(/\.json$/, '');
      saveModel(nme, obj); await this.refreshModels(); this.loadObj(obj, nme);
    } catch (e) { this.status.textContent = 'Import fehlgeschlagen: ' + e.message; }
  }

  // ----- watch game -----
  newWatch() {
    clearTimeout(this.watch.timer);
    const w = this.watch;
    w.game = newGame({ numPlayers: 2 });
    const mkNet = () => new NNAgent(this.net, { sample: true, temperature: 1 });
    w.agents = [mkNet(), this.oppWatch.value === 'heuristic' ? new HeuristicAgent() : mkNet()];
    this.watchBoard.state = null;
    this.watchBoard.setState(w.game);
    this.stepWatch();
  }
  stepWatch() {
    const w = this.watch;
    clearTimeout(w.timer);
    if (!this.root.isConnected || this.root.hidden) { w.timer = setTimeout(() => this.stepWatch(), 600); return; }
    if (this.watchPause) { w.timer = setTimeout(() => this.stepWatch(), 300); return; }
    const s = w.game;
    if (s.over) { w.timer = setTimeout(() => this.newWatch(), 2500); return; }
    const pi = actor(s);
    const ag = w.agents[pi];
    const info = {};
    let a;
    if (ag instanceof NNAgent) { ag.net = this.net; a = ag.act(s, info); } else a = ag.act(s);
    if (info.acts) this.showDecision(s, pi, info);
    applyAction(s, a);
    this.watchBoard.setState(s);
    const stepMs = a.t === 'end' ? 120 : 260;
    w.timer = setTimeout(() => this.stepWatch(), stepMs);
  }
  showDecision(s, pi, info) {
    const box = this.watchInfo;
    box.innerHTML = '';
    box.append(el('div', { class: 'muted' }, `${s.players[pi].name} am Zug · Value-Schätzung V = ${info.value.toFixed(2)} (Gewinnchance ≈ ${(((info.value + 1) / 2) * 100).toFixed(0)}%)`));
    const order = [...info.probs.keys()].sort((a, b) => info.probs[b] - info.probs[a]).slice(0, 6);
    for (const i of order) {
      box.append(el('div', { class: 'bar' + (i === info.idx ? ' chosen' : '') }, el('i', { style: `width:${(info.probs[i] * 100).toFixed(0)}%` }), el('span', {}, `${candidateText(info.acts[i], s)} · ${(info.probs[i] * 100).toFixed(0)}%`)));
    }
    // activations for chosen candidate
    const fp = this.net.forwardPolicy(info.sf, info.A.subarray(info.idx * AF, (info.idx + 1) * AF), 1, true);
    bars(this.cvA1, fp.H1, { w: 460, color: '#d6a745' });
    bars(this.cvA2, fp.H2, { w: 230, color: '#8e55c9' });
  }
}
