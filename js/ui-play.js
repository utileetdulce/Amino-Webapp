// Play tab: setup screen, multiplayer lobby, and the interactive game UI.
import { legalActions, actor, score, TILE_TYPES, PLAYER_COLORS, PLAYER_COLOR_NAMES, KINDS, WIN_POINTS, winnerText, hasCard, cloneState, applyAction, bfs, antStarts, antNode, nodeCell, resNode } from './engine.js';
import { BoardView, drawTilePreview, KIND_ICON } from './render.js';
import { cellX, cellY, cellId } from './geometry.js';
import { GameController } from './game.js';
import { describeAction, candidateText, CARD_NAMES, CARD_HELP } from './text.js';
import { HostLink, GuestLink, openBus, unpack, extractCode } from './net.js';
import { Net } from './nn.js';
import { netFromJSON } from './gnet.js';
import { listAllModels, fetchModel } from './store.js';
import { modelLabel, rulesStatus, RULES_VERSION } from './modelmeta.js';

const CARD_ICON = { bridge: '🌉', superimpose: '🧩', blockage: '⛔', wings: '🪽', revenge: '🔥', transport: '🎲' };

// search strength of a "Netz + Baumsuche" seat: simulations per decision and the thinking-time limit (the search stops at
// whichever comes first; 0 = no time limit)
const SIMS_STEPS = [8, 16, 32, 64, 128, 256, 512, 1024];
const TIME_STEPS = [250, 500, 1000, 2000, 3000, 5000, 10000, 0];
const STRENGTH_KEY = 'amino.searchStrength';
const DEFAULT_STRENGTH = { si: 4, ti: 0 }; // 128 simulations, at most 0.25 s (as before)
function loadStrength(i) {
  try { const v = JSON.parse(localStorage.getItem(STRENGTH_KEY) || '{}')[i]; if (v && SIMS_STEPS[v.si] && TIME_STEPS[v.ti] != null) return v; } catch { /* ignore */ }
  return { ...DEFAULT_STRENGTH };
}
function saveStrength(i, v) {
  try { const all = JSON.parse(localStorage.getItem(STRENGTH_KEY) || '{}'); all[i] = v; localStorage.setItem(STRENGTH_KEY, JSON.stringify(all)); } catch { /* ignore */ }
}
const timeText = (ms) => (ms ? (ms < 1000 ? `${ms / 1000} s`.replace('.', ',') : `${ms / 1000} s`) : 'unbegrenzt');
const strengthText = (v) => `${SIMS_STEPS[v.si]} Simulationen · Bedenkzeit ${TIME_STEPS[v.ti] ? 'höchstens ' + timeText(TIME_STEPS[v.ti]) : 'unbegrenzt'}`;

export const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const k in attrs) {
    if (k === 'class') e.className = attrs[k];
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), attrs[k]);
    else if (k === 'html') e.innerHTML = attrs[k];
    else if (attrs[k] !== false && attrs[k] != null) e.setAttribute(k, attrs[k] === true ? '' : attrs[k]);
  }
  for (const c of kids.flat()) if (c != null) e.append(c.nodeType ? c : document.createTextNode(c));
  return e;
};

export async function getNetByName(key) {
  const m = await fetchModel(key);
  return m ? netFromJSON(m.net, Net) : null;
}
export const modelNames = listAllModels;

const KIND_OPTIONS = [
  ['human', 'Mensch (an diesem Gerät)'], ['heuristic', 'KI · Heuristik'], ['random', 'KI · Zufall'], ['nn', 'KI · Neuronales Netz'], ['mcts', 'KI · Netz + Baumsuche (MCTS)'], ['remote', 'Online-Mitspieler'],
];

export class PlayUI {
  constructor(root) {
    this.root = root;
    this.ctrl = null;
    this.setup = { n: 2, seats: [{ kind: 'human' }, { kind: 'heuristic' }, { kind: 'heuristic' }, { kind: 'heuristic' }, { kind: 'heuristic' }, { kind: 'heuristic' }], links: {} };
    this.models = [];
    const q = new URLSearchParams(location.search);
    const join = q.get('join'), answer = q.get('answer');
    if (join || answer) history.replaceState(null, '', location.pathname + location.hash);
    if (answer) this.renderAnswerRelay(answer);
    else if (join) this.renderJoinOnly(join);
    else this.renderSetup();
  }

  // opened an invite link: show only the join card (plus an escape hatch)
  renderJoinOnly(code) {
    this.root.innerHTML = '';
    this.root.append(this.joinPanel(code), el('div', { class: 'setup' }, el('button', { onclick: () => { this.guest?.close?.(); this.renderSetup(); } }, 'Stattdessen eigenes Spiel starten')));
  }

  // ---------------- setup / lobby ----------------
  async renderSetup() {
    this.models = await modelNames();
    const root = this.root;
    root.innerHTML = '';
    const st = this.setup;
    const wrap = el('div', { class: 'card setup' });
    wrap.append(el('h2', {}, 'Neues Spiel'));
    const nsel = el('select', { onchange: (e) => { st.n = +e.target.value; this.renderSetup(); } },
      [2, 3, 4, 5, 6].map((n) => el('option', { value: n, selected: n === st.n }, `${n} Spieler`)));
    wrap.append(el('label', { class: 'row' }, 'Spielerzahl ', nsel));
    for (let i = 0; i < st.n; i++) {
      const seat = st.seats[i];
      const row = el('div', { class: 'seat-row' });
      row.append(el('span', { class: 'dot', style: `background:${PLAYER_COLORS[i]}` }), el('b', {}, PLAYER_COLOR_NAMES[i]));
      row.append(el('select', { onchange: (e) => { seat.kind = e.target.value; this.renderSetup(); } },
        KIND_OPTIONS.map(([v, l]) => el('option', { value: v, selected: v === seat.kind }, l))));
      if (seat.kind === 'nn' || seat.kind === 'mcts') {
        if (!this.models.length) row.append(el('span', { class: 'muted' }, 'kein Modell – im Training-Tab erzeugen'));
        else {
          if (!seat.model || !this.models.some((m) => m.key === seat.model)) seat.model = (this.models.find((m) => m.default) || this.models.find((m) => m.status !== 'weak') || this.models[0]).key;
          row.append(el('select', { onchange: (e) => { seat.model = e.target.value; this.renderSetup(); } },
            this.models.map((m) => el('option', { value: m.key, selected: m.key === seat.model }, modelLabel(m)))));
          const cur = this.models.find((m) => m.key === seat.model);
          const rs = rulesStatus(cur);
          row.append(el('div', { class: 'modelinfo' + (rs.ok && cur.status !== 'weak' ? '' : ' warn') },
            `${cur.updatedAt ? 'trainiert ' + new Date(cur.updatedAt).toLocaleString('de-DE', { dateStyle: 'medium', timeStyle: 'short' }) : 'Trainingsdatum unbekannt'} · ${rs.text}${cur.note ? ' · ' + cur.note : ''}`));
          if (seat.kind === 'mcts') row.append(this.strengthBox(seat, i));
        }
      }
      if (seat.kind === 'remote') row.append(this.inviteBox(i));
      wrap.append(row);
    }
    const hint = el('p', { class: 'muted' }, 'Ziel: Als Erstes 7 Punkte an Ressourcen in den Hügel bringen. Jeder Spieler bewertet Blume/Pilz/Raupe anders (1–3 Punkte).');
    const go = el('button', { class: 'primary', onclick: () => this.startFromSetup() }, 'Spiel starten');
    const friend = el('button', { onclick: () => { st.seats[0].kind = 'human'; st.seats[1].kind = 'remote'; this.renderSetup(); } }, '👥 Freund online einladen');
    wrap.append(hint, el('div', { class: 'grp' }, go, friend));
    root.append(wrap);

    root.append(this.joinPanel(''));
  }

  // sliders for the search strength of an MCTS seat
  strengthBox(seat, i) {
    if (!seat.strength) seat.strength = loadStrength(i);
    const v = seat.strength;
    const simsOut = el('output', {}, String(SIMS_STEPS[v.si]));
    const timeOut = el('output', {}, timeText(TIME_STEPS[v.ti]));
    const slider = (key, max, out, fmt) => el('input', { type: 'range', min: 0, max, step: 1, value: v[key], oninput: (e) => { v[key] = +e.target.value; out.textContent = fmt(); saveStrength(i, v); } });
    return el('div', { class: 'strength' },
      el('label', {}, 'Simulationen ', slider('si', SIMS_STEPS.length - 1, simsOut, () => String(SIMS_STEPS[v.si])), simsOut),
      el('label', {}, 'Bedenkzeit max. ', slider('ti', TIME_STEPS.length - 1, timeOut, () => timeText(TIME_STEPS[v.ti])), timeOut),
      el('small', { class: 'muted' }, 'Mehr Simulationen = stärker, aber langsamer. Die Suche stoppt bei der ersten Grenze – auf dem Handy reicht die Zeit oft nicht für alle Simulationen.'));
  }

  // ---------------- online matchmaking ----------------
  inviteUrl(code) { return `${location.origin}${location.pathname}?join=${code}`; }
  answerUrl(code) { return `${location.origin}${location.pathname}?answer=${code}`; }

  async copyText(text, btn) {
    try { await navigator.clipboard.writeText(text); } catch {
      const t = el('textarea', { style: 'position:fixed;opacity:0' }, text); document.body.append(t); t.select();
      try { document.execCommand('copy'); } catch { /* ignore */ } t.remove();
    }
    if (btn) { const old = btn.textContent; btn.textContent = '✓ kopiert'; setTimeout(() => (btn.textContent = old), 1400); }
  }
  async shareUrl(url, text, btn) {
    if (navigator.share) { try { await navigator.share({ title: 'Amino', text, url }); return; } catch { return; } }
    this.copyText(url, btn);
  }

  // host side: one invite per remote seat
  ensureBus() {
    if (this.bus !== undefined) return;
    this.bus = openBus();
    if (this.bus) this.bus.onmessage = async (e) => {
      if (e.data?.type !== 'answer') return;
      try {
        const m = await unpack(e.data.code);
        const l = this.setup.links[m.s];
        if (l && l.tag === m.g && !l.open) { await l.acceptAnswer(e.data.code); this.bus.postMessage({ type: 'ack' }); }
      } catch (err) { console.warn(err); }
    };
  }

  inviteBox(i) {
    this.ensureBus();
    const st = this.setup;
    const box = el('div', { class: 'invite' });
    const draw = () => {
      const l = st.links[i];
      box.innerHTML = '';
      if (!l || !l.code) { box.append(el('span', { class: 'muted' }, 'Einladung wird erzeugt …')); return; }
      const label = l.open ? '✅ verbunden' : l.state === 'failed' ? '❌ Verbindung fehlgeschlagen – bitte neue Einladung erzeugen' : l.state === 'connecting' ? '⏳ verbinde …' : '⌛ wartet auf Mitspieler';
      const url = this.inviteUrl(l.code);
      const link = el('input', { type: 'text', readonly: true, value: url, onfocus: (e) => e.target.select() });
      const cp = el('button', { class: 'primary', onclick: () => this.copyText(url, cp) }, '📋 Einladungs-Link kopieren');
      const parts = [el('div', { class: 'invite-status' }, label)];
      if (!l.open) {
        parts.push(el('div', { class: 'grp' }, link, cp,
          navigator.share ? el('button', { onclick: (e) => this.shareUrl(url, 'Spiel mit mir Amino!', e.target) }, '↗ Teilen') : null,
          el('button', { title: 'Neue Einladung', onclick: () => this.makeInvite(i, true) }, '↻')));
        const ans = el('input', {
          type: 'text', placeholder: '2. Schritt: Antwort-Link des Mitspielers hier einfügen',
          oninput: async (e) => {
            const v = e.target.value.trim();
            if (v.length < 40) return;
            try { await l.acceptAnswer(v); e.target.value = ''; e.target.placeholder = 'verbinde …'; } catch (err) { e.target.value = ''; e.target.placeholder = 'Fehler: ' + err.message; }
          },
        });
        parts.push(ans, el('p', { class: 'muted small' }, '1. Link an deinen Mitspieler schicken · 2. Er öffnet ihn und schickt dir einen Antwort-Link zurück · 3. Hier einfügen – fertig. Öffnest du den Antwort-Link im selben Browser, verbindet es automatisch.'));
      }
      box.append(...parts);
    };
    box._draw = draw;
    if (!st.links[i]) this.makeInvite(i); else this.wireLink(i, draw);
    draw();
    return box;
  }

  wireLink(i, redraw) {
    const l = this.setup.links[i];
    l.onState = redraw; l.onOpen = redraw; l.onClose = redraw;
  }

  async makeInvite(i, fresh = false) {
    const st = this.setup;
    st.links[i]?.close?.();
    const l = new HostLink(i, this.tag || (this.tag = Math.random().toString(36).slice(2, 8)));
    st.links[i] = l;
    await l.createInvite();
    if (this.root.isConnected) this.renderSetup();
  }

  // guest side
  joinPanel(prefill) {
    const join = el('div', { class: 'card setup join' });
    join.append(el('h2', {}, prefill ? '🐜 Du wurdest zu einem Amino-Spiel eingeladen' : 'Online beitreten (ohne Server)'));
    const body = el('div');
    join.append(body);
    const status = el('div', { class: 'invite-status' });
    const run = async (code) => {
      try {
        this.guest?.close?.();
        const g = new GuestLink();
        this.guest = g;
        body.querySelectorAll('.answer').forEach((n) => n.remove());
        const ac = await g.acceptInvite(code);
        const url = this.answerUrl(ac);
        const cp = el('button', { class: 'primary big', onclick: () => this.copyText(url, cp) }, '📋 Antwort-Link kopieren');
        body.append(el('div', { class: 'answer' },
          el('p', {}, 'Schicke diesen Antwort-Link an den Gastgeber. Sobald er ihn öffnet oder einfügt, startet die Verbindung.'),
          el('div', { class: 'grp' }, el('input', { type: 'text', readonly: true, value: url, onfocus: (e) => e.target.select() }), cp,
            navigator.share ? el('button', { class: 'big', onclick: (e) => this.shareUrl(url, 'Meine Amino-Antwort', e.target) }, '↗ Teilen') : null)));
        status.textContent = '⌛ warte auf den Gastgeber …';
        g.onState = (s) => { if (s === 'connecting') status.textContent = '⏳ verbinde …'; if (s === 'failed') status.textContent = '❌ Verbindung fehlgeschlagen – bitte neuen Einladungs-Link anfordern'; };
        g.onOpen = () => { status.textContent = '✅ verbunden – warte auf Spielstart …'; g.send({ type: 'hello' }); body.querySelectorAll('.answer').forEach((n) => n.remove()); };
        g.onMessage = (m) => this.onHostMessage(m);
      } catch (e) { status.textContent = '❌ ' + e.message; }
    };
    const inp = el('input', {
      type: 'text', placeholder: 'Einladungs-Link oder -Code hier einfügen',
      oninput: (e) => { const v = e.target.value.trim(); if (v.length > 40) run(v); },
    });
    if (!prefill) body.append(el('p', { class: 'muted' }, 'Peer-to-Peer per WebRTC, kein Server, keine Anmeldung. Öffne den Einladungs-Link deines Gastgebers – oder füge ihn hier ein:'), inp);
    body.append(status);
    if (prefill) run(prefill);
    return join;
  }

  // opened ?answer=… link (host side): hand it to the waiting game tab
  renderAnswerRelay(code) {
    const root = this.root;
    root.innerHTML = '';
    const card = el('div', { class: 'card setup' }, el('h2', {}, 'Antwort wird übergeben …'));
    root.append(card);
    this.ensureBus();
    let acked = false;
    if (this.bus) {
      this.bus.onmessage = (e) => {
        if (e.data?.type === 'ack' && !acked) {
          acked = true;
          card.innerHTML = '';
          card.append(el('h2', {}, '✅ Antwort übergeben'), el('p', {}, 'Dein Spiel-Fenster verbindet sich jetzt. Du kannst diesen Tab schließen.'),
            el('button', { onclick: () => window.close() }, 'Tab schließen'));
        }
      };
      this.bus.postMessage({ type: 'answer', code });
    }
    setTimeout(() => {
      if (acked) return;
      card.innerHTML = '';
      const url = this.answerUrl(code);
      const cp = el('button', { class: 'primary', onclick: () => this.copyText(url, cp) }, '📋 Antwort-Link kopieren');
      card.append(el('h2', {}, 'Kein offenes Spiel in diesem Browser gefunden'),
        el('p', {}, 'Öffne Amino dort, wo du die Einladung erstellt hast, und füge den Antwort-Link in das Feld „Antwort-Link“ ein.'), el('div', { class: 'grp' }, el('input', { type: 'text', readonly: true, value: url, onfocus: (e) => e.target.select() }), cp),
        el('button', { onclick: () => { this.renderSetup(); } }, 'Zum Menü'));
    }, 1600);
  }

  async startFromSetup() {
    const st = this.setup;
    const seats = [];
    const nets = {};
    for (let i = 0; i < st.n; i++) {
      const s = st.seats[i];
      const name = PLAYER_COLOR_NAMES[i];
      if (s.kind === 'human') seats.push({ kind: 'human', name });
      else if (s.kind === 'remote') {
        if (!st.links[i]?.open) { alert(`Seat ${name}: Online-Mitspieler ist noch nicht verbunden.`); return; }
        seats.push({ kind: 'remote', name, link: st.links[i] });
      } else if (s.kind === 'nn' || s.kind === 'mcts') {
        if (!s.model) { alert('Kein Modell verfügbar. Trainiere eines im Training-Tab.'); return; }
        if (!nets[s.model]) nets[s.model] = await getNetByName(s.model);
        const mi = this.models.find((m) => m.key === s.model);
        const str = s.kind === 'mcts' ? (s.strength || loadStrength(i)) : null;
        const label = (mi ? modelLabel(mi) : s.model) + (str ? ` · ${strengthText(str)}` : '');
        seats.push({ kind: 'ai', ai: s.kind, model: s.model, modelLabel: label, modelWarn: mi && !rulesStatus(mi).ok ? rulesStatus(mi).text : '', name: `${name} (${s.kind === 'mcts' ? 'KI-Suche' : 'KI'})`,
          ...(str ? { sims: SIMS_STEPS[str.si], timeMs: TIME_STEPS[str.ti] } : {}) });
      } else seats.push({ kind: 'ai', ai: s.kind, name: `${name} (${s.kind === 'random' ? 'Zufall' : 'KI'})` });
    }
    const hasRemote = seats.some((s) => s.kind === 'remote');
    const ctrl = new GameController({ seats, nets, mode: hasRemote ? 'host' : 'local' });
    if (hasRemote) {
      seats.forEach((s, i) => {
        if (s.kind === 'remote') s.link.onMessage = (m) => ctrl.receiveFromClient(i, m);
      });
      seats.forEach((s, i) => { if (s.kind === 'remote') s.link.send({ type: 'start', seat: i, names: seats.map((q) => q.name) }); });
      ctrl.broadcast();
    }
    this.openGame(ctrl);
  }

  onHostMessage(m) {
    if (m.type === 'start') {
      const ctrl = new GameController({ seats: m.names.map((n) => ({ kind: 'human', name: n })), mode: 'client' });
      ctrl.mySeat = m.seat; ctrl.link = this.guest;
      this.pendingClient = ctrl;
    } else if (m.type === 'state') {
      if (!this.ctrl && this.pendingClient) { this.pendingClient.state = m.state; this.openGame(this.pendingClient); } else this.ctrl?.setRemoteState(m.state);
    }
  }

  // ---------------- game screen ----------------
  openGame(ctrl) {
    this.ctrl = ctrl;
    this.mode = null; this.selTile = null; this.piece = 0; this.rot = 0; this.bridgeFirst = null;
    this.logLines = [];
    this.lastFightRef = null;
    const root = this.root;
    root.innerHTML = '';
    const layout = el('div', { class: 'game' });
    const boardWrap = el('div', { class: 'board-wrap' });
    const canvas = el('canvas', { class: 'board' });
    boardWrap.append(canvas);
    this.banner = el('div', { class: 'banner' });
    boardWrap.append(this.banner);
    this.ctxMenu = el('div', { class: 'ctxmenu', hidden: true });
    this.capEl = el('div', { class: 'caption', hidden: true });
    this.bigEl = el('div', { class: 'bigpop', hidden: true });
    boardWrap.append(this.ctxMenu, this.capEl, this.bigEl);
    this.boardWrap = boardWrap;
    this.hud = el('div', { class: 'hud' });
    boardWrap.append(this.hud);
    this.tab = window.matchMedia('(max-width: 860px)').matches ? null : 'income'; this.pp = null;
    this.skipBtn = el('button', { class: 'skipbtn', hidden: true, title: 'Weiter (Leertaste / Klick auf das Brett)', onclick: () => this.skipAI() }, '⏭ Weiter');
    boardWrap.append(el('div', { class: 'board-tools' }, this.skipBtn,
      el('button', { onclick: () => { this.board.autoFit = true; this.board.flyToAll({ ms: 500 }); }, title: 'Ansicht einpassen' }, '⤢'),
      el('button', { onclick: () => this.rotate(1), title: 'Dreieck drehen (R)' }, '⟳')));
    this.side = el('div', { class: 'side' });
    layout.append(boardWrap, this.side);
    root.append(layout);
    this.modal = el('div', { class: 'modal', hidden: true });
    root.append(this.modal);
    this.board = new BoardView(canvas);
    this.board.onClick = (cid, ev) => this.onBoardClick(cid, ev);
    this.board.onHover = (cid) => this.onBoardHover(cid);
    this.board.onWheelRotate = (d) => this.rotate(d);
    this._key = (e) => {
      if (!this.ctrl || !this.root.isConnected) return;
      if (e.key === 'r' || e.key === 'R') this.rotate(1);
      if (e.key === 'Escape') { this.mode = null; this.bridgeFirst = null; this.refresh(); }
      if ((e.key === ' ' || e.key === 'Enter') && !this.myTurn() && !['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON'].includes(document.activeElement?.tagName)) { e.preventDefault(); this.skipAI(); }
    };
    window.addEventListener('keydown', this._key);
    // click outside the panel collapses the open tab (the click itself still reaches the board)
    this._outside = (e) => {
      if (!this.ctrl || !this.tab || !this.root.isConnected) return;
      if (e.target.closest('.side, .ctxmenu, .modal, .tabbar')) return;
      if (!this.root.contains(e.target)) return;
      const pd = this.ctrl.state.pending;
      if (pd && this.myTurn() && ['tmove', 'super'].includes(pd.type)) return;
      this.tab = null;
      if (this.mode !== 'place' && this.mode !== 'overlay') this.mode = this.mode === 'move' ? 'move' : this.mode;
      this.refresh();
    };
    document.addEventListener('pointerdown', this._outside, true);
    ctrl.on((s, ev) => this.onState(s, ev));
    this.onState(ctrl.state, {});
    this.board.fit();
  }

  closeGame() {
    window.removeEventListener('keydown', this._key);
    document.removeEventListener('pointerdown', this._outside, true);
    this.ctrl?.destroy();
    this.ctrl = null;
    this.renderSetup();
  }

  viewSeat() {
    const c = this.ctrl, s = c.state;
    if (c.mode === 'client') return c.mySeat;
    const humans = c.seats.map((x, i) => (x.kind === 'human' ? i : -1)).filter((i) => i >= 0);
    if (humans.length === 1) return humans[0];
    if (humans.includes(actor(s))) return actor(s);
    return humans[0] ?? 0;
  }
  myTurn() {
    const c = this.ctrl, s = c.state;
    return !s.over && c.isLocalSeat(actor(s));
  }

  skipAI() { this.hideBig(); this.ctrl?.skip(); }

  caption(pre, sub, ms) {
    const el_ = this.capEl;
    clearTimeout(this._capT);
    const s = this.ctrl.state;
    el_.innerHTML = '';
    el_.style.setProperty('--pc', pre.color || '#888');
    el_.append(el('span', { class: 'cap-icon' }, pre.icon), el('div', { class: 'cap-body' }, el('b', {}, pre.text), (sub || pre.sub) ? el('span', {}, sub || pre.sub) : null),
      this.myTurn() ? null : el('i', { class: 'cap-hint' }, 'Klick = weiter'));
    el_.hidden = false;
    el_.classList.remove('in'); void el_.offsetWidth; el_.classList.add('in');
    this._capT = setTimeout(() => { el_.hidden = true; }, ms || 3200);
    void s;
  }

  showBig(big) {
    const e = this.bigEl;
    clearTimeout(this._bigT);
    e.innerHTML = '';
    e.className = 'bigpop ' + (big.kind || '');
    e.style.setProperty('--pc', big.color || '#444');
    e.append(el('div', { class: 'big-icon' }, big.fists ? el('span', { class: 'fists' }, el('i', {}, big.fists[0] ? '🤜' : '🤛'), el('em', {}, 'vs'), el('i', {}, big.fists[1] ? '🤜' : '🤛')) : big.icon),
      el('div', { class: 'big-title' }, big.title), big.sub ? el('div', { class: 'big-sub' }, big.sub) : null);
    e.hidden = false;
    this._bigT = setTimeout(() => this.hideBig(), big.ms || (this.ctrl?.fast ? 900 : 2400));
  }
  hideBig() { clearTimeout(this._bigT); if (this.bigEl) this.bigEl.hidden = true; }

  cameraFor(pre) {
    const c = this.ctrl;
    if (!c.cinematic || c.fast || !this.board) return;
    const b = this.board;
    this.board.insetBottom = this.isMobile() ? this.side.offsetHeight : 0;
    const nextLocal = pre.nextSeat != null && c.isLocalSeat(pre.nextSeat);
    if (pre.overview || !pre.focus.length) { if (pre.overview && (nextLocal || !c.state.over)) b.flyToAll({ ms: 600 }); return; }
    b.flyTo(pre.focus, { ms: 700, maxScale: pre.zoom === 'tight' ? 130 : 105, minScale: 48, pad: pre.zoom === 'tight' ? 1.6 : 1.4 });
  }

  onState(s, ev) {
    if (ev?.type === 'announce') {
      this._announced = true;
      const col = s.players[ev.seat].color;
      this.caption({ ...ev.pre, color: col }, null, (this.ctrl.fast ? 900 : 4200));
      this.cameraFor(ev.pre);
      this.skipBtn.hidden = false;
      return;
    }
    if (ev?.type === 'applied') {
      const { pre, post, seat } = ev;
      const local = this.ctrl.isLocalSeat(seat);
      const showCard = post.card && post.card !== 'x' && local;
      this.logLines.push((pre.log || pre.text) + (post.sub ? ' – ' + post.sub : '') + (showCard ? ` – ${CARD_NAMES[post.card]}` : ''));
      if (this.logLines.length > 60) this.logLines.shift();
      const col = s.players[seat].color;
      if (!local && !this._announced) { this.caption({ ...pre, color: col }, post.sub, 3200); this.cameraFor(pre); }
      else if (post.sub && !s.over) this.caption({ ...pre, color: col }, post.sub, this.ctrl.fast ? 900 : 3200);
      this._announced = false;
      if (post.dice) this.board.rollDice({ cell: post.dice.cell, values: post.dice.values, card: post.dice.card });
      if (post.big) this.showBig(post.big);
      else if (showCard) this.showBig({ icon: CARD_ICON[post.card], title: `Neue Karte: ${CARD_NAMES[post.card]}`, sub: CARD_HELP[post.card], kind: 'card', color: col, ms: 3600 });
    }
    this.skipBtn.hidden = s.over || this.myTurn() || this.ctrl.seats[actor(s)]?.kind !== 'ai';
    this.refresh();
  }

  refresh() {
    const c = this.ctrl;
    if (!c) return;
    const s = c.state;
    this.acts = this.myTurn() ? legalActions(s, true) : [];
    const me = this.viewSeat();
    // auto modes
    const pd = s.pending;
    if (this.myTurn() && pd) {
      if (pd.type === 'tmove') this.mode = 'tmove';
      else if (pd.type === 'super') this.mode = 'overlay';
    } else if (this.mode === 'tmove' || this.mode === 'overlay') this.mode = null;
    // keep tile selection valid
    const hand = s.players[me]?.hand || [];
    if (this.selTile != null && !hand.includes(this.selTile)) this.selTile = null;
    if (this.selTile == null && hand.length && (this.mode === 'place' || this.mode === 'overlay')) this.selTile = hand[0];
    this.hideCtx();
    this.renderSide();
    this.boardWrap.style.setProperty('--sideh', (this.isMobile() ? this.side.offsetHeight : 0) + 'px');
    this.renderHud();
    const hudH = this.isMobile() ? this.hud.offsetHeight : 0;
    this.boardWrap.style.setProperty('--hudh', hudH + 'px');
    this.board.insetTop = this.isMobile() ? hudH + 60 : 0; // keep the board clear of the player list and the banner
    this.renderModal();
    this.updateBoardMode();
    this.board.focusSeat = me;
    this.board.setState(s);
    this.ensureVisible();
    this.renderBanner();
  }

  ensureVisible() {
    const b = this.board, s = this.ctrl.state;
    if (this.ctrl.cinematic && !this.myTurn()) return;
    if (b.cam) return;
    let out = false;
    for (const k in s.cells) {
      const c = s.cells[k];
      const x = b.tx((c.x + 1) * 0.5), y = b.ty((c.y + 0.5) * 0.866);
      if (x < 30 || y < 30 || x > b.w - 30 || y > b.h - 30) { out = true; break; }
    }
    if (out) b.fit();
  }

  renderBanner() {
    const s = this.ctrl.state;
    const pd = s.pending;
    let t;
    if (s.over) t = '🏁 ' + winnerText(s);
    else {
      const a = actor(s);
      const mine = this.myTurn();
      const nm = s.players[a].name;
      if (pd) {
        t = { hide: 'Verteidiger versteckt die Ressource …', guess: 'Angreifer tippt auf eine Faust …', revenge: 'Rache?', tmove: `Transport: Wurf ${s.lastRoll?.[1] != null ? s.lastRoll[0] + '+' + s.lastRoll[1] : s.lastRoll?.[0]} – Ressource voranbringen`, super: 'Überlagern: Dreiecke über bestehende legen' }[pd.type];
        t = `${nm}: ${t}`;
      } else t = mine ? `Du bist dran (${nm})` : `${nm} ist dran …`;
    }
    this.banner.textContent = t;
    this.banner.style.borderColor = s.over ? '#444' : PLAYER_COLORS[actor(s)];
  }

  // ---- side panel: tabs ("Reiter") per action plan + always-visible end-turn button ----
  setTab(id) {
    this.tab = this.tab === id ? null : id;
    this.bridgeFirst = null;
    if (this.tab === 'place') { this.mode = this.ctrl.state.pending?.type === 'super' ? 'overlay' : 'place'; }
    else if (this.mode === 'place' || (this.mode === 'overlay' && !this.ctrl.state.pending)) this.mode = null;
    if (this.tab !== 'move' && this.mode === 'move') this.mode = null;
    this.pp = null;
    this.refresh();
  }

  seq(list) { this.ctrl.dispatchSeq(list, actor(this.ctrl.state)); }

  // simulate a chain of actions on a copy; returns the steps or null when impossible
  plan(build) {
    const sim = cloneState(this.ctrl.state);
    const steps = [];
    const find = (t, f = () => true) => legalActions(sim, true).find((a) => a.t === t && f(a));
    const run = (a) => { applyAction(sim, a); steps.push(a); return true; };
    return build({ sim, find, run, steps }) ? steps : null;
  }
  planAttack(q, fly = false) {
    const me = this.viewSeat();
    return this.plan(({ sim, find, run, steps }) => {
      const qc = sim.players[q].ant.cell;
      if (sim.players[me].carry >= 0) { const u = find('unload'); if (!u) return false; run(u); }
      if (fly) { if (sim.players[me].ant.cell !== qc) { const w = find('wings', (a) => Math.floor(a.to / 3) === qc); if (!w) return false; run(w); } }
      else if (sim.players[me].ant.cell !== qc) { const mv = find('move', (a) => Math.floor(a.to / 3) === qc); if (!mv) return false; run(mv); }
      const f = find('fight', (a) => a.q === q);
      if (!f) return false;
      steps.push(f); return true;
    });
  }
  planCollect(cell) {
    const me = this.viewSeat();
    return this.plan(({ sim, find, run }) => {
      if (sim.players[me].carry >= 0) return false;
      const rn = resNode(sim.cells[cell]);
      if (antNode(sim, me) !== rn) { const mv = find('move', (a) => a.to === rn); if (!mv) return false; run(mv); }
      const l = find('load');
      if (!l) return false;
      run(l); return true;
    });
  }
  attackTargets() {
    const s = this.ctrl.state, me = this.viewSeat(), p = s.players[me];
    if (!this.myTurn() || s.pending || !p.ant || s.used.fight) return [];
    const dist = bfs(s, me, antStarts(s, me));
    const cellDist = new Map();
    for (const [n, d] of dist) { const c = nodeCell(n); if (!cellDist.has(c) || cellDist.get(c) > d) cellDist.set(c, d); }
    const out = [];
    for (const q of s.players) {
      if (q.idx === me || !q.ant || q.carry < 0) continue;
      const steps = this.planAttack(q.idx);
      const fly = steps ? null : (hasCard(p, 'wings') || p.cards.length >= 2 ? this.planAttack(q.idx, true) : null);
      if (!steps && !fly) continue;
      const kind = s.resources[q.carry].kind;
      out.push({ q: q.idx, kind, steps, fly, dist: cellDist.get(q.ant.cell) ?? 0, unload: p.carry >= 0 });
    }
    return out;
  }

  renderSide() {
    const c = this.ctrl, s = c.state;
    const me = this.viewSeat();
    const side = this.side;
    side.innerHTML = '';
    const mine = this.myTurn();
    const A = this.acts || [];
    const pd = s.pending;
    if (mine && pd) { if (pd.type === 'tmove') this.tab = 'move'; if (pd.type === 'super') this.tab = 'place'; }
    side.classList.toggle('open', !!this.tab);
    const has = (t, f = () => true) => A.some((a) => a.t === t && f(a));
    const p = s.players[me];
    const targets = this.tab === 'attack' || mine ? this.attackTargets() : [];
    this._targets = targets;

    const badge = {
      income: !mine ? '' : s.used.income ? '✓' : '!',
      place: p.hand.length || '',
      attack: targets.length || '',
      move: p.carry >= 0 ? KIND_ICON[s.resources[p.carry].kind] : (has('transport') ? '🎲' : ''),
      cards: p.cards.length || '',
      log: '',
    };
    const TABS = [['income', '📥', 'Einnahme'], ['place', '🔺', 'Auslegen'], ['attack', '⚔️', 'Angriff'], ['move', '🐜', 'Bewegen'], ['cards', '🃏', 'Karten'], ['log', '📜', 'Verlauf']];
    const rail = el('div', { class: 'rail', role: 'tablist' });
    for (const [id, icon, label] of TABS) {
      rail.append(el('button', { class: 'tab' + (this.tab === id ? ' active' : '') + (id === 'income' && mine && !s.used.income ? ' attn' : '') + (id === 'attack' && targets.length ? ' hot' : ''), role: 'tab', title: label, onclick: () => this.setTab(id) },
        el('span', { class: 'ti' }, icon), el('span', { class: 'tl' }, label), badge[id] !== '' && badge[id] != null ? el('span', { class: 'tb' + (badge[id] === '✓' ? ' ok' : '') }, String(badge[id])) : null));
    }
    rail.append(el('button', { class: 'endbtn', disabled: !has('end'), title: 'Runde beenden', onclick: () => { this.mode = null; this.pp = null; this.ctrl.dispatch({ t: 'end' }, actor(s)); } },
      el('span', { class: 'ti' }, '⏭'), el('span', { class: 'tl' }, 'Runde beenden')));

    const pane = el('div', { class: 'pane' });
    if (this.tab) {
      const builders = { income: () => this.paneIncome(), place: () => this.panePlace(), attack: () => this.paneAttack(), move: () => this.paneMove(), cards: () => this.paneCards(), log: () => this.paneLog() };
      pane.append(el('div', { class: 'pane-head', title: 'Einklappen', onclick: () => this.setTab(this.tab) }, el('b', {}, TABS.find((t) => t[0] === this.tab)[2]), el('span', { class: 'x' }, '✕')));
      pane.append(builders[this.tab]());
    }
    side.append(pane, rail);
  }

  opt(icon, title, sub, enabled, fn, cls = '') {
    return el('button', { class: 'opt ' + cls, disabled: !enabled, onclick: fn }, el('span', { class: 'oi' }, icon), el('span', { class: 'ot' }, el('b', {}, title), sub ? el('small', {}, sub) : null));
  }
  turnHint() {
    if (this.myTurn()) return el('span');
    return el('p', { class: 'muted' }, 'Du bist gerade nicht am Zug.');
  }

  paneIncome() {
    const s = this.ctrl.state, A = this.acts || [];
    const has = (t) => A.some((a) => a.t === t);
    const dsp = (a) => { this.tab = null; this.ctrl.dispatch(a, actor(s)); };
    const p = s.players[this.viewSeat()];
    const box = el('div', { class: 'opts' });
    box.append(this.turnHint());
    box.append(el('small', { class: 'muted' }, s.used.income ? '✓ Einkommen für diesen Zug schon genommen.' : 'Wähle eine Einnahme (1× pro Zug):'));
    box.append(this.opt('🔺', '4 Dreiecke', `Stapel: ${s.tileStack.length}`, has('inc_tiles'), () => dsp({ t: 'inc_tiles' })));
    box.append(this.opt('🃏', 'Ameisenkarte', `Stapel: ${s.cardStack.length}`, has('inc_card'), () => dsp({ t: 'inc_card' })));
    box.append(this.opt('⛰', 'Hügel zurücknehmen', `${2 - p.hillsInHand} Hügel liegen auf dem Brett`, has('inc_hill'), () => this.pickHill()));
    box.append(el('hr'));
    box.append(this.opt('🔄', 'Dreieck tauschen', s.used.swap ? 'schon benutzt' : '1× pro Zug – gegen ein Dreieck eines Gegners', has('swap'), () => this.swapDialog()));
    return box;
  }

  panePlace() {
    const s = this.ctrl.state, me = this.viewSeat(), p = s.players[me], pd = s.pending;
    const mine = this.myTurn();
    const box = el('div', { class: 'opts' });
    box.append(this.turnHint());
    if (pd?.type === 'super') {
      box.append(el('p', {}, '🧩 Überlagern: Wähle ein Dreieck und tippe auf ein freies, noch nicht überlagertes Feld. Am Ende müssen alle Kanten passen.'));
      box.append(el('div', { class: 'grp' }, el('button', { class: 'primary', disabled: !(this.acts || []).some((a) => a.t === 'sdone'), onclick: () => this.ctrl.dispatch({ t: 'sdone' }, actor(s)) }, '✔ Fertig'),
        el('button', { disabled: !(this.acts || []).some((a) => a.t === 'sundo'), onclick: () => this.ctrl.dispatch({ t: 'sundo' }, actor(s)) }, '↺ Zurücksetzen')));
    } else box.append(el('small', { class: 'muted' }, 'Wähle ein Dreieck, dann ein hervorgehobenes Feld auf dem Brett. Dort kannst du Dreieck, Drehung und Beilage noch ändern.'));
    const tiles = el('div', { class: 'tiles' });
    const counts = {};
    p.hand.forEach((t) => (counts[t] = (counts[t] || 0) + 1));
    for (const t of Object.keys(counts).map(Number).sort()) {
      const cv = el('canvas', { class: 'tilecv' });
      tiles.append(el('button', { class: 'tile' + (this.selTile === t ? ' sel' : ''), title: TILE_TYPES[t].name, disabled: !mine, onclick: () => { this.selTile = t; if (this.mode !== 'overlay') this.mode = 'place'; this.closeSheet(); this.refresh(); } }, cv, el('span', { class: 'cnt' }, '×' + counts[t])));
      drawTilePreview(cv, t, this.rot % 3, { color: this.selTile === t ? '#9fc58a' : '#8db37a' });
    }
    if (!p.hand.length) tiles.append(el('span', { class: 'muted' }, 'Keine Dreiecke – hole dir Einnahme.'));
    box.append(tiles);
    if (this.mode !== 'overlay') {
      const pieces = el('div', { class: 'pieces' });
      for (const [v, l, ok] of [[0, 'Nur Dreieck', true], [1, `⛰ Hügel (${p.hillsInHand})`, p.hillsInHand > 0], [2, `❔ Ressource (${s.resStack.length})`, s.resStack.length > 0]]) {
        pieces.append(el('label', {}, el('input', { type: 'radio', name: 'piece', checked: this.piece === v, disabled: !ok, onchange: () => { this.piece = v; this.refresh(); } }), l));
      }
      box.append(pieces);
    }
    box.append(el('button', { onclick: () => this.rotate(1) }, '⟳ Drehen (R / Rechtsklick)'));
    return box;
  }

  paneAttack() {
    const s = this.ctrl.state, me = this.viewSeat(), p = s.players[me];
    const box = el('div', { class: 'opts' });
    box.append(this.turnHint());
    if (s.used.fight) box.append(el('small', { class: 'muted' }, '✓ Angriff in diesem Zug schon benutzt.'));
    const list = this._targets || [];
    if (!list.length) box.append(el('p', { class: 'muted' }, 'Kein erreichbarer Gegner mit Ressource. Nur Ameisen, die eine Ressource tragen und zu denen ein Weg führt, sind wählbar.'));
    for (const t of list) {
      const q = s.players[t.q];
      const row = el('div', { class: 'target', style: `--pc:${q.color}` });
      row.append(el('div', { class: 'th' }, el('span', { class: 'dot', style: `background:${q.color}` }), el('b', {}, q.name), el('span', { class: 'tr' }, `${KIND_ICON[t.kind]} ${KINDS[t.kind]} · ${p.values[t.kind]} P für dich`)));
      const how = t.steps ? [t.unload ? 'legt deine Ressource ab' : null, t.dist ? `läuft ${t.dist} Feld${t.dist > 1 ? 'er' : ''}` : 'steht schon dort', 'greift an'] : ['fliegt (Flügel-Karte)', 'greift an'];
      row.append(el('small', {}, how.filter(Boolean).join(' → ')));
      row.append(el('div', { class: 'grp' },
        t.steps ? el('button', { class: 'primary', onclick: () => { this.tab = null; this.seq(t.steps); } }, '⚔ Angreifen') : null,
        t.fly ? el('button', { class: t.steps ? '' : 'primary', onclick: () => { this.tab = null; this.seq(t.fly); } }, '🪽 per Flügel') : null));
      box.append(row);
    }
    box.append(el('small', { class: 'muted' }, 'Die Ameise läuft automatisch zum Gegner. Der Verteidiger versteckt die Ressource in einer Faust, du tippst.'));
    return box;
  }

  paneMove() {
    const s = this.ctrl.state, me = this.viewSeat(), p = s.players[me], pd = s.pending;
    const A = this.acts || [];
    const has = (t, f = () => true) => A.some((a) => a.t === t && f(a));
    const dsp = (a) => this.ctrl.dispatch(a, actor(s));
    const box = el('div', { class: 'opts' });
    box.append(this.turnHint());
    if (pd?.type === 'tmove') {
      box.append(el('p', {}, `🎲 Wurf: ${s.lastRoll?.[1] != null ? s.lastRoll[0] + ' + ' + s.lastRoll[1] : s.lastRoll?.[0]} – tippe auf ein hervorgehobenes Feld (≤ ${pd.n} Dreiecke) oder in den Hügel.`));
      box.append(el('button', { onclick: () => dsp({ t: 'tstay' }) }, 'Stehen bleiben'));
      return box;
    }
    if (p.carry >= 0) {
      const k = s.resources[p.carry].kind;
      box.append(el('p', {}, `Trägt ${KIND_ICON[k]} ${KINDS[k]} (${p.values[k]} P).`, s.resources[p.carry].laidTurn === s.turnNo ? ' In diesem Zug gelegt – noch nicht transportierbar.' : ''));
      box.append(el('p', { class: 'hint' }, '🎲 Mit Ressource bewegt sich die Ameise nur per Transport (Würfelwurf) – ohne Ausnahme. Normales Bewegen ist gesperrt, solange sie trägt.'));
      const here = s.cells[p.ant?.cell];
      if (here && here.hill === me) box.append(el('p', { class: 'hint' }, '🏠 Du stehst im eigenen Hügel. Die Ressource liegt noch auf dem Ameisenrücken – mit „Transport“ wird sie eingelagert und zählt.'));
    } else box.append(el('p', { class: 'muted' }, 'Die Ameise trägt nichts.'));
    box.append(this.opt('🎲', 'Transport', s.used.transport ? 'schon benutzt' : 'Würfel (0–5): so weit Richtung Hügel', has('transport', (a) => !a.card), () => dsp({ t: 'transport', card: 0 }), 'big'));
    if (hasCard(p, 'transport')) box.append(this.opt('🎲🎲', 'Transport mit Karte', 'zwei Würfel', has('transport', (a) => a.card), () => dsp({ t: 'transport', card: 1 })));
    box.append(el('hr'));
    box.append(this.opt('🐜', 'Ameise bewegen', 'dann Zielfeld auf dem Brett antippen', has('move'), () => { this.mode = this.mode === 'move' ? null : 'move'; this.closeSheet(); this.refresh(); }, this.mode === 'move' ? 'on' : ''));
    const row = el('div', { class: 'grp' }, el('button', { disabled: !has('load'), onclick: () => dsp({ t: 'load' }) }, '⬆ Aufladen'), el('button', { disabled: !has('unload'), onclick: () => dsp({ t: 'unload' }) }, '⬇ Abladen'));
    box.append(row);
    // quick collect
    if (this.myTurn() && !pd && p.carry < 0 && p.ant) {
      const items = [];
      for (const r of s.resources) {
        if (r.owner >= 0 || r.carrier >= 0 || r.cell < 0) continue;
        const steps = this.planCollect(r.cell);
        if (steps) items.push({ r, steps });
      }
      if (items.length) {
        box.append(el('small', { class: 'muted' }, 'Freie Ressourcen in Reichweite:'));
        for (const { r, steps } of items) box.append(this.opt(KIND_ICON[r.kind], `${KINDS[r.kind]} holen`, `${p.values[r.kind]} P · läuft hin & lädt auf`, true, () => this.seq(steps)));
      }
    }
    return box;
  }

  paneCards() {
    const s = this.ctrl.state, me = this.viewSeat(), p = s.players[me];
    const A = this.acts || [];
    const has = (t) => A.some((a) => a.t === t);
    const dsp = (a) => this.ctrl.dispatch(a, actor(s));
    const box = el('div', { class: 'opts' });
    box.append(this.turnHint());
    const cnt = {};
    p.cards.forEach((t) => (cnt[t] = (cnt[t] || 0) + 1));
    if (!p.cards.length) box.append(el('p', { class: 'muted' }, 'Keine Ameisenkarten. Hol dir eine über „Einnahme“.'));
    const go = (mode) => () => { this.mode = mode; this.bridgeFirst = null; this.closeSheet(); this.refresh(); };
    const defs = [
      ['bridge', '🌉', 'Brücke', 'zwei Felder verbinden', has('bridge'), go('bridge')],
      ['blockage', '⛔', 'Blockade', 'ein Feld bis zu deinem nächsten Zug sperren', has('blockage'), go('blockage')],
      ['wings', '🪽', 'Flügel', 'zu einem beliebigen Feld fliegen', has('wings'), go('wings')],
      ['superimpose', '🧩', 'Überlagern', 'Dreiecke über bestehende legen', has('super'), () => dsp({ t: 'super' })],
      ['revenge', '🔥', 'Rache', 'wird automatisch nach einem verlorenen Kampf angeboten', false, null],
      ['transport', '🎲', 'Transport', 'wähle sie im Reiter „Bewegen“ beim Transport', false, null],
    ];
    for (const [id, icon, name, sub, ok, fn] of defs) {
      if (!cnt[id]) continue;
      box.append(this.opt(icon, `${name} ×${cnt[id]}`, sub, ok && this.myTurn(), fn || (() => {}), this.mode === id ? 'on' : ''));
    }
    if (!cnt.wings && p.cards.length >= 2) box.append(this.opt('🪽', 'Flügel (2 Karten tauschen)', 'zwei beliebige Karten werden zu Flügeln', has('wings') && this.myTurn(), go('wings'), this.mode === 'wings' ? 'on' : ''));
    return box;
  }

  paneLog() {
    const c = this.ctrl, s = c.state;
    const box = el('div', { class: 'opts' });
    if (c.mode !== 'client') {
      const tg = (label, on, fn, title) => el('button', { class: on ? 'on' : '', title, onclick: () => { fn(); this.renderSide(); } }, label);
      box.append(el('div', { class: 'grp' },
        el('label', { class: 'speed', title: 'Tempo der KI-Züge' }, '🐢', el('input', { type: 'range', min: 150, max: 2000, value: 2150 - c.speed, oninput: (e) => (c.speed = 2150 - +e.target.value) }), '🐇'),
        tg('🎥 Kamera', c.cinematic, () => { c.cinematic = !c.cinematic; }, 'Kamerafahrten zu den KI-Aktionen'),
        tg('⏩ Schnell', c.fast, () => { c.fast = !c.fast; }, 'KI-Züge ohne Kamera und Pausen'),
        tg('⏸ Schritt', c.paused, () => c.setPaused(!c.paused), 'Schrittmodus: KI geht nur per Klick/Leertaste weiter')));
    }
    if (c.lastAI) {
      const L = c.lastAI;
      const ib = el('div', { class: 'inspector' }, el('b', {}, `KI-Entscheidung (${s.players[L.seat].name}) · V=${L.value.toFixed(2)}`));
      const order = [...L.probs.keys()].sort((a, b) => L.probs[b] - L.probs[a]).slice(0, 5);
      for (const i of order) ib.append(el('div', { class: 'bar' + (i === L.idx ? ' chosen' : '') }, el('i', { style: `width:${(L.probs[i] * 100).toFixed(0)}%` }), el('span', {}, `${candidateText(L.acts[i], s)} · ${(L.probs[i] * 100).toFixed(0)}%`)));
      box.append(ib);
    }
    const ais = c.seats.map((q, i) => ({ q, i })).filter((x) => x.q.modelLabel);
    if (ais.length) {
      const mb = el('div', { class: 'modelbox' }, el('b', {}, 'KI-Modelle in diesem Spiel'));
      for (const { q, i } of ais) mb.append(el('div', { class: 'modelinfo' + (q.modelWarn ? ' warn' : '') }, el('span', { class: 'dot', style: `background:${s.players[i].color}` }), `${s.players[i].name}: ${q.modelLabel}${q.modelWarn ? ' · ⚠ ' + q.modelWarn : ''}`));
      box.append(mb);
    }
    const ul = el('div', { class: 'loglines' });
    this.logLines.slice(-30).reverse().forEach((l) => ul.append(el('div', {}, l)));
    box.append(ul, el('button', { onclick: () => { if (confirm('Spiel verlassen?')) this.closeGame(); } }, 'Zum Menü'));
    return box;
  }

  renderHud() {
    const s = this.ctrl.state, me = this.viewSeat();
    const hud = this.hud;
    hud.innerHTML = '';
    for (const p of s.players) {
      const store = [0, 1, 2].map((k) => { const n = s.resources.filter((r) => r.owner === p.idx && r.kind === k).length; return n ? el('span', { class: 'chip', title: `${KINDS[k]} = ${p.values[k]} P` }, KIND_ICON[k], '×' + n) : null; });
      hud.append(el('div', { class: 'hp' + (actor(s) === p.idx && !s.over ? ' active' : '') + (p.idx === me ? ' me' : ''), style: `--pc:${p.color}`, title: this.ctrl.seats[p.idx]?.modelLabel ? `Modell: ${this.ctrl.seats[p.idx].modelLabel}` : null },
        el('span', { class: 'dot', style: `background:${p.color}` }), el('b', {}, p.name), el('span', { class: 'score' }, `${score(s, p.idx)}/${WIN_POINTS}`),
        el('span', { class: 'hm' }, store, el('span', { class: 'chip', title: 'Dreiecke' }, '▲', p.hand.length), el('span', { class: 'chip', title: 'Karten' }, '🃏', p.cards.length),
          p.carry >= 0 ? el('span', { class: 'chip', title: 'trägt' }, '🐜', KIND_ICON[s.resources[p.carry].kind]) : null)));
    }
  }

  pickHill() {
    const hills = this.acts.filter((a) => a.t === 'inc_hill');
    if (hills.length === 1) return this.ctrl.dispatch(hills[0], actor(this.ctrl.state));
    this.mode = 'hill'; this.closeSheet(); this.refresh();
  }

  swapDialog() {
    const s = this.ctrl.state;
    const swaps = this.acts.filter((a) => a.t === 'swap');
    const m = this.modal;
    this.swapOpen = true;
    const close = () => { this.swapOpen = false; m.hidden = true; };
    m.hidden = false; m.innerHTML = '';
    const box = el('div', { class: 'dlg' }, el('h3', {}, 'Dreieck tauschen (Gegner kann sich nicht wehren)'));
    const qs = [...new Set(swaps.map((a) => a.q))];
    let q = qs[0], give = null, take = null;
    const draw = () => {
      body.innerHTML = '';
      body.append(el('div', {}, 'Mit: ', qs.map((x) => el('button', { class: x === q ? 'on' : '', onclick: () => { q = x; give = take = null; draw(); } }, s.players[x].name))));
      const gs = [...new Set(swaps.filter((a) => a.q === q).map((a) => a.give))];
      const ts = [...new Set(swaps.filter((a) => a.q === q).map((a) => a.take))];
      const mk = (list, cur, set) => el('div', { class: 'tiles' }, list.map((t) => {
        const cv = el('canvas', { class: 'tilecv' });
        const b = el('button', { class: 'tile' + (cur === t ? ' sel' : ''), onclick: () => { set(t); draw(); } }, cv);
        drawTilePreview(cv, t, 0);
        return b;
      }));
      body.append(el('div', {}, 'Du gibst:'), mk(gs, give, (t) => (give = t)), el('div', {}, `Du nimmst (von ${s.players[q].name}):`), mk(ts, take, (t) => (take = t)));
      body.append(el('button', { class: 'primary', disabled: give == null || take == null, onclick: () => { close(); this.tab = null; this.ctrl.dispatch({ t: 'swap', q, give, take }, actor(s)); } }, 'Tauschen'), el('button', { onclick: close }, 'Abbrechen'));
    };
    const body = el('div');
    box.append(body); m.append(box); draw();
  }

  renderModal() {
    const c = this.ctrl, s = c.state;
    const pd = s.pending;
    const m = this.modal;
    if (s.over) {
      m.hidden = false; m.innerHTML = '';
      const table = el('table', {}, s.players.map((p) => el('tr', {}, el('td', {}, p.name), el('td', {}, score(s, p.idx) + ' P'))));
      m.append(el('div', { class: 'dlg' }, el('h2', {}, '🏁 ' + winnerText(s)), table,
        el('button', { class: 'primary', onclick: () => this.closeGame() }, 'Zum Menü')));
      return;
    }
    if (!pd || !this.myTurn() || !['hide', 'guess', 'revenge'].includes(pd.type)) {
      if (!this.swapOpen) m.hidden = true;
      return;
    }
    m.hidden = false; m.innerHTML = '';
    const dsp = (a) => c.dispatch(a, actor(s));
    const nm = s.players[pd.actor].name;
    if (pd.type === 'hide' || pd.type === 'guess') {
      const hide = pd.type === 'hide';
      const resK = s.resources[pd.res].kind;
      m.append(el('div', { class: 'dlg' },
        el('h3', {}, hide ? `${nm} verteidigt ${KIND_ICON[resK]} – in welcher Faust ist die Ressource? (Gegner wegschauen)` : `${nm} greift an – in welcher Faust ist die Ressource?`),
        el('div', { class: 'fists' },
          el('button', { onclick: () => dsp({ t: hide ? 'hide' : 'guess', v: 0 }) }, '🤛 links'),
          el('button', { onclick: () => dsp({ t: hide ? 'hide' : 'guess', v: 1 }) }, 'rechts 🤜'))));
    } else {
      const lost = pd.attWin ? s.players[pd.def].name : s.players[pd.att].name;
      m.append(el('div', { class: 'dlg rev' },
        el('h3', {}, `${nm}: Kampf verloren (${pd.attWin ? 'Angreifer traf' : 'Verteidiger setzte sich durch'}). Rache-Karte spielen?`),
        el('button', { class: 'primary', onclick: () => dsp({ t: 'revenge' }) }, 'Rache!'), el('button', { onclick: () => dsp({ t: 'decline' }) }, 'Nein')));
    }
  }

  // ---- board interaction ----
  rotate(d) {
    if (this.pp) { this.cyclePickerRot(); return; }
    this.rot = (this.rot + d + 3) % 3; this.refresh();
  }

  placeMap() {
    const m = new Map();
    const want = this.mode === 'overlay' ? 'overlay' : 'place';
    for (const a of this.acts) {
      if (a.t !== want || a.tile !== this.selTile) continue;
      if (want === 'place' && a.piece !== this.piece) continue;
      if (!m.has(a.cell)) m.set(a.cell, []);
      m.get(a.cell).push(a.rot);
    }
    return m;
  }

  updateBoardMode() {
    const b = this.board;
    b.cellColors = new Map(); b.ghost = null; b.edgeMarks = [];
    if (!this.myTurn()) return;
    const s = this.ctrl.state;
    const A = this.acts;
    const hi = 'rgba(255,255,255,.30)';
    switch (this.mode) {
      case 'place': case 'overlay': {
        this.pmap = this.placeMap();
        const want = this.mode === 'overlay' ? 'overlay' : 'place';
        for (const a of A) if (a.t === want) b.cellColors.set(a.cell, this.pmap.has(a.cell) ? hi : 'rgba(255,255,255,.14)');
        break;
      }
      case 'move': case 'tmove': case 'wings': {
        const t = this.mode;
        this.dmap = new Map();
        for (const a of A) if (a.t === t) { const cid = Math.floor(a.to / 3); if (!this.dmap.has(cid)) this.dmap.set(cid, this.destFor(t, cid)); }
        for (const c of this.dmap.keys()) b.cellColors.set(c, 'rgba(255,240,140,.35)');
        break;
      }
      case 'blockage': for (const a of A) if (a.t === 'blockage') b.cellColors.set(a.cell, 'rgba(210,60,60,.35)'); break;
      case 'hill': for (const a of A) if (a.t === 'inc_hill') b.cellColors.set(a.cell, 'rgba(255,255,255,.4)'); break;
      case 'bridge': {
        const bs = A.filter((a) => a.t === 'bridge');
        if (this.bridgeFirst == null) {
          for (const a of bs) { b.cellColors.set(a.a, 'rgba(120,180,255,.35)'); b.cellColors.set(a.b, 'rgba(120,180,255,.35)'); }
        } else {
          b.cellColors.set(this.bridgeFirst, 'rgba(60,120,255,.55)');
          for (const a of bs) {
            if (a.a === this.bridgeFirst) b.cellColors.set(a.b, 'rgba(255,240,140,.5)');
            if (a.b === this.bridgeFirst) b.cellColors.set(a.a, 'rgba(255,240,140,.5)');
          }
        }
        break;
      }
      default: break;
    }
    if (this.pp) b.ghost = { cell: this.pp.cid, tile: this.pp.tile, rot: this.pp.rot, piece: this.mode === 'place' ? this.pp.piece : 0, comp: this.pp.comp, seat: this.viewSeat() };
  }

  onBoardHover(cid) {
    if (!this.myTurn() || this.pp) return;
    const b = this.board;
    if ((this.mode === 'place' || this.mode === 'overlay') && this.pmap?.has(cid)) {
      const rots = this.pmap.get(cid);
      const rot = rots.includes(this.rot) ? this.rot : rots[0];
      const ca = this.mode === 'place' && this.piece === 2 ? this.acts.find((a) => a.t === 'place' && a.cell === cid && a.tile === this.selTile && a.rot === rot && a.piece === 2) : null;
      b.ghost = { cell: cid, tile: this.selTile, rot, piece: this.mode === 'place' ? this.piece : 0, comp: ca ? ca.comp : 0, seat: this.viewSeat() };
    } else b.ghost = null;
    b.draw();
  }

  isMobile() { return window.matchMedia('(max-width: 860px)').matches; }
  closeSheet() { if (this.isMobile()) this.tab = null; }

  hideCtx() { if (this.ctxMenu) { this.ctxMenu.hidden = true; this.ctxMenu.classList.remove('picker'); } }

  openMenu(items, ev, title = '') {
    const m = this.ctxMenu;
    m.innerHTML = '';
    if (title) m.append(el('div', { class: 'ctx-title' }, title));
    for (const it of items) {
      m.append(el('button', { class: it.cls || '', disabled: it.disabled, onclick: (e) => { e.stopPropagation(); this.hideCtx(); it.fn(); } }, it.label));
    }
    if (!items.length) { m.hidden = true; return; }
    this.placeCtx(ev);
  }

  placeCtx(ev) {
    const m = this.ctxMenu;
    m.hidden = false;
    const r = this.boardWrap.getBoundingClientRect();
    const mw = m.offsetWidth, mh = m.offsetHeight;
    let x = (ev?.clientX ?? r.left + r.width / 2) - r.left + 8, y = (ev?.clientY ?? r.top + r.height / 2) - r.top + 8;
    x = Math.max(6, Math.min(x, r.width - mw - 6)); y = Math.max(6, Math.min(y, r.height - mh - 6));
    m.style.left = x + 'px'; m.style.top = y + 'px';
    this._menuEv = ev;
  }

  previewPlace(cid, ev) { this.openPicker(cid, ev); }

  // move-like action into cell cid; on a triangle with separate paths prefer the piece the resource lies on
  destFor(t, cid) {
    const c = this.ctrl.state.cells[cid];
    const list = (this.acts || []).filter((a) => a.t === t && Math.floor(a.to / 3) === cid);
    return (c && c.res >= 0 && list.find((a) => a.to === resNode(c))) || list[0];
  }

  // placement picker: choose tile, accessory and rotation for the tapped cell, ghost shows the result
  pickerOptions(cid) {
    const want = this.mode === 'overlay' ? 'overlay' : 'place';
    const list = (this.acts || []).filter((a) => a.t === want && a.cell === cid);
    return { want, list };
  }
  openPicker(cid, ev) {
    const { list } = this.pickerOptions(cid);
    if (!list.length) return;
    const tiles = [...new Set(list.map((a) => a.tile))];
    const tile = tiles.includes(this.selTile) ? this.selTile : tiles[0];
    const pieces = [...new Set(list.filter((a) => a.tile === tile).map((a) => a.piece ?? 0))];
    const piece = pieces.includes(this.piece) ? this.piece : pieces[0];
    const rots = [...new Set(list.filter((a) => a.tile === tile && (a.piece ?? 0) === piece).map((a) => a.rot))];
    this.pp = { cid, tile, piece, rot: rots.includes(this.rot) ? this.rot : rots[0], comp: 0, ev };
    this.drawPicker();
  }
  // path pieces a resource may go on for the picker's tile/rotation
  pickerComps(list) {
    const pp = this.pp;
    return list.filter((a) => a.tile === pp.tile && a.piece === 2 && a.rot === pp.rot).map((a) => a.comp);
  }
  cyclePickerRot() {
    const { list } = this.pickerOptions(this.pp.cid);
    const rots = [...new Set(list.filter((a) => a.tile === this.pp.tile && (a.piece ?? 0) === this.pp.piece).map((a) => a.rot))];
    this.pp.rot = rots[(rots.indexOf(this.pp.rot) + 1) % rots.length];
    this.rot = this.pp.rot;
    this.drawPicker();
  }
  drawPicker() {
    const pp = this.pp, m = this.ctxMenu;
    const { want, list } = this.pickerOptions(pp.cid);
    const over = want === 'overlay';
    const tiles = [...new Set(list.map((a) => a.tile))].sort();
    const pieces = [...new Set(list.filter((a) => a.tile === pp.tile).map((a) => a.piece ?? 0))].sort();
    const rots = [...new Set(list.filter((a) => a.tile === pp.tile && (a.piece ?? 0) === pp.piece).map((a) => a.rot))];
    this.selTile = pp.tile; this.piece = pp.piece;
    const comps = !over && pp.piece === 2 ? this.pickerComps(list) : [];
    if (comps.length && !comps.includes(pp.comp)) pp.comp = comps[0];
    m.innerHTML = '';
    m.classList.add('picker');
    m.append(el('div', { class: 'ctx-title' }, over ? 'Überlagern – Dreieck wählen' : 'Dreieck wählen'));
    const row = el('div', { class: 'tiles' });
    for (const t of tiles) {
      const cv = el('canvas', { class: 'tilecv' });
      row.append(el('button', { class: 'tile' + (t === pp.tile ? ' sel' : ''), title: TILE_TYPES[t].name, onclick: (e) => {
        e.stopPropagation();
        const ps = [...new Set(list.filter((a) => a.tile === t).map((a) => a.piece ?? 0))];
        pp.tile = t; if (!ps.includes(pp.piece)) pp.piece = ps[0];
        const rs = list.filter((a) => a.tile === t && (a.piece ?? 0) === pp.piece).map((a) => a.rot);
        if (!rs.includes(pp.rot)) pp.rot = rs[0];
        this.drawPicker();
      } }, cv));
      drawTilePreview(cv, t, t === pp.tile ? pp.rot : (rots[0] ?? 0), { color: t === pp.tile ? '#9fc58a' : '#8db37a', size: 46 });
    }
    m.append(row);
    if (!over && pieces.length > 1) {
      const pn = ['Nur Dreieck', '⛰ Hügel', '❔ Ressource'];
      const pr = el('div', { class: 'seg' });
      for (const v of pieces) pr.append(el('button', { class: v === pp.piece ? 'on' : '', onclick: (e) => { e.stopPropagation(); pp.piece = v; const rs = list.filter((a) => a.tile === pp.tile && (a.piece ?? 0) === v).map((a) => a.rot); if (!rs.includes(pp.rot)) pp.rot = rs[0]; this.drawPicker(); } }, pn[v]));
      m.append(pr);
    }
    const ok = el('button', { class: 'primary', onclick: (e) => {
      e.stopPropagation();
      const a = over ? { t: 'overlay', cell: pp.cid, tile: pp.tile, rot: pp.rot } : { t: 'place', cell: pp.cid, tile: pp.tile, rot: pp.rot, piece: pp.piece };
      if (!over && pp.piece === 2) a.comp = pp.comp;
      this.pp = null; m.classList.remove('picker'); this.hideCtx();
      this.ctrl.dispatch(a, actor(this.ctrl.state));
    } }, over ? '✔ Überlagern' : '✔ Hier legen');
    if (comps.length > 1) m.append(el('div', { class: 'seg' }, el('button', { onclick: (e) => { e.stopPropagation(); pp.comp = comps[(comps.indexOf(pp.comp) + 1) % comps.length]; this.drawPicker(); } }, `❔ Anderes Wegstück (${comps.indexOf(pp.comp) + 1}/${comps.length})`)));
    m.append(el('div', { class: 'seg' }, rots.length > 1 ? el('button', { onclick: (e) => { e.stopPropagation(); this.cyclePickerRot(); } }, '⟳ Drehen') : null, ok,
      el('button', { onclick: (e) => { e.stopPropagation(); this.pp = null; m.classList.remove('picker'); this.refresh(); } }, '✖')));
    this.updateBoardMode(); this.board.draw();
    this.placeCtx(pp.ev);
    ok.focus({ preventScroll: true });
  }

  cellMenu(cid, ev) {
    const s = this.ctrl.state, A = this.acts, me = this.viewSeat();
    const dsp = (a) => this.ctrl.dispatch(a, actor(s));
    const items = [];
    const c = s.cells[cid];
    const mine = s.players[me];
    // placement options on empty cell
    const places = A.filter((a) => a.t === 'place' && a.cell === cid);
    const seen = new Set();
    for (const a of places) {
      const k = a.tile + ':' + a.piece;
      if (seen.has(k)) continue; seen.add(k);
      const lbl = `▲ ${TILE_TYPES[a.tile].name}${a.piece === 1 ? ' + ⛰ Hügel' : a.piece === 2 ? ' + ❔ Ressource' : ''}`;
      items.push({ label: lbl, fn: () => { this.selTile = a.tile; this.piece = a.piece; this.mode = 'place'; this.refresh(); this.previewPlace(cid, ev); } });
    }
    const ov = A.filter((a) => a.t === 'overlay' && a.cell === cid);
    for (const t of new Set(ov.map((a) => a.tile))) items.push({ label: `▲ Überlagern mit ${TILE_TYPES[t].name}`, fn: () => { this.selTile = t; this.mode = 'overlay'; this.refresh(); this.previewPlace(cid, ev); } });
    const mv = this.destFor('move', cid);
    if (mv) items.push({ label: '🐜 Ameise hierher', fn: () => dsp(mv) });
    const tm = this.destFor('tmove', cid);
    if (tm) items.push({ label: '🎲 Ressource hierher bringen', fn: () => dsp(tm) });
    const wg = this.destFor('wings', cid);
    if (wg) items.push({ label: '🪽 Flügel hierher', fn: () => dsp(wg) });
    const bl = A.find((a) => a.t === 'blockage' && a.cell === cid);
    if (bl) items.push({ label: '⛔ Blockade hier', fn: () => dsp(bl) });
    const br = A.filter((a) => a.t === 'bridge' && (a.a === cid || a.b === cid));
    if (br.length) items.push({ label: '🌉 Brücke von hier …', fn: () => { this.mode = 'bridge'; this.bridgeFirst = cid; this.refresh(); } });
    const ih = A.find((a) => a.t === 'inc_hill' && a.cell === cid);
    if (ih) items.push({ label: '⛰ Hügel zurücknehmen', fn: () => dsp(ih) });
    const antHere = mine.ant && mine.ant.cell === cid;
    if (antHere) {
      for (const [t, lbl] of [['load', '⬆ Ressource aufladen'], ['unload', '⬇ Ressource abladen'], ['fight', '⚔ Angreifen']]) {
        const a = A.find((x) => x.t === t); if (a) items.push({ label: lbl, fn: () => dsp(a) });
      }
      for (const a of A.filter((x) => x.t === 'transport')) items.push({ label: a.card ? '🎲🎲 Transport (Karte)' : '🎲 Transport', fn: () => dsp(a) });
    }
    if (c) {
      const here = s.players.filter((q) => q.ant && q.ant.cell === cid).map((q) => q.name);
      const info = [here.length ? '🐜 ' + here.join(', ') : '', c.hill >= 0 ? `⛰ ${s.players[c.hill].name}` : '', c.res >= 0 ? KIND_ICON[s.resources[c.res].kind] : '', c.block >= 0 ? '⛔' : ''].filter(Boolean).join(' · ');
      return this.openMenu(items, ev, info || 'Feld');
    }
    this.openMenu(items, ev, items.length ? 'Leeres Feld' : '');
  }

  onBoardClick(cid, ev) {
    this.hideCtx();
    if (!this.myTurn() && !this.ctrl.state.over) { this.skipAI(); return; }
    if (this.pp) { this.pp = null; this.refresh(); if (cid < 0) return; }
    if (!this.myTurn() || cid < 0) return;
    const s = this.ctrl.state;
    const dsp = (a) => { this.board.ghost = null; this.ctrl.dispatch(a, actor(s)); };
    switch (this.mode) {
      case 'place': case 'overlay': {
        if (this.pickerOptions(cid).list.length) return this.openPicker(cid, ev);
        break;
      }
      case 'move': case 'tmove': case 'wings': {
        const a = this.dmap?.get(cid);
        if (!a) break;
        if (this.mode !== 'tmove') this.mode = null;
        return dsp(a);
      }
      case 'blockage': case 'hill': {
        const a = this.acts.find((x) => (this.mode === 'blockage' ? x.t === 'blockage' : x.t === 'inc_hill') && x.cell === cid);
        if (a) { this.mode = null; return dsp(a); }
        break;
      }
      case 'bridge': {
        const bs = this.acts.filter((a) => a.t === 'bridge');
        if (this.bridgeFirst == null) {
          if (bs.some((a) => a.a === cid || a.b === cid)) { this.bridgeFirst = cid; return this.refresh(); }
        } else {
          const a = bs.find((x) => (x.a === this.bridgeFirst && x.b === cid) || (x.b === this.bridgeFirst && x.a === cid));
          if (a) { this.mode = null; this.bridgeFirst = null; return dsp(a); }
          this.bridgeFirst = null; return this.refresh();
        }
        break;
      }
      default: break;
    }
    this.cellMenu(cid, ev);
  }
}
