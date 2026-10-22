// Play tab: setup screen, multiplayer lobby, and the interactive game UI.
import { legalActions, actor, score, TILE_TYPES, PLAYER_COLORS, PLAYER_COLOR_NAMES, KINDS, WIN_POINTS, winnerText, hasCard } from './engine.js';
import { BoardView, drawTilePreview, KIND_ICON } from './render.js';
import { cellX, cellY, cellId } from './geometry.js';
import { GameController } from './game.js';
import { describeAction, candidateText, CARD_NAMES, CARD_HELP } from './text.js';
import { HostLink, GuestLink, openBus, unpack, extractCode } from './net.js';
import { Net } from './nn.js';
import { listModels, loadModel, loadPretrained } from './store.js';

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

export async function getNetByName(name) {
  if (name === 'Vortrainiert') {
    const p = await loadPretrained();
    return p ? Net.fromJSON(p.net) : null;
  }
  const m = loadModel(name);
  return m ? Net.fromJSON(m.net) : null;
}
export async function modelNames() {
  const p = await loadPretrained();
  return [...(p ? ['Vortrainiert'] : []), ...listModels()];
}

const KIND_OPTIONS = [
  ['human', 'Mensch (an diesem Gerät)'], ['heuristic', 'KI · Heuristik'], ['random', 'KI · Zufall'], ['nn', 'KI · Neuronales Netz'], ['remote', 'Online-Mitspieler'],
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
      if (seat.kind === 'nn') {
        if (!this.models.length) row.append(el('span', { class: 'muted' }, 'kein Modell – im Training-Tab erzeugen'));
        else {
          if (!seat.model || !this.models.includes(seat.model)) seat.model = this.models[0];
          row.append(el('select', { onchange: (e) => (seat.model = e.target.value) },
            this.models.map((m) => el('option', { value: m, selected: m === seat.model }, m))));
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
      } else if (s.kind === 'nn') {
        if (!s.model) { alert('Kein Modell verfügbar. Trainiere eines im Training-Tab.'); return; }
        if (!nets[s.model]) nets[s.model] = await getNetByName(s.model);
        seats.push({ kind: 'ai', ai: 'nn', model: s.model, name: `${name} (KI)` });
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
    this.sheetOpen = false; this.pp = null;
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
    ctrl.on((s, ev) => this.onState(s, ev));
    this.onState(ctrl.state, {});
    this.board.fit();
  }

  closeGame() {
    window.removeEventListener('keydown', this._key);
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
    this._bigT = setTimeout(() => this.hideBig(), this.ctrl?.fast ? 900 : 2400);
  }
  hideBig() { clearTimeout(this._bigT); if (this.bigEl) this.bigEl.hidden = true; }

  cameraFor(pre) {
    const c = this.ctrl;
    if (!c.cinematic || c.fast || !this.board) return;
    const b = this.board;
    this.board.insetBottom = this.isMobile() ? (this.sheetOpen ? this.board.h * 0.55 : 150) : 0;
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
      this.logLines.push(pre.text + (post.sub ? ' – ' + post.sub : ''));
      if (this.logLines.length > 60) this.logLines.shift();
      const col = s.players[seat].color;
      if (!local && !this._announced) { this.caption({ ...pre, color: col }, post.sub, 3200); this.cameraFor(pre); }
      else if (post.sub && !s.over) this.caption({ ...pre, color: col }, post.sub, this.ctrl.fast ? 900 : 3200);
      this._announced = false;
      if (post.dice) this.board.rollDice({ cell: post.dice.cell, values: post.dice.values, card: post.dice.card });
      if (post.big) this.showBig(post.big);
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

  // ---- side panel ----
  renderSide() {
    const c = this.ctrl, s = c.state;
    const me = this.viewSeat();
    const side = this.side;
    side.innerHTML = '';
    side.classList.toggle('open', !!this.sheetOpen);
    side.append(el('button', { class: 'sheet-handle', onclick: () => { this.sheetOpen = !this.sheetOpen; this.renderSide(); } },
      el('span', { class: 'grip' }), this.sheetOpen ? 'Menü schließen' : `Aktionen & Hand (▲${s.players[this.viewSeat()].hand.length})`));
    // players
    const pl = el('div', { class: 'players' });
    for (const p of s.players) {
      const sc = score(s, p.idx);
      const store = [0, 1, 2].map((k) => {
        const n = s.resources.filter((r) => r.owner === p.idx && r.kind === k).length;
        return el('span', { class: 'chip', title: `${KINDS[k]} = ${p.values[k]} P` }, KIND_ICON[k], `×${n}`, el('sub', {}, `${p.values[k]}P`));
      });
      const row = el('div', { class: 'player' + (actor(s) === p.idx && !s.over ? ' active' : '') + (p.idx === me ? ' me' : ''), style: `--pc:${p.color}` },
        el('div', { class: 'ph' }, el('span', { class: 'dot', style: `background:${p.color}` }), el('b', {}, p.name), el('span', { class: 'score' }, `${sc}/${WIN_POINTS}`)),
        el('div', { class: 'pm' }, store, el('span', { class: 'chip', title: 'Dreiecke auf der Hand' }, '▲', p.hand.length), el('span', { class: 'chip', title: 'Ameisenkarten' }, '🃏', p.cards.length),
          el('span', { class: 'chip', title: 'Hügel/Ameise' }, '⛰', 2 - p.hillsInHand, p.carry >= 0 ? el('span', {}, ' 🐜', KIND_ICON[s.resources[p.carry].kind]) : '')));
      pl.append(row);
    }
    side.append(pl);

    const mine = this.myTurn();
    const A = this.acts || [];
    const has = (t, f = () => true) => A.some((a) => a.t === t && f(a));
    const me_p = s.players[me];
    const pd = s.pending;

    // hand tiles
    const handBox = el('div', { class: 'card' });
    handBox.append(el('h3', {}, `Hand von ${me_p.name}`));
    const tiles = el('div', { class: 'tiles' });
    const counts = {};
    me_p.hand.forEach((t) => (counts[t] = (counts[t] || 0) + 1));
    for (const t of Object.keys(counts).map(Number).sort()) {
      const cv = el('canvas', { class: 'tilecv' });
      const btn = el('button', {
        class: 'tile' + (this.selTile === t ? ' sel' : ''), title: TILE_TYPES[t].name, disabled: !mine,
        onclick: () => { this.selTile = t; if (this.mode !== 'overlay') this.mode = 'place'; this.closeSheet(); this.refresh(); },
      }, cv, el('span', { class: 'cnt' }, '×' + counts[t]));
      tiles.append(btn);
      drawTilePreview(cv, t, this.rot % 3, { color: this.selTile === t ? '#9fc58a' : '#8db37a' });
    }
    if (!me_p.hand.length) tiles.append(el('span', { class: 'muted' }, 'keine Dreiecke'));
    handBox.append(tiles);
    if (this.mode === 'place' || this.mode === 'overlay') {
      const piece = el('div', { class: 'pieces' });
      if (this.mode === 'place') {
        for (const [v, l, ok] of [[0, 'Nur Dreieck', true], [1, `⛰ Hügel (${me_p.hillsInHand})`, me_p.hillsInHand > 0], [2, `❔ Ressource (${s.resStack.length})`, s.resStack.length > 0]]) {
          piece.append(el('label', {}, el('input', { type: 'radio', name: 'piece', checked: this.piece === v, disabled: !ok, onchange: () => { this.piece = v; this.refresh(); } }), l));
        }
      }
      piece.append(el('button', { onclick: () => this.rotate(1) }, '⟳ Drehen (R / Rechtsklick)'));
      handBox.append(piece);
    }
    side.append(handBox);

    // actions
    const act = el('div', { class: 'card actions' });
    const btn = (label, enabled, fn, cls = '', title = '') => el('button', { class: cls, disabled: !enabled, title, onclick: fn }, label);
    const dsp = (a) => this.ctrl.dispatch(a, actor(s));
    act.append(el('h3', {}, 'Aktionen'));
    if (pd && pd.type === 'tmove') act.append(btn('Stehen bleiben', mine, () => dsp({ t: 'tstay' })));
    else if (pd && pd.type === 'super') {
      act.append(btn('✔ Fertig', has('sdone'), () => dsp({ t: 'sdone' }), 'primary'), btn('↺ Zurücksetzen', has('sundo'), () => dsp({ t: 'sundo' })));
    } else if (!pd) {
      const g1 = el('div', { class: 'grp' }, el('span', { class: 'lbl' }, 'Einkommen'));
      g1.append(btn('4 Dreiecke', has('inc_tiles'), () => dsp({ t: 'inc_tiles' })), btn('Ameisenkarte', has('inc_card'), () => dsp({ t: 'inc_card' })),
        btn('Hügel zurück', has('inc_hill'), () => this.pickHill()), btn('Tauschen', has('swap'), () => this.swapDialog()));
      const g2 = el('div', { class: 'grp' }, el('span', { class: 'lbl' }, 'Ameise'));
      g2.append(btn('Bewegen', has('move'), () => { this.mode = 'move'; this.closeSheet(); this.refresh(); }, this.mode === 'move' ? 'on' : ''),
        btn('Aufladen', has('load'), () => dsp({ t: 'load' })), btn('Abladen', has('unload'), () => dsp({ t: 'unload' })),
        btn('⚔ Angreifen', has('fight'), () => dsp(A.find((a) => a.t === 'fight'))),
        btn('🎲 Transport', has('transport', (a) => !a.card), () => dsp({ t: 'transport', card: 0 })),
        btn('🎲🎲 Transport (Karte)', has('transport', (a) => a.card), () => dsp({ t: 'transport', card: 1 })));
      const g3 = el('div', { class: 'grp' }, el('span', { class: 'lbl' }, 'Karten'));
      const wingsOk = has('wings');
      g3.append(
        btn('Brücke', has('bridge'), () => { this.mode = 'bridge'; this.bridgeFirst = null; this.closeSheet(); this.refresh(); }, this.mode === 'bridge' ? 'on' : '', CARD_HELP.bridge),
        btn('Blockade', has('blockage'), () => { this.mode = 'blockage'; this.closeSheet(); this.refresh(); }, this.mode === 'blockage' ? 'on' : '', CARD_HELP.blockage),
        btn('Flügel', wingsOk, () => { this.mode = 'wings'; this.closeSheet(); this.refresh(); }, this.mode === 'wings' ? 'on' : '', CARD_HELP.wings + (A.some((a) => a.t === 'wings' && a.via) ? ' (nutzt 2 andere Karten)' : '')),
        btn('Überlagern', has('super'), () => dsp({ t: 'super' }), '', CARD_HELP.superimpose));
      act.append(g1, g2, g3, el('div', { class: 'grp' }, btn('Zug beenden ▶', has('end'), () => { this.mode = null; dsp({ t: 'end' }); }, 'primary')));
    } else if (mine) act.append(el('span', { class: 'muted' }, 'Bitte im Dialog entscheiden.'));
    side.append(act);

    // cards
    const cardsBox = el('div', { class: 'card' });
    cardsBox.append(el('h3', {}, 'Ameisenkarten'));
    const cl = el('div', { class: 'cardlist' });
    me_p.cards.forEach((t) => cl.append(el('span', { class: 'acard', title: CARD_HELP[t] || '' }, CARD_NAMES[t])));
    if (!me_p.cards.length) cl.append(el('span', { class: 'muted' }, 'keine'));
    cardsBox.append(cl);
    side.append(cardsBox);

    // AI inspector
    if (c.lastAI) {
      const L = c.lastAI;
      const box = el('div', { class: 'card inspector' });
      box.append(el('h3', {}, `KI-Entscheidung (${s.players[L.seat].name}) · Wertschätzung V=${L.value.toFixed(2)}`));
      const order = [...L.probs.keys()].sort((a, b) => L.probs[b] - L.probs[a]).slice(0, 5);
      for (const i of order) {
        box.append(el('div', { class: 'bar' + (i === L.idx ? ' chosen' : '') },
          el('i', { style: `width:${(L.probs[i] * 100).toFixed(0)}%` }), el('span', {}, `${candidateText(L.acts[i], s)} · ${(L.probs[i] * 100).toFixed(0)}%`)));
      }
      side.append(box);
    }

    // log
    const lg = el('div', { class: 'card log' }, el('h3', {}, 'Verlauf'));
    const ul = el('div', { class: 'loglines' });
    this.logLines.slice(-14).reverse().forEach((l) => ul.append(el('div', {}, l)));
    lg.append(ul);
    const tools = el('div', { class: 'grp' });
    if (c.mode !== 'client') {
      const tg = (label, on, fn, title) => el('button', { class: on ? 'on' : '', title, onclick: () => { fn(); this.renderSide(); } }, label);
      tools.append(el('label', { class: 'speed', title: 'Tempo der KI-Züge' }, '🐢', el('input', { type: 'range', min: 150, max: 2000, value: 2150 - c.speed, oninput: (e) => (c.speed = 2150 - +e.target.value) }), '🐇'),
        tg('🎥 Kamera', c.cinematic, () => { c.cinematic = !c.cinematic; }, 'Kamerafahrten zu den KI-Aktionen'),
        tg('⏩ Schnell', c.fast, () => { c.fast = !c.fast; }, 'KI-Züge ohne Kamera und Pausen'),
        tg('⏸ Schritt', c.paused, () => c.setPaused(!c.paused), 'Schrittmodus: KI geht nur per Klick/Leertaste weiter'));
    }
    tools.append(el('button', { onclick: () => { if (confirm('Spiel verlassen?')) this.closeGame(); } }, 'Zum Menü'));
    lg.append(tools);
    side.append(lg);
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
      body.append(el('button', { class: 'primary', disabled: give == null || take == null, onclick: () => { close(); this.ctrl.dispatch({ t: 'swap', q, give, take }, actor(s)); } }, 'Tauschen'), el('button', { onclick: close }, 'Abbrechen'));
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
  rotate(d) { this.rot = (this.rot + d + 3) % 3; this.refresh(); }

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
        for (const c of this.pmap.keys()) b.cellColors.set(c, hi);
        break;
      }
      case 'move': case 'tmove': case 'wings': {
        const t = this.mode;
        this.dmap = new Map();
        for (const a of A) if (a.t === t) { const cid = Math.floor(a.to / 3); if (!this.dmap.has(cid)) this.dmap.set(cid, a); }
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
    if (this.pp && this.pmap?.has(this.pp.cid)) {
      b.ghost = { cell: this.pp.cid, tile: this.selTile, rot: this.pp.rots[this.pp.i], piece: this.mode === 'place' ? this.piece : 0, seat: this.viewSeat() };
    }
  }

  onBoardHover(cid) {
    if (!this.myTurn() || this.pp) return;
    const b = this.board;
    if ((this.mode === 'place' || this.mode === 'overlay') && this.pmap?.has(cid)) {
      const rots = this.pmap.get(cid);
      const rot = rots.includes(this.rot) ? this.rot : rots[0];
      b.ghost = { cell: cid, tile: this.selTile, rot, piece: this.mode === 'place' ? this.piece : 0, seat: this.viewSeat() };
    } else b.ghost = null;
    b.draw();
  }

  isMobile() { return window.matchMedia('(max-width: 860px)').matches; }
  closeSheet() { if (this.isMobile() && this.sheetOpen) { this.sheetOpen = false; } }

  hideCtx() { if (this.ctxMenu) this.ctxMenu.hidden = true; }

  openMenu(items, ev, title = '') {
    const m = this.ctxMenu;
    m.innerHTML = '';
    if (title) m.append(el('div', { class: 'ctx-title' }, title));
    for (const it of items) {
      m.append(el('button', { class: it.cls || '', disabled: it.disabled, onclick: (e) => { e.stopPropagation(); this.hideCtx(); it.fn(); } }, it.label));
    }
    if (!items.length) { m.hidden = true; return; }
    m.hidden = false;
    const r = this.boardWrap.getBoundingClientRect();
    const mw = m.offsetWidth, mh = m.offsetHeight;
    let x = (ev?.clientX ?? r.left + r.width / 2) - r.left + 8, y = (ev?.clientY ?? r.top + r.height / 2) - r.top + 8;
    x = Math.max(6, Math.min(x, r.width - mw - 6)); y = Math.max(6, Math.min(y, r.height - mh - 6));
    m.style.left = x + 'px'; m.style.top = y + 'px';
    this._menuEv = ev;
  }

  // confirm popover with ghost preview (touch-friendly placement)
  previewPlace(cid, ev) {
    const rots = this.pmap.get(cid);
    this.pp = { cid, rots, i: Math.max(0, rots.indexOf(this.rot)) };
    this.updateBoardMode(); this.board.draw();
    const show = () => {
      this.updateBoardMode(); this.board.draw();
      const over = this.mode === 'overlay';
      const pieces = over ? [] : [...new Set(this.acts.filter((a) => a.t === 'place' && a.cell === cid && a.tile === this.selTile).map((a) => a.piece))].sort();
      const pn = ['Nur Dreieck', '⛰ mit Hügel', '❔ mit Ressource'];
      this.openMenu([
        ...(pieces.length > 1 ? [{ label: `Beilage: ${pn[this.piece]} ›`, fn: () => { this.piece = pieces[(pieces.indexOf(this.piece) + 1) % pieces.length]; this.pmap = this.placeMap(); show(); } }] : []),
        { label: `✔ ${over ? 'Überlagern' : 'Hier legen'}`, cls: 'primary', fn: () => {
          const rot = this.pp.rots[this.pp.i]; this.pp = null;
          const a = over ? { t: 'overlay', cell: cid, tile: this.selTile, rot } : { t: 'place', cell: cid, tile: this.selTile, rot, piece: this.piece };
          this.ctrl.dispatch(a, actor(this.ctrl.state));
        } },
        ...(rots.length > 1 ? [{ label: '⟳ Drehen', fn: () => { this.pp.i = (this.pp.i + 1) % rots.length; this.rot = this.pp.rots[this.pp.i]; show(); } }] : []),
        { label: '✖ Abbrechen', fn: () => { this.pp = null; this.refresh(); } },
      ], ev);
    };
    show();
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
    const mv = A.find((a) => a.t === 'move' && Math.floor(a.to / 3) === cid);
    if (mv) items.push({ label: '🐜 Ameise hierher', fn: () => dsp(mv) });
    const tm = A.find((a) => a.t === 'tmove' && Math.floor(a.to / 3) === cid);
    if (tm) items.push({ label: '🎲 Ressource hierher bringen', fn: () => dsp(tm) });
    const wg = A.find((a) => a.t === 'wings' && Math.floor(a.to / 3) === cid);
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
    const touch = ev?.pointerType === 'touch';
    switch (this.mode) {
      case 'place': case 'overlay': {
        const rots = this.pmap?.get(cid);
        if (!rots) break;
        if (touch) return this.previewPlace(cid, ev);
        const rot = rots.includes(this.rot) ? this.rot : rots[0];
        if (this.mode === 'overlay') dsp({ t: 'overlay', cell: cid, tile: this.selTile, rot });
        else dsp({ t: 'place', cell: cid, tile: this.selTile, rot, piece: this.piece });
        return;
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
