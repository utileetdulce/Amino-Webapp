// Presentation layer: human-readable descriptions of actions plus camera/popup/dice hints.
import { TILE_TYPES, KINDS, score } from './engine.js';
import { nodeCell } from './engine.js';
import { KIND_ICON } from './render.js';
import { CARD_NAMES } from './text.js';

const nm = (s, i) => s.players[i].name;
const cellOf = (n) => Math.floor(n / 3);
const side = (v) => (v ? 'rechte' : 'linke');
const fist = (v) => (v ? '🤜' : '🤛');

function antCell(s, pi) { return s.players[pi].ant ? s.players[pi].ant.cell : null; }

// Describes an action BEFORE it is applied (state = pre-state). focus: cells the camera should show.
export function describeRich(s, a, pi) {
  const p = s.players[pi];
  const me = nm(s, pi);
  const from = antCell(s, pi);
  const r = (icon, text, sub = '', focus = [], extra = {}) => ({ icon, text, sub, focus: focus.filter((c) => c != null), ...extra });
  switch (a.t) {
    case 'inc_tiles': return r('🔺', `${me} nimmt ${Math.min(4, s.tileStack.length)} neue Dreiecke`, 'Einkommen: Die Hand füllt sich – damit lässt sich das Labyrinth erweitern.', [], { overview: true });
    case 'inc_card': return r('🃏', `${me} zieht eine Ameisenkarte`, 'Einkommen: Karten können Wege blockieren, Brücken bauen oder Kämpfe drehen.', [], { overview: true });
    case 'inc_hill': return r('⛰', `${me} nimmt einen Hügel zurück auf die Hand`, 'Der Hügel kann an anderer Stelle neu gebaut werden.', [a.cell]);
    case 'swap': return r('🔄', `${me} tauscht ein Dreieck mit ${nm(s, a.q)}`, `${TILE_TYPES[a.give].name} gegen ${TILE_TYPES[a.take].name} – der Gegner kann sich nicht wehren.`, [], { overview: true });
    case 'place': {
      const tn = TILE_TYPES[a.tile].name;
      if (a.piece === 1) return r('⛰', `${me} baut einen Ameisenhügel`, `Neues Dreieck (${tn})${p.ant ? ' mit zweitem Eingang' : ' – die Ameise erscheint'}. Zwischen Hügel und Ressourcen müssen 2 Dreiecke Platz bleiben.`, [a.cell], { zoom: 'tight' });
      if (a.piece === 2) return r('🌱', `${me} legt eine neue Ressource aus`, `Dreieck (${tn}) mit verdeckt gezogener Ressource – sie ist erst nächsten Zug transportierbar.`, [a.cell], { zoom: 'tight' });
      return r('🔺', `${me} verlängert das Labyrinth`, `Dreieck (${tn}) wird passend angelegt.`, [a.cell], { zoom: 'tight' });
    }
    case 'move': {
      const to = cellOf(a.to);
      const tc = s.cells[to];
      let why = 'Die Ameise läuft durch das Wegenetz.';
      if (tc.res >= 0) why = `Ziel: freie Ressource ${KIND_ICON[s.resources[tc.res].kind]} ${KINDS[s.resources[tc.res].kind]}.`;
      else if (tc.hill === pi) why = 'Zurück zum eigenen Hügel.';
      else if (tc.hill >= 0) why = `Unterwegs über den Hügel von ${nm(s, tc.hill)}.`;
      else {
        const enemy = s.players.find((q) => q.idx !== pi && q.ant && q.ant.cell === to);
        if (enemy) why = `${enemy.name}s Ameise ist dort${enemy.carry >= 0 ? ' und trägt eine Ressource' : ''}.`;
      }
      if (p.carry >= 0 && tc.hill !== pi) why += ' Sie trägt eine Ressource mit.';
      const under = from != null && s.cells[from].hill === pi && tc.hill === pi;
      const carry = p.carry >= 0 ? ` (trägt ${KIND_ICON[s.resources[p.carry].kind]} ${KINDS[s.resources[p.carry].kind]})` : '';
      const dest = tc.hill === pi ? ' zum eigenen Hügel' : tc.hill >= 0 ? ` über den Hügel von ${nm(s, tc.hill)}` : tc.res >= 0 ? ` zu ${KIND_ICON[s.resources[tc.res].kind]} ${KINDS[s.resources[tc.res].kind]}` : '';
      const log = under ? `${me} nutzt den Tunnel zwischen den Hügeln${carry}` : `${me} bewegt die Ameise${dest}${carry}${p.carry >= 0 && tc.hill === pi ? ' – noch nicht eingelagert, dafür braucht es einen Transport' : ''}`;
      return r('🐜', under ? `${me} nutzt den Tunnel zwischen den Hügeln` : `${me} bewegt die Ameise`, why, [from, to], { path: true, log });
    }
    case 'load': {
      const c = s.cells[from];
      const k = s.resources[c.res].kind;
      return r(KIND_ICON[k], `${me} lädt ${KINDS[k]} auf`, `Für ${me} ${p.values[k]} Punkt${p.values[k] > 1 ? 'e' : ''} wert. Nur ein Kampf kann sie jetzt noch abnehmen.`, [from], { zoom: 'tight' });
    }
    case 'unload': return r('⬇', `${me} legt die Ressource ab`, 'Die Ameise hat die Hände frei – z. B. um anzugreifen.', [from], { zoom: 'tight' });
    case 'fight': {
      const k = s.resources[s.players[a.q].carry].kind;
      return r('⚔', `${me} greift ${nm(s, a.q)} an!`, `Ziel: ${KIND_ICON[k]} ${KINDS[k]}. ${nm(s, a.q)} versteckt sie in einer Faust, ${me} muss die richtige erraten.`, [from], { zoom: 'tight' });
    }
    case 'hide': return r('🙈', `${me} versteckt die Ressource`, 'Eine Faust – die andere bleibt leer.', [antCell(s, s.pending.att)], { zoom: 'tight' });
    case 'guess': return r(fist(a.v), `${me} tippt auf die ${side(a.v)} Faust`, 'Gleich wird aufgedeckt …', [antCell(s, s.pending.att)], { zoom: 'tight' });
    case 'revenge': return r('🔥', `${me} spielt Rache!`, 'Der Kampfausgang wird aufgehoben, der Kampf beginnt von vorn.', [from]);
    case 'decline': return r('🤷', `${me} verzichtet auf Rache`, '', []);
    case 'transport': return r('🎲', `${me} transportiert Richtung Hügel`, a.card ? 'Mit Transport-Karte: Es wird mit zwei Würfeln gewürfelt.' : 'Der Würfel (0–5) zeigt, wie viele Dreiecke die Ressource vorankommt.', [from], { zoom: 'wide', dice: true });
    case 'tmove': {
      const to = cellOf(a.to);
      if (s.cells[to].hill === pi) return r('🏠', `${me} bringt die Ressource in den Hügel!`, 'Sie wird sicher eingelagert und zählt jetzt Punkte.', [from, to], { path: true });
      return r('🐜', `${me} schleppt die Ressource voran`, 'Noch nicht im Hügel – nächste Runde weiter.', [from, to], { path: true });
    }
    case 'tstay': return r('✋', `${me} bleibt stehen`, 'Der Wurf wird nicht genutzt.', [from]);
    case 'bridge': return r('🌉', `${me} baut eine Brücke`, 'Nur die eigene Ameise kann sie überqueren; sie kehrt nach dem nächsten Zug zurück.', [a.a, a.b], { zoom: 'tight' });
    case 'blockage': return r('⛔', `${me} blockiert ein Dreieck`, 'Bis zum nächsten Zug von ' + me + ' kommt keine Ameise vorbei.', [a.cell], { zoom: 'tight' });
    case 'wings': return r('🪽', `${me} fliegt über das Spielfeld`, a.via ? 'Zwei Karten wurden zu Flügeln getauscht. Eine getragene Ressource bleibt zurück.' : 'Flügel-Karte: Die Ameise landet auf einem beliebigen Dreieck.', [from, cellOf(a.to)], { path: true });
    case 'super': return r('🧩', `${me} spielt „Überlagern“`, 'Dreiecke werden über bestehende Felder gelegt und verändern die Wege.', [], { overview: true });
    case 'overlay': return r('🧩', `${me} überlagert ein Feld`, `Neues Dreieck (${TILE_TYPES[a.tile].name}) liegt jetzt über dem alten.`, [a.cell], { zoom: 'tight' });
    case 'sdone': return r('✅', `${me} beendet das Überlagern`, '', [], { overview: true });
    case 'sundo': return r('↺', `${me} bricht das Überlagern ab`, '', [], { overview: true });
    case 'end': {
      const nx = s.players[(pi + 1) % s.n];
      return r('⏭', `${me} beendet den Zug`, `${nx.name} ist dran.`, [], { overview: true, nextSeat: nx.idx, quick: true });
    }
  }
  return r('•', `${me}: ${a.t}`);
}

// Result info AFTER the action was applied. before = {scores:[...]} captured before apply.
export function describeResult(s, a, pi, before) {
  const out = {};
  const me = nm(s, pi);
  if (a.t === 'place' && a.piece === 2) {
    const res = s.resources[s.resources.length - 1];
    out.sub = `Gezogen: ${KIND_ICON[res.kind]} ${KINDS[res.kind]}`;
  }
  if (a.t === 'inc_card') { const cs = s.players[pi].cards; out.card = cs[cs.length - 1]; }
  if (a.t === 'transport') {
    out.dice = { values: s.lastRoll, card: !!a.card, seat: pi, cell: antCell(s, pi) };
    const sum = (s.lastRoll[0] || 0) + (s.lastRoll[1] || 0);
    if (before && score(s, pi) > before.scores[pi]) out.sub = `Gewürfelt: ${sum} – Ressource im Hügel eingelagert`;
    else out.sub = sum ? `Gewürfelt: ${s.lastRoll[1] != null ? s.lastRoll[0] + ' + ' + s.lastRoll[1] + ' = ' : ''}${sum}` : 'Gewürfelt: 0 – leider nichts bewegt.';
  }
  if (before) {
    const gain = score(s, pi) - before.scores[pi];
    if (gain > 0) out.big = { icon: '🏠', title: `+${gain} Punkt${gain > 1 ? 'e' : ''} für ${me}`, sub: `${me}: ${score(s, pi)} / 7`, kind: 'score', color: s.players[pi].color };
  }
  if ((a.t === 'guess' || a.t === 'decline') && s.lastFight && !s.pending) {
    const f = s.lastFight;
    out.big = fightBig(s, f);
  }
  if (a.t === 'guess' && s.pending && s.pending.type === 'revenge') {
    out.big = fightBig(s, { att: s.pending.att, def: s.pending.def, attWin: s.pending.attWin, hide: s.pending.hide, guess: s.pending.guessV });
    out.big.sub += ' Rache möglich!';
  }
  if (s.over) out.big = { icon: '🏆', title: s.winner < 0 ? 'Unentschieden' : `${nm(s, s.winner)} gewinnt!`, sub: s.players.map((q) => `${q.name} ${score(s, q.idx)}`).join(' · '), kind: 'win', color: s.winner >= 0 ? s.players[s.winner].color : '#888' };
  return out;
}

function fightBig(s, f) {
  const at = nm(s, f.att), de = nm(s, f.def);
  return {
    icon: f.attWin ? '⚔' : '🛡',
    title: f.attWin ? `${at} trifft – Ressource erbeutet!` : `${de} verteidigt erfolgreich!`,
    sub: `${de} hatte sie in der ${side(f.hide)} Faust ${fist(f.hide)}, ${at} tippte auf die ${side(f.guess)} ${fist(f.guess)}.`,
    kind: 'fight', color: s.players[f.attWin ? f.att : f.def].color, fists: [f.hide, f.guess],
  };
}
