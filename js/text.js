import { TILE_TYPES, KINDS } from './engine.js';
import { cellX, cellY } from './geometry.js';

export const CARD_NAMES = { bridge: 'Brücke', superimpose: 'Überlagern', blockage: 'Blockade', wings: 'Flügel', revenge: 'Rache', transport: 'Transport', x: '?' };
export const CARD_HELP = {
  bridge: 'Verbindet zwei benachbarte Dreiecke, nur du kannst sie überqueren. Kehrt am Ende deines nächsten Zuges zurück.',
  superimpose: 'Lege Dreiecke aus der Hand über freie, noch nicht überlagerte Felder und verändere so das Wegenetz.',
  blockage: 'Sperre ein Dreieck bis zu deinem nächsten Zug. Keine Ameise darf drüber oder darauf.',
  wings: 'Fliege sofort zu einem beliebigen Dreieck (ohne Ressource). Auch durch Tausch von 2 beliebigen Karten.',
  revenge: 'Verlorenen Kampf aufheben und sofort wiederholen.',
  transport: 'Würfle beim Ressourcen-Transport mit zwei statt einem Würfel.',
};

const coord = (c) => `(${cellX(c)},${cellY(c)})`;

export function describeAction(s, a, pi) {
  const name = s.players[pi]?.name ?? '';
  switch (a.t) {
    case 'inc_tiles': return `${name} nimmt 4 Dreiecke`;
    case 'inc_card': return `${name} zieht eine Ameisenkarte`;
    case 'inc_hill': return `${name} nimmt einen Hügel zurück`;
    case 'swap': return `${name} tauscht ein Dreieck mit ${s.players[a.q].name}`;
    case 'place': return `${name} legt ein Dreieck${a.piece === 1 ? ' mit Hügel' : a.piece === 2 ? ' mit Ressource' : ''}`;
    case 'move': return `${name} bewegt die Ameise`;
    case 'load': return `${name} lädt eine Ressource auf`;
    case 'unload': return `${name} lädt die Ressource ab`;
    case 'fight': return `${name} greift ${s.players[a.q].name} an`;
    case 'hide': return `${name} versteckt die Ressource`;
    case 'guess': return `${name} tippt auf eine Faust`;
    case 'revenge': return `${name} spielt Rache – Kampf wird wiederholt`;
    case 'decline': return `${name} verzichtet auf Rache`;
    case 'transport': return `${name} transportiert${a.card ? ' (2 Würfel)' : ''}`;
    case 'tmove': return `${name} bringt die Ressource voran`;
    case 'tstay': return `${name} bleibt stehen`;
    case 'bridge': return `${name} baut eine Brücke`;
    case 'blockage': return `${name} legt eine Blockade`;
    case 'wings': return `${name} fliegt`;
    case 'super': return `${name} spielt Überlagern`;
    case 'overlay': return `${name} überlagert ein Feld`;
    case 'sdone': return `${name} beendet das Überlagern`;
    case 'sundo': return `${name} bricht das Überlagern ab`;
    case 'end': return `${name} beendet den Zug`;
  }
  return a.t;
}

export const tileName = (t) => TILE_TYPES[t].name;
export const kindName = (k) => KINDS[k];

// Short description of a candidate action (for inspector lists)
export function candidateText(a, s) {
  switch (a.t) {
    case 'place': return `Dreieck ${TILE_TYPES[a.tile].name} ${coord(a.cell)}${a.piece === 1 ? ' + Hügel' : a.piece === 2 ? ' + Ressource' : ''}`;
    case 'move': return `Ameise → ${coord(Math.floor(a.to / 3))}`;
    case 'tmove': return `Transport → ${coord(Math.floor(a.to / 3))}`;
    case 'wings': return `Flügel → ${coord(Math.floor(a.to / 3))}${a.via ? ' (2 Karten)' : ''}`;
    case 'bridge': return `Brücke ${coord(a.a)}–${coord(a.b)}`;
    case 'blockage': return `Blockade ${coord(a.cell)}`;
    case 'overlay': return `Überlagern ${coord(a.cell)} mit ${TILE_TYPES[a.tile].name}`;
    case 'swap': return `Tausche ${TILE_TYPES[a.give].name} ↔ ${TILE_TYPES[a.take].name}`;
    case 'hide': case 'guess': return (a.t === 'hide' ? 'Verstecke in ' : 'Tippe auf ') + (a.v ? 'rechte' : 'linke') + ' Faust';
    case 'inc_tiles': return 'Einkommen: 4 Dreiecke';
    case 'inc_card': return 'Einkommen: Ameisenkarte';
    case 'inc_hill': return 'Einkommen: Hügel zurück';
    case 'load': return 'Ressource aufladen';
    case 'unload': return 'Ressource abladen';
    case 'fight': return 'Angreifen';
    case 'transport': return a.card ? 'Transport (2 Würfel)' : 'Transport';
    case 'tstay': return 'Stehen bleiben';
    case 'super': return 'Karte: Überlagern';
    case 'sdone': return 'Überlagern beenden';
    case 'sundo': return 'Überlagern abbrechen';
    case 'revenge': return 'Rache spielen';
    case 'decline': return 'Keine Rache';
    case 'end': return 'Zug beenden';
  }
  return a.t;
}
