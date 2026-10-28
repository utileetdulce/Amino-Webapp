// Original Amino graphics, cut from the print-and-play PDFs by tools/extract-print-assets.py.
// Images load in the background; until then (or if one fails) the drawn fallback is used.
import { TILE_ART } from './tile-art.js';

const listeners = new Set();
function load(src) {
  if (typeof Image === 'undefined') return {}; // no DOM (node tools)
  const im = new Image();
  im.onload = () => { im.ok = true; for (const f of listeners) f(); };
  im.src = src;
  return im;
}
export const ART = {
  tiles: load('assets/tiles.webp'),
  res: [0, 1, 2].map((k) => load(`assets/res-${k}.png`)), // Blume, Pilz, Raupe
  ant: load('assets/ant.png'),
  hill: load('assets/hill.png'),
  texture: load('assets/texture.webp'), // painted back of the landscape triangles, tinted green
};
export function onArtLoad(fn) { listeners.add(fn); return () => listeners.delete(fn); }

// painted triangles per TILE_TYPES index (only Y, Kurve and Sackgasse exist in print)
export const TILE_ART_BY_TYPE = [];
for (const a of TILE_ART) (TILE_ART_BY_TYPE[a.t] = TILE_ART_BY_TYPE[a.t] || []).push(a);

export const cardSrc = (type) => `assets/card-${type}.webp`;
export const CARD_BACK = 'assets/card-back.webp';
// the Koloniekarte whose chambers match a player's resource values (values[k] = points of KINDS[k])
export const colonySrc = (values) => `assets/colony-${values.join('')}.webp`;
export const resSrc = (k) => `assets/res-${k}.png`;

// hill dome tinted in a player colour (cached per colour)
const hills = new Map();
export function hillSprite(color) {
  if (!ART.hill.ok) return null;
  let c = hills.get(color);
  if (c) return c;
  const im = ART.hill;
  c = document.createElement('canvas');
  c.width = im.width; c.height = im.height;
  const g = c.getContext('2d');
  g.drawImage(im, 0, 0);
  g.globalCompositeOperation = 'multiply';
  g.fillStyle = color; g.fillRect(0, 0, c.width, c.height);
  g.globalCompositeOperation = 'destination-in';
  g.drawImage(im, 0, 0);
  hills.set(color, c);
  return c;
}
