// Model persistence (localStorage) + import/export helpers.
import { summary } from './modelmeta.js';
const KEY = 'amino.models.v1';

function readAll() {
  try { return JSON.parse(localStorage.getItem(KEY) || '{}'); } catch { return {}; }
}
export function listModels() { return Object.keys(readAll()); }
export function loadModel(name) { return readAll()[name] || null; }
export function saveModel(name, obj) {
  const all = readAll();
  all[name] = obj;
  try { localStorage.setItem(KEY, JSON.stringify(all)); return true; } catch (e) { console.warn(e); return false; }
}
export function deleteModel(name) {
  const all = readAll(); delete all[name];
  try { localStorage.setItem(KEY, JSON.stringify(all)); } catch { /* ignore */ }
}
let _pre = null;
export async function loadPretrained() {
  if (_pre) return _pre;
  try {
    const r = await fetch('models/pretrained.json');
    if (r.ok) _pre = await r.json();
  } catch { /* offline / missing */ }
  return _pre;
}
export function downloadJSON(name, obj) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(obj)], { type: 'application/json' }));
  a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

// ---- model catalogue: shipped models (models/index.json) + models saved in this browser ----
const _cache = new Map();
export async function loadIndex() {
  try { const r = await fetch('models/index.json', { cache: 'no-cache' }); if (r.ok) return (await r.json()).models || []; } catch { /* offline */ }
  return [];
}
export async function listAllModels() {
  let idx = await loadIndex();
  if (!idx.length) { const p = await loadPretrained(); if (p) idx = [{ file: 'pretrained.json', ...summary(p, 'Vortrainiert'), default: true }]; }
  const shipped = idx.map((e) => ({ ...e, key: 'file:' + e.file, source: 'file' }))
    .sort((a, b) => (b.default ? 1 : 0) - (a.default ? 1 : 0) || (a.status === 'weak') - (b.status === 'weak') || (b.updatedAt || '').localeCompare(a.updatedAt || ''));
  const local = listModels().map((n) => ({ ...summary(loadModel(n), n), name: n, key: 'local:' + n, source: 'local' }));
  return [...shipped, ...local];
}
export async function fetchModel(key) {
  if (key === 'Vortrainiert') key = 'file:pretrained.json';
  if (key.startsWith('file:')) {
    const f = key.slice(5);
    if (!_cache.has(f)) { const r = await fetch('models/' + f); if (!r.ok) return null; _cache.set(f, await r.json()); }
    return _cache.get(f);
  }
  return loadModel(key.startsWith('local:') ? key.slice(6) : key);
}
