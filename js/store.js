// Model persistence (localStorage) + import/export helpers.
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
