// Writes models/index.json: the list of shipped models with their metadata summary (the app reads this).
import fs from 'node:fs';
import { summary } from '../js/modelmeta.js';
const notes = fs.existsSync('models/notes.json') ? JSON.parse(fs.readFileSync('models/notes.json', 'utf8')) : {}; // overrides (note/status) without touching model files
const entries = [];
for (const f of fs.readdirSync('models').filter((x) => x.endsWith('.json') && x !== 'index.json' && x !== 'notes.json').sort()) {
  const j = JSON.parse(fs.readFileSync('models/' + f, 'utf8'));
  entries.push({ file: f, ...summary(j, f.replace(/\.json$/, '')), ...(notes[f] || {}) });
}
fs.writeFileSync('models/index.json', JSON.stringify({ updated: new Date().toISOString(), models: entries }, null, 1));
console.log(entries.map((e) => `${e.file}: ${e.name} · ${e.kind} · rules ${e.rulesVersion} · ${e.games} games${e.default ? ' · DEFAULT' : ''}`).join('\n'));
