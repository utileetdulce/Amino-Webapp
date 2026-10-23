// One-off: attach metadata to models that were trained before metadata existed (dates from git history).
import fs from 'node:fs';
const T = {
  'models/pretrained.json': { name: 'PPO (Regelversion 3)', kind: 'ppo', trainer: 'js-cli', createdAt: '2026-10-22T17:15:00Z', updatedAt: '2026-10-22T18:04:00Z', rulesVersion: 3, trainMinutes: 45,
    hyper: { lr: 0.0003, gamma: 0.995, lam: 0.95, clip: 0.2, epochs: 4, mb: 96, entCoef: 0.01, shape: 0.1, eps: 12, opponent: 'mixed' }, host: { os: 'linux', arch: 'x64', cpus: 4 },
    note: 'PPO mit Imitations-Warmstart; ca. 84 % gegen die Heuristik. Ohne Anti-Schleifen-Filter trainiert.' },
  'models/az.json': { name: 'AlphaZero M4 (zusammengebrochen)', kind: 'az', trainer: 'js/rust (MacBook M4)', createdAt: '2026-10-23T18:00:00Z', updatedAt: '2026-10-23T20:58:03Z', rulesVersion: 3, trainMinutes: null,
    hyper: { sims: 96 }, status: 'weak', note: 'Trainingslauf mit Besuchszahl-Zielen; Spielstärke eingebrochen (9–19 % gegen die Heuristik). Nicht empfohlen.' },
  'models/az.best.json': { name: 'AlphaZero M4 (früher Stand)', kind: 'az', trainer: 'js/rust (MacBook M4)', createdAt: '2026-10-23T18:00:00Z', updatedAt: '2026-10-23T19:00:00Z', rulesVersion: 3, trainMinutes: null,
    hyper: { sims: 96 }, status: 'weak', note: 'Bester Zwischenstand des M4-Laufs nach 5 Iterationen (ca. 34–41 % gegen die Heuristik). Nicht empfohlen.' },
};
for (const [f, m] of Object.entries(T)) {
  if (!fs.existsSync(f)) continue;
  const j = JSON.parse(fs.readFileSync(f, 'utf8'));
  if (j.meta) { console.log('skip (has meta)', f); continue; }
  j.meta = { id: 'm-legacy-' + f.replace(/\W+/g, '-'), parent: null, git: null, ...m };
  j.version = 3; j.name = m.name;
  if (f.endsWith('pretrained.json')) j.meta.default = true;
  fs.writeFileSync(f, JSON.stringify(j));
  console.log('backfilled', f);
}
