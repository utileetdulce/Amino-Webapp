// Metadata attached to every trained model: who/when/how it was trained and under which rules version.
import { RULES_VERSION, RULES_CHANGELOG } from './engine.js';
export { RULES_VERSION, RULES_CHANGELOG };

export const KIND_LABEL = { ppo: 'PPO', az: 'AlphaZero (Suche)', mcts: 'AlphaZero (Suche)', imitation: 'Imitation' };
export const nowIso = () => new Date().toISOString();
export const newModelId = () => 'm-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 6);

export function makeMeta({ name = 'Modell', kind = 'ppo', trainer = 'js', parent = null, hyper = {}, trainMinutes = 0, host = null, git = null, note = '' } = {}) {
  const t = nowIso();
  return { id: newModelId(), name, kind, trainer, createdAt: t, updatedAt: t, rulesVersion: RULES_VERSION, parent, hyper, trainMinutes, host, git, note };
}

// Carry on an existing meta (resume) or start a new one that points to its parent.
export function childMeta(parentModel, opts) {
  const pm = parentModel?.meta;
  const base = makeMeta({ ...opts, parent: pm ? { id: pm.id, name: pm.name, updatedAt: pm.updatedAt, rulesVersion: pm.rulesVersion, games: parentModel.games } : null });
  return base;
}

export function touchMeta(meta, extra = {}) {
  return { ...meta, ...extra, updatedAt: nowIso(), rulesVersion: RULES_VERSION };
}

const fmtDate = (iso) => {
  if (!iso) return '—';
  const d = new Date(iso);
  return Number.isNaN(+d) ? '—' : d.toLocaleString('de-DE', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' });
};
const lastEval = (m) => (m.history && m.history.length ? m.history[m.history.length - 1] : null);

export function summary(m, fallbackName = 'Modell') {
  const me = m.meta || {};
  const ev = lastEval(m);
  return {
    id: me.id || null, name: me.name || m.name || fallbackName, kind: me.kind || (m.kind === 'mcts' ? 'az' : 'ppo'), trainer: me.trainer || null,
    createdAt: me.createdAt || null, updatedAt: me.updatedAt || null, rulesVersion: me.rulesVersion ?? null,
    games: m.games || 0, iter: m.iter || 0, note: me.note || '', status: me.status || '', default: !!me.default,
    heuristicRate: ev ? (ev.rawHeuristic != null && me.kind === 'az' ? ev.heuristic : ev.heuristic) : null, rawRate: ev?.rawHeuristic ?? null, randomRate: ev && ev.rawHeuristic == null ? ev.random : null,
  };
}

export function rulesStatus(sum) {
  if (sum.rulesVersion == null) return { ok: false, level: 'warn', text: 'Regelversion unbekannt (Modell ohne Metadaten)' };
  if (sum.rulesVersion < RULES_VERSION) return { ok: false, level: 'warn', text: `trainiert mit Regelversion ${sum.rulesVersion} (aktuell ${RULES_VERSION})` };
  return { ok: true, level: 'ok', text: `Regelversion ${sum.rulesVersion} (aktuell)` };
}

export function modelLabel(sum) {
  const parts = [sum.name, KIND_LABEL[sum.kind] || sum.kind];
  if (sum.updatedAt) parts.push(new Date(sum.updatedAt).toLocaleDateString('de-DE'));
  if (sum.games) parts.push(`${sum.games.toLocaleString('de-DE')} Spiele`);
  let l = parts.join(' · ');
  if (sum.default) l += ' ★';
  if (sum.status === 'weak') l += ' · schwach';
  if (!rulesStatus(sum).ok) l += ' ⚠';
  return l;
}

// rows for an info table
export function modelDetails(m) {
  const me = m.meta || {};
  const sum = summary(m);
  const rs = rulesStatus(sum);
  const pct = (x) => (x == null ? '—' : Math.round(x * 100) + ' %');
  const rows = [
    ['Name', sum.name], ['Typ', KIND_LABEL[sum.kind] || sum.kind], ['Trainer', me.trainer || '—'],
    ['Erstellt', fmtDate(me.createdAt)], ['Zuletzt trainiert', fmtDate(me.updatedAt)],
    ['Spiele / Iterationen', `${(m.games || 0).toLocaleString('de-DE')} / ${m.iter || 0}`], ['Trainingsdauer', me.trainMinutes ? `${Math.round(me.trainMinutes)} min` : '—'],
    ['Regeln', rs.text + (rs.ok ? '' : ' ⚠')],
    ['Parameter', m.net ? m.net.theta.length.toLocaleString('de-DE') : '—'],
    ['Herkunft', me.parent ? `weitertrainiert von „${me.parent.name}“ (${fmtDate(me.parent.updatedAt)})` : (me.id ? 'neu gestartet / unbekannt' : '—')],
    ['Letzte Auswertung', (() => { const ev = lastEval(m); if (!ev) return '—'; return m.meta?.kind === 'az' || m.kind === 'mcts' ? `Netz allein ${pct(ev.rawHeuristic)} · mit Suche ${pct(ev.heuristic)} vs Heuristik (Iter. ${ev.iter})` : `${pct(ev.random)} vs Zufall · ${pct(ev.heuristic)} vs Heuristik (Iter. ${ev.iter})`; })()],
  ];
  if (me.hyper && Object.keys(me.hyper).length) rows.push(['Hyperparameter', Object.entries(me.hyper).map(([k, v]) => `${k}=${v}`).join(', ')]);
  if (me.host) rows.push(['Rechner', typeof me.host === 'string' ? me.host : `${me.host.os || ''} ${me.host.arch || ''} · ${me.host.cpus || '?'} Kerne`]);
  if (me.git) rows.push(['Code-Stand', me.git]);
  if (me.id) rows.push(['Modell-ID', me.id]);
  if (me.note) rows.push(['Hinweis', me.note]);
  return rows;
}
