// Training runs off the main thread so the UI stays responsive.
import { Trainer } from './trainer.js';
import { Net } from './nn.js';

let tr = null, running = false, evalEvery = 20, nextEval = 0, token = 0;
const history = { evalPts: [] };

function postNet(type, extra = {}) {
  self.postMessage({ type, theta: tr.net.theta.slice(), arch: tr.net.arch, ...extra });
}

async function loop(my) {
  const t0 = performance.now();
  let g0 = tr.games;
  while (running && my === token) {
    const st = tr.iterate();
    const dt = (performance.now() - t0) / 1000;
    postNet('iter', { stats: st, gamesPerSec: (tr.games - g0) / dt });
    if (st.iter >= nextEval && evalEvery > 0) {
      const ev = tr.evalAll(20);
      nextEval = st.iter + evalEvery;
      self.postMessage({ type: 'eval', iter: st.iter, games: st.games, random: ev.random.rate, heuristic: ev.heuristic.rate, ev });
    }
    await new Promise((r) => setTimeout(r, 0));
  }
  self.postMessage({ type: 'state', running: false });
}

self.onmessage = (e) => {
  const m = e.data;
  switch (m.cmd) {
    case 'init': {
      token++; running = false;
      const net = m.net ? Net.fromJSON(m.net) : new Net(m.arch);
      tr = new Trainer({ net, cfg: m.cfg, opponent: m.opponent, episodesPerIter: m.eps });
      if (m.games) tr.games = m.games;
      if (m.iter) tr.iter = m.iter;
      nextEval = tr.iter;
      postNet('ready');
      break;
    }
    case 'cfg':
      Object.assign(tr.cfg, m.cfg);
      if (m.opponent) tr.opponent = m.opponent;
      if (m.eps) tr.episodesPerIter = m.eps;
      if (m.evalEvery != null) evalEvery = m.evalEvery;
      break;
    case 'start':
      if (!running) { running = true; token++; self.postMessage({ type: 'state', running: true }); loop(token); }
      break;
    case 'pause': running = false; token++; break;
    case 'clone': {
      const wasRunning = running; running = false; token++;
      const r = tr.clone(m.games || 10, 3);
      self.postMessage({ type: 'clone', loss: r.loss, demos: r.demos });
      postNet('ready');
      if (wasRunning) { running = true; loop(++token); }
      break;
    }
    case 'eval': {
      const ev = tr.evalAll(m.games || 30);
      self.postMessage({ type: 'eval', iter: tr.iter, games: tr.games, random: ev.random.rate, heuristic: ev.heuristic.rate, ev });
      break;
    }
    case 'snapshot': postNet('snapshot'); break;
    default: break;
  }
};
