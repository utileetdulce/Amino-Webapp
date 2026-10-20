// Training loop shared by the browser worker and the CLI.
import { Net, DEFAULT_ARCH } from './nn.js';
import { collectEpisode, computeGAE, ppoUpdate, DEFAULT_PPO, collectHeuristicDemos, cloneUpdate, evaluate } from './selfplay.js';

export class Trainer {
  constructor({ net = null, cfg = {}, opponent = 'mixed', episodesPerIter = 12 } = {}) {
    this.net = net || new Net(DEFAULT_ARCH);
    this.cfg = { ...DEFAULT_PPO, ...cfg };
    this.opponent = opponent;
    this.episodesPerIter = episodesPerIter;
    this.iter = 0;
    this.games = 0;
    this.pool = []; // past snapshots
    this.recent = [];
  }
  pickOpponent() {
    if (this.opponent !== 'mixed') return this.opponent;
    const r = Math.random();
    if (r < 0.4) return 'self';
    if (r < 0.7) return 'heuristic';
    if (r < 0.8) return 'random';
    return this.pool.length ? 'past' : 'self';
  }
  clone(nGames = 10, epochs = 3) {
    const demos = collectHeuristicDemos(nGames);
    const loss = cloneUpdate(this.net, demos, { epochs });
    return { loss, demos: demos.length };
  }
  iterate() {
    const steps = [];
    let wins = 0, losses = 0, draws = 0, games = 0, turns = 0, ret = 0, retN = 0;
    const byOpp = {};
    for (let e = 0; e < this.episodesPerIter; e++) {
      const opp = this.pickOpponent();
      const oppNet = opp === 'past' ? this.pool[Math.floor(Math.random() * this.pool.length)] : null;
      const ep = collectEpisode(this.net, { opponent: opp, shape: this.cfg.shape, oppNet });
      for (const T of ep.trajs) { computeGAE(T, this.cfg.gamma, this.cfg.lam); ret += T.reduce((a, t) => a + t.r, 0); retN++; }
      for (const t of ep.steps) steps.push(t);
      games++; turns += ep.turns;
      const o = (byOpp[opp] = byOpp[opp] || { w: 0, l: 0, d: 0 });
      if (ep.winner < 0) { draws++; o.d++; } else if (ep.learnerSeats.includes(ep.winner)) { if (opp !== 'self') { wins++; o.w++; } } else { losses++; o.l++; }
    }
    const upd = steps.length ? ppoUpdate(this.net, steps, this.cfg) : {};
    this.iter++; this.games += games;
    if (this.iter % 10 === 0) { this.pool.push(this.net.clone()); if (this.pool.length > 8) this.pool.shift(); }
    return { iter: this.iter, games: this.games, steps: steps.length, meanReturn: ret / Math.max(1, retN), avgTurns: turns / games, wins, losses, draws, byOpp, ...upd };
  }
  evalAll(games = 20) {
    return { random: evaluate(this.net, 'random', games), heuristic: evaluate(this.net, 'heuristic', games) };
  }
}
