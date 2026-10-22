// Game controller: owns the state, seats (human / AI / remote), AI stepping and network sync.
import { newGame, applyAction, legalActions, actor, actionKey, viewFor, score } from './engine.js';
import { describeRich, describeResult } from './present.js';
import { RandomAgent, HeuristicAgent, NNAgent } from './agents.js';

export class GameController {
  // seats: [{kind:'human'|'ai'|'remote', ai:'random'|'heuristic'|'nn', name, link?}]
  constructor({ seats, nets = {}, mode = 'local', seed }) {
    this.mode = mode; // 'local' | 'host' | 'client'
    this.seats = seats;
    this.nets = nets; // name -> Net
    this.listeners = [];
    this.speed = 800; // ms base for AI pacing
    this.paused = false; // step mode: AI only advances on click
    this.fast = false;
    this.cinematic = true;
    this.waiting = null;
    this.mySeat = -1; // client mode
    this.timer = null;
    this.lastAI = null; // info for inspector
    if (mode !== 'client') {
      this.state = newGame({ numPlayers: seats.length, names: seats.map((s) => s.name), seed });
      this.agents = seats.map((s) => this.makeAgent(s));
      setTimeout(() => this.schedule(), 60);
    }
  }
  makeAgent(seat) {
    if (seat.kind !== 'ai') return null;
    if (seat.ai === 'random') return new RandomAgent();
    if (seat.ai === 'nn' && this.nets[seat.model]) return new NNAgent(this.nets[seat.model], { sample: true, temperature: seat.temp ?? 0.5 });
    return new HeuristicAgent();
  }
  on(fn) { this.listeners.push(fn); }
  emit(extra) { for (const f of this.listeners) f(this.state, extra); }
  isLocalSeat(i) {
    if (this.mode === 'client') return i === this.mySeat;
    return this.seats[i].kind === 'human';
  }
  currentActor() { return actor(this.state); }

  // human (or remote) action
  dispatch(a, seat) {
    const s = this.state;
    if (this.mode === 'client') { this.link.send({ type: 'action', action: a }); return true; }
    if (s.over || actor(s) !== seat) return false;
    const k = actionKey(a);
    if (!legalActions(s, true).some((x) => actionKey(x) === k)) { console.warn('illegal', a); return false; }
    this.apply(a, seat);
    return true;
  }
  // several actions in a row (e.g. auto-move + attack); stops at the first illegal one
  dispatchSeq(list, seat) {
    if (this.mode === 'client') { this.link.send({ type: 'actions', actions: list }); return true; }
    for (const a of list) if (!this.dispatch(a, seat)) return false;
    return true;
  }
  apply(a, seat, pre = null) {
    const s = this.state;
    pre = pre || describeRich(s, a, seat);
    const before = { scores: s.players.map((_, i) => score(s, i)) };
    applyAction(s, a);
    const post = describeResult(s, a, seat, before);
    this.emit({ type: 'applied', action: a, seat, pre, post });
    if (this.mode === 'host') this.broadcast();
    if (this.seats[actor(s)]?.kind === 'ai') this.schedule();
    return { pre, post };
  }
  broadcast() {
    this.seats.forEach((st, i) => { if (st.kind === 'remote' && st.link) st.link.send({ type: 'state', state: viewFor(this.state, i) }); });
  }
  // ---- AI pacing: announce (popup + camera) -> apply -> hold; clicks skip the current wait ----
  wait(ms) {
    return new Promise((res) => {
      const w = { res, timer: null };
      this.waiting = w;
      const done = () => { clearTimeout(w.timer); if (this.waiting === w) this.waiting = null; res(); };
      w.done = done;
      if (!this.paused) w.timer = setTimeout(done, ms);
    });
  }
  skip() { this.waiting?.done(); }
  setPaused(p) {
    this.paused = p;
    if (!p) { this.waiting?.done(); this.schedule(); } else if (this.waiting) clearTimeout(this.waiting.timer);
  }
  schedule() {
    const s = this.state;
    if (!s || s.over || this.running) return;
    if (this.seats[actor(s)]?.kind !== 'ai') return;
    this.running = true;
    this.stepAI().finally(() => { this.running = false; if (!this.destroyed && !this.state.over && this.seats[actor(this.state)]?.kind === 'ai') this.schedule(); });
  }
  async stepAI() {
    const s = this.state;
    const pi = actor(s);
    const ag = this.agents[pi];
    const info = {};
    const a = ag instanceof NNAgent ? ag.act(s, info) : ag.act(s);
    this.lastAI = info.acts ? { seat: pi, ...info } : null;
    const pre = describeRich(s, a, pi);
    const humans = this.seats.some((x) => x.kind !== 'ai');
    const f = this.fast ? 0.12 : humans ? 1 : 0.5;
    const quick = pre.quick ? 0.55 : 1;
    this.emit({ type: 'announce', seat: pi, action: a, pre });
    await this.wait(Math.max(60, this.speed * 0.95 * f * quick));
    if (this.destroyed) return;
    const { post } = this.apply(a, pi, pre);
    const extra = this.fast ? 0 : (post.dice ? 1900 : 0) + (post.big ? 1700 : 0);
    await this.wait(Math.max(40, this.speed * 0.5 * f * quick, extra));
  }
  // client: install state from host
  setRemoteState(st) {
    const prev = this.state?.log?.length ? this.state.log[this.state.log.length - 1] : null;
    this.state = st;
    const last = st.log.length ? st.log[st.log.length - 1] : null;
    if (last && JSON.stringify(last) !== JSON.stringify(prev)) {
      const pre = describeRich(st, last.a, last.p);
      this.emit({ type: 'applied', action: last.a, seat: last.p, pre, post: describeResult(st, last.a, last.p, null), remote: true });
    } else this.emit({});
  }
  receiveFromClient(seat, msg) {
    if (msg.type === 'action') this.dispatch(msg.action, seat);
    else if (msg.type === 'actions') this.dispatchSeq(msg.actions, seat);
    else if (msg.type === 'hello') this.broadcast();
  }
  destroy() { this.destroyed = true; this.waiting?.done(); this.listeners = []; }
}
