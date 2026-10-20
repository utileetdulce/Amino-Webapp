// Game controller: owns the state, seats (human / AI / remote), AI stepping and network sync.
import { newGame, applyAction, legalActions, actor, actionKey, viewFor, cloneState } from './engine.js';
import { RandomAgent, HeuristicAgent, NNAgent } from './agents.js';

export class GameController {
  // seats: [{kind:'human'|'ai'|'remote', ai:'random'|'heuristic'|'nn', name, link?}]
  constructor({ seats, nets = {}, mode = 'local', seed }) {
    this.mode = mode; // 'local' | 'host' | 'client'
    this.seats = seats;
    this.nets = nets; // name -> Net
    this.listeners = [];
    this.speed = 450;
    this.paused = false;
    this.mySeat = -1; // client mode
    this.timer = null;
    this.lastAI = null; // info for inspector
    if (mode !== 'client') {
      this.state = newGame({ numPlayers: seats.length, names: seats.map((s) => s.name), seed });
      this.agents = seats.map((s) => this.makeAgent(s));
      this.schedule();
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
  apply(a, seat) {
    applyAction(this.state, a);
    this.emit({ action: a, seat });
    if (this.mode === 'host') this.broadcast();
    this.schedule();
  }
  broadcast() {
    this.seats.forEach((st, i) => { if (st.kind === 'remote' && st.link) st.link.send({ type: 'state', state: viewFor(this.state, i) }); });
  }
  schedule() {
    clearTimeout(this.timer);
    const s = this.state;
    if (!s || s.over || this.paused) return;
    const seat = this.seats[actor(s)];
    if (seat.kind !== 'ai') return;
    const humans = this.seats.some((x) => x.kind !== 'ai');
    this.timer = setTimeout(() => this.stepAI(), humans ? this.speed : Math.max(20, this.speed / 4));
  }
  stepAI() {
    const s = this.state;
    if (s.over) return;
    const pi = actor(s);
    const ag = this.agents[pi];
    if (!ag) return;
    const info = {};
    const a = ag instanceof NNAgent ? ag.act(s, info) : ag.act(s);
    this.lastAI = info.acts ? { seat: pi, ...info } : null;
    this.apply(a, pi);
  }
  // client: install state from host
  setRemoteState(st) {
    const prev = this.state?.log?.length ? this.state.log[this.state.log.length - 1] : null;
    this.state = st;
    const last = st.log.length ? st.log[st.log.length - 1] : null;
    this.emit(last && last !== prev && JSON.stringify(last) !== JSON.stringify(prev) ? { action: last.a, seat: last.p } : {});
  }
  receiveFromClient(seat, msg) {
    if (msg.type === 'action') this.dispatch(msg.action, seat);
    else if (msg.type === 'hello') this.broadcast();
  }
  destroy() { clearTimeout(this.timer); this.listeners = []; }
}
