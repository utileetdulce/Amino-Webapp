// Serverless multiplayer: WebRTC data channels. Signalling needs no server: the invite and the answer
// are compressed into short codes that travel as links (?join=… / ?answer=…) or copy & paste.
const RTC_CFG = { iceServers: [{ urls: 'stun:stun.l.google.com:19302' }, { urls: 'stun:global.stun.twilio.com:3478' }] };

const toB64u = (bytes) => {
  let bin = '';
  bytes.forEach((b) => (bin += String.fromCharCode(b)));
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
};
const fromB64u = (s) => {
  s = s.replace(/-/g, '+').replace(/_/g, '/');
  return Uint8Array.from(atob(s + '='.repeat((4 - (s.length % 4)) % 4)), (c) => c.charCodeAt(0));
};

async function pack(obj) {
  const bytes = new TextEncoder().encode(JSON.stringify(obj));
  if (typeof CompressionStream === 'undefined') return 'p' + toB64u(bytes);
  const cs = new CompressionStream('deflate-raw');
  const w = cs.writable.getWriter(); w.write(bytes); w.close();
  return 'z' + toB64u(new Uint8Array(await new Response(cs.readable).arrayBuffer()));
}

// Accepts a bare code or a full link containing ?join=… / ?answer=…
export function extractCode(str) {
  str = (str || '').trim();
  const m = str.match(/[?&](?:join|answer)=([A-Za-z0-9_-]+)/);
  return m ? m[1] : str.replace(/\s+/g, '');
}

export async function unpack(str) {
  const code = extractCode(str);
  if (!code) throw new Error('leerer Code');
  const bytes = fromB64u(code.slice(1));
  let out = bytes;
  if (code[0] === 'z') {
    const ds = new DecompressionStream('deflate-raw');
    const w = ds.writable.getWriter(); w.write(bytes); w.close();
    out = new Uint8Array(await new Response(ds.readable).arrayBuffer());
  } else if (code[0] !== 'p') throw new Error('ungültiger Code');
  return JSON.parse(new TextDecoder().decode(out));
}

function gatherComplete(pc, timeout = 4000) {
  return new Promise((res) => {
    if (pc.iceGatheringState === 'complete') return res();
    const t = setTimeout(res, timeout);
    pc.addEventListener('icegatheringstatechange', () => { if (pc.iceGatheringState === 'complete') { clearTimeout(t); res(); } });
  });
}

class Link {
  constructor() {
    this.pc = new RTCPeerConnection(RTC_CFG);
    this.dc = null; this.open = false; this.state = 'new';
    this.onMessage = () => {}; this.onOpen = () => {}; this.onClose = () => {}; this.onState = () => {};
    this.pc.onconnectionstatechange = () => { this.state = this.pc.connectionState; this.onState(this.state); };
  }
  wire(dc) {
    this.dc = dc;
    dc.onopen = () => { this.open = true; this.onOpen(); this.onState('connected'); };
    dc.onclose = () => { this.open = false; this.onClose(); this.onState('closed'); };
    dc.onmessage = (e) => this.onMessage(JSON.parse(e.data));
  }
  send(msg) { if (this.open) this.dc.send(JSON.stringify(msg)); }
  close() { try { this.pc.close(); } catch { /* ignore */ } }
}

export class HostLink extends Link {
  constructor(seat, tag) {
    super();
    this.seat = seat; this.tag = tag;
    this.wire(this.pc.createDataChannel('amino'));
  }
  async createInvite() {
    await this.pc.setLocalDescription(await this.pc.createOffer());
    await gatherComplete(this.pc);
    this.code = await pack({ k: 'offer', s: this.seat, g: this.tag, d: this.pc.localDescription.toJSON() });
    return this.code;
  }
  async acceptAnswer(str) {
    const m = await unpack(str);
    if (m.k !== 'answer') throw new Error('Das ist kein Antwort-Code (Einladungs-Code erwischt?)');
    if (m.g !== this.tag || m.s !== this.seat) throw new Error('Antwort gehört zu einer anderen Einladung');
    await this.pc.setRemoteDescription(m.d);
  }
}

export class GuestLink extends Link {
  constructor() {
    super();
    this.pc.ondatachannel = (e) => this.wire(e.channel);
  }
  async acceptInvite(str) {
    const m = await unpack(str);
    if (m.k !== 'offer') throw new Error('Das ist kein Einladungs-Code (Antwort-Code erwischt?)');
    await this.pc.setRemoteDescription(m.d);
    await this.pc.setLocalDescription(await this.pc.createAnswer());
    await gatherComplete(this.pc);
    this.code = await pack({ k: 'answer', s: m.s, g: m.g, d: this.pc.localDescription.toJSON() });
    return this.code;
  }
}

// Same-browser shortcut: an opened answer link hands its code to the host tab.
export function openBus() {
  try { return new BroadcastChannel('amino-signal'); } catch { return null; }
}
