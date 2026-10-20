// Serverless multiplayer: WebRTC data channels, manual copy/paste signalling (no server involved).
const RTC_CFG = { iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] };

async function pack(obj) {
  const json = JSON.stringify(obj);
  const bytes = new TextEncoder().encode(json);
  let out = bytes, prefix = 'p';
  if (typeof CompressionStream !== 'undefined') {
    const cs = new CompressionStream('deflate-raw');
    const w = cs.writable.getWriter(); w.write(bytes); w.close();
    out = new Uint8Array(await new Response(cs.readable).arrayBuffer()); prefix = 'z';
  }
  let bin = ''; out.forEach((b) => (bin += String.fromCharCode(b)));
  return prefix + btoa(bin);
}
async function unpack(str) {
  str = str.trim();
  const prefix = str[0];
  const bin = atob(str.slice(1));
  const bytes = Uint8Array.from(bin, (c) => c.charCodeAt(0));
  let out = bytes;
  if (prefix === 'z') {
    const ds = new DecompressionStream('deflate-raw');
    const w = ds.writable.getWriter(); w.write(bytes); w.close();
    out = new Uint8Array(await new Response(ds.readable).arrayBuffer());
  }
  return JSON.parse(new TextDecoder().decode(out));
}

function gatherComplete(pc, timeout = 5000) {
  return new Promise((res) => {
    if (pc.iceGatheringState === 'complete') return res();
    const t = setTimeout(res, timeout);
    pc.addEventListener('icegatheringstatechange', () => { if (pc.iceGatheringState === 'complete') { clearTimeout(t); res(); } });
  });
}

// One link between host and a guest.
export class HostLink {
  constructor() {
    this.pc = new RTCPeerConnection(RTC_CFG);
    this.dc = this.pc.createDataChannel('amino');
    this.onMessage = () => {}; this.onOpen = () => {}; this.onClose = () => {};
    this.open = false;
    this.dc.onopen = () => { this.open = true; this.onOpen(); };
    this.dc.onclose = () => { this.open = false; this.onClose(); };
    this.dc.onmessage = (e) => this.onMessage(JSON.parse(e.data));
  }
  async createInvite() {
    await this.pc.setLocalDescription(await this.pc.createOffer());
    await gatherComplete(this.pc);
    return pack(this.pc.localDescription.toJSON());
  }
  async acceptAnswer(code) {
    await this.pc.setRemoteDescription(await unpack(code));
  }
  send(msg) { if (this.open) this.dc.send(JSON.stringify(msg)); }
  close() { try { this.pc.close(); } catch { /* ignore */ } }
}

export class GuestLink {
  constructor() {
    this.pc = new RTCPeerConnection(RTC_CFG);
    this.dc = null; this.open = false;
    this.onMessage = () => {}; this.onOpen = () => {}; this.onClose = () => {};
    this.pc.ondatachannel = (e) => {
      this.dc = e.channel;
      this.dc.onopen = () => { this.open = true; this.onOpen(); };
      this.dc.onclose = () => { this.open = false; this.onClose(); };
      this.dc.onmessage = (ev) => this.onMessage(JSON.parse(ev.data));
    };
  }
  async acceptInvite(code) {
    await this.pc.setRemoteDescription(await unpack(code));
    await this.pc.setLocalDescription(await this.pc.createAnswer());
    await gatherComplete(this.pc);
    return pack(this.pc.localDescription.toJSON());
  }
  send(msg) { if (this.open) this.dc.send(JSON.stringify(msg)); }
  close() { try { this.pc.close(); } catch { /* ignore */ } }
}
