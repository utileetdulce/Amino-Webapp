import { PlayUI, el } from './ui-play.js';
import { TrainUI } from './ui-train.js';
import { RULES_HTML } from './rules.js';

const tabs = { play: document.getElementById('tab-play'), train: document.getElementById('tab-train'), rules: document.getElementById('tab-rules') };
const ui = { play: new PlayUI(tabs.play), train: null };
tabs.rules.innerHTML = RULES_HTML;

function show(name) {
  for (const k in tabs) tabs[k].hidden = k !== name;
  document.querySelectorAll('nav button').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
  if (name === 'train' && !ui.train) ui.train = new TrainUI(tabs.train);
  if (name === 'play') ui.play.ctrl?.state && ui.play.board?.resize();
  location.hash = name;
}
document.querySelectorAll('nav button').forEach((b) => b.addEventListener('click', () => show(b.dataset.tab)));
const h = location.hash.slice(1);
show(tabs[h] ? h : 'play');
window.__amino = { ui };
