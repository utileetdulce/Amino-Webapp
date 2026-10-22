# Amino – Webapp mit KI-Training im Browser

Spielbare Umsetzung des Legespiels **Amino** ([Original & Handbuch](https://github.com/asishallab/amino)) – komplett clientseitig, ohne Build-Schritt und ohne Server.

* **Spielen**: 2–6 Spieler. Pro Platz wählbar: Mensch (Hotseat), KI · Heuristik, KI · Zufall, KI · Neuronales Netz oder **Online-Mitspieler**.
* **Serverloser Multiplayer**: WebRTC-Datenkanäle (P2P). Der Gastgeber erzeugt einen Einladungs-Link (`?join=…`), der Gast öffnet ihn und schickt einen Antwort-Link (`?answer=…`) zurück; im selben Browser wird er automatisch übergeben, sonst einfach einfügen. Kein Server nötig. Der Gastgeber führt die Spiellogik aus; Gäste sehen nur ihre eigenen Ameisenkarten.
* **Inszenierte KI-Züge**: Jede Aktion wird als Popup beschrieben, die Kamera fährt zum Geschehen, Würfel rollen sichtbar auf dem Brett, Kämpfe und Punkte erscheinen als Pop-ups. Klick/Leertaste geht schneller weiter, „⏸ Schritt“ steuert Aktion für Aktion, „⏩ Schnell“ überspringt alles.
* **KI-Training im Browser** (Tab „KI-Training“): PPO-Self-Play im Web-Worker mit Live-Charts (Siegquote vs. Zufall/Heuristik, Return, Entropie/KL, Verluste), Gewichts-Heatmaps, Neuron-Aktivierungen und einem Entscheidungs-Inspektor, der zeigt, wie das Netz Aktionen bewertet. Modelle speichern, exportieren, importieren und im Spiel als Gegner wählen.
* Ein vortrainiertes Modell liegt in `models/pretrained.json`.

## Starten

```bash
python3 -m http.server 8000     # oder: npx serve .
# http://localhost:8000
```

(ES-Module und Web-Worker brauchen http://, `file://` funktioniert nicht.) 
**Deployment** (reine statische Seite, kein Build): 
* *Netlify*: „Add new site → Import from Git“, Repo wählen – `netlify.toml` setzt Publish-Verzeichnis `.` und keinen Build-Befehl.
* *Render*: „New → Static Site“ (oder Blueprint mit `render.yaml`), Build Command leer/`echo`, Publish Directory `.`.
Beide liefern HTTPS, das WebRTC-Multiplayer braucht.

## KI-Architektur

| Baustein | Umsetzung |
|---|---|
| Aktionsraum | Ein Zug = Folge atomarer Aktionen (`js/engine.js`); pro Entscheidung wird die Menge legaler Kandidaten erzeugt (variabel groß) |
| Policy | `logit = MLP([Zustandsmerkmale ‖ Aktionsmerkmale])` (41+40 → 64 → 32 → 1), Softmax über Kandidaten (Action-Scoring, wie bei Karten-/Strategiespielen üblich) |
| Value | MLP über Zustandsmerkmale (41 → 48 → 1) |
| Algorithmus | PPO (GAE λ=0.95, Clip 0.2, Entropie-Bonus, Adam, Grad-Clipping), handgeschriebene Backprop ohne Bibliotheken (`js/nn.js`, `js/selfplay.js`) |
| Gegner im Training | Mix aus Self-Play, Heuristik, Zufall und eingefrorenen früheren Versionen (Liga-Prinzip) |
| Warmstart | optionales Behavior Cloning der Heuristik, danach PPO |
| Reward | +1 Sieg / −1 Niederlage, optionales Shaping pro erzielten Punkt |

### Baumsuche (MCTS, AlphaZero-Stil)

`js/mcts.js` sucht vor jeder Entscheidung mit PUCT: das Netz liefert Aktions-Prioritäten (Policy) und eine Stellungsbewertung (Value), die Suche verbessert beides durch Vorausschau.
Versteckte Informationen (Karten des Gegners, Stapel, die verdeckte Faust) werden pro Simulation per *Determinisierung* neu verteilt, Würfel/Ziehungen werden jedes Mal neu gewürfelt (Open-Loop-Baum).
Im Spiel wählbar als „KI · Netz + Baumsuche (MCTS)“ (Zeitbudget ca. 250 ms pro Entscheidung).

Training (Self-Play mit Such-Ergebnissen als Ziel: Policy = Besuchsverteilung, Value = Spielausgang), parallel auf allen CPU-Kernen:

```bash
# 1) optional: Netz per PPO vorwärmen (schnell)        node tools/train-cli.mjs --minutes 20 --out models/pretrained.json
# 2) AlphaZero-Schleife, z. B. auf einem MacBook M4:
node tools/train-az.mjs --init models/pretrained.json --out models/az.json --sims 96 --games 40 --workers 9 --minutes 240
node tools/mcts-bench.mjs models/az.json 96 20      # Suche vs. Heuristik messen
```

Wichtige Optionen: `--sims` (Simulationen pro Zug), `--games` (Selbstspiele je Iteration), `--steps/--batch/--lr` (Updates), `--buffer` (Replay-Größe), `--eval-every/--eval-games`, `--opp-mix` (Anteil Spiele gegen die Heuristik zur Stabilisierung), `--resume` (Zähler fortführen). Es wird laufend `models/az.json` und der beste Stand als `models/az.best.json` gespeichert – beide lassen sich im Training-Tab importieren und im Spiel auswählen.

Headless-Training (z. B. für ein stärkeres Modell):

```bash
node tools/train-cli.mjs --minutes 60 --out models/pretrained.json
node tools/bench.mjs 50       # Heuristik vs. Zufall
node tools/test.mjs 1 100     # Zufallspartien als Engine-Test
```

## Code-Überblick

```
js/geometry.js   Dreiecksgitter       js/engine.js   Regeln & Zustand (JSON-fähig)
js/features.js   Merkmale             js/nn.js       Netz + Adam
js/agents.js     Zufall/Heuristik/NN  js/selfplay.js PPO, Imitation, Evaluation
js/trainer.js    Trainingsschleife    js/train-worker.js  Web-Worker
js/render.js     Canvas-Brett         js/ui-*.js     Oberfläche
js/game.js       Spielcontroller      js/net.js      WebRTC-Signalisierung
```

## Abweichungen von der Tabletop-Vorlage

* Die genauen Wegemuster der sechs Dreieckstypen sind nachempfunden (Y, Kurve, Sackgasse, Gabel, 2 bzw. 3 getrennte Stummel; 50 Dreiecke).
* Hügel-/Ressourcenfelder verbinden alle ihre Wege; Brücken verbinden zwei *benachbarte* Dreiecke; Ressourcen werden verdeckt gezogen.
* **Hausregel (ohne Ausnahme):** Eine Ameise mit Ressource bewegt sich nur per Transport-Aktion mit Würfelwurf, nie durch normales Bewegen.
* Spielende nach 90 Runden mit Punktevergleich (Schutz gegen Endlosspiele).

Amino © Asis Hallab & Philipp Niederlag – diese Webapp ist eine unabhängige Umsetzung für private Zwecke.
