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

### Modelle, Metadaten und Regelversion

Jedes Modell trägt Metadaten (`meta`): Name, Typ (PPO / AlphaZero), Trainer (Browser / JS / Rust), Erstell- und Trainingsdatum, Trainingsdauer, Spiele, Hyperparameter, Rechner, Git-Stand, Herkunft (von welchem Modell weitertrainiert) und die **Regelversion** (`RULES_VERSION` in `js/engine.js`, Änderungsliste dort). Im Spielen-Tab zeigt die Modellauswahl Datum und Regelversion; ⚠ markiert Modelle, die mit älteren Regeln trainiert wurden, ★ das empfohlene. Im Training-Tab gibt es eine Info-Tabelle. `models/index.json` (erzeugt mit `node tools/update-models-index.mjs`) listet die mitgelieferten Modelle.

**Weitertrainieren oder neu starten?** Solange Netzgröße und Merkmale gleich bleiben, ist Weitertrainieren fast immer besser: Das Netz kennt Wege, Ressourcen und Taktik schon, nur Feinheiten (z. B. die Transport-Hausregel) lernt es um. Neu starten lohnt sich bei geänderten Merkmalen/Architektur oder wenn das alte Modell schlecht ist (zusammengebrochen). Die Trainer warnen, wenn das Ausgangsmodell mit älteren Regeln trainiert wurde.

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

**Erfahrungen (Stand Oktober 2026):** Besuchszahlen als Lernziel (Standard-AlphaZero) flachten bei ~100 Simulationen die Policy ab und zerstörten das Netz; deshalb sind Top-12-Suche und Q-basierte Ziele (`--topk`, `--qbeta`) Standard. Ein sehr langer Lauf (31.000 Spiele, Lernrate 1e-4) driftete trotzdem ab (gegen die Heuristik 85 % → 62 %, im Direktvergleich nicht besser als der Start), ein kurzer Lauf (100 Iterationen, Lernrate 5e-5) schlug das Ausgangsmodell dagegen klar (66 % im Direktvergleich). Der Rust-Trainer hat daher **Champion-Gating**: alle `--gate-every` Iterationen (Standard 20) spielt der aktuelle Stand gegen den Champion; ab `--gate-win` (0,55) wird er neuer Champion (`<out>.champion.json`), unter `--gate-revert` (0,40) wird auf den Champion zurückgesetzt. Nimm für das Spiel die `.champion.json`. Standard sind jetzt Lernrate 5e-5 und 200 Gating-Partien (≥ 56 % zum Befördern, < 42 % zum Zurücksetzen): Ein Lauf mit Lernrate 3e-4 über 45.000 Spiele (M4) war trotz Gating deutlich schwächer als das Ausgangsmodell (40 % bzw. 37 % im Direktvergleich), weil 60 Gating-Partien (±6 Punkte Rauschen) durch Zufall beförderten. Kurze Läufe (≈ 100–200 Iterationen) mit kleiner Lernrate sind bisher das einzige, was nachweislich besser wurde. Weitere Versuche ab „AZ kurz“ (Oktober 2026, je 20–25 Minuten, Gating mit 200 Partien): mit den Regeln v5 (Lernrate 5e-5) und mit einem auf 23.000 Parameter vergrößerten Netz (`tools/widen-model.mjs`, funktionsgleich gestartet, Lernrate 1e-4) wurde kein Zwischenstand je befördert – der aktuelle Stand verlor immer mit 36–48 % gegen den Champion. Auch mehr Simulationen und größere Iterationen allein (`--sims 200 --games 100`, 30 Minuten, 31.500 Spiele, Modell `az-s200`) brachten nichts: kein Zwischenstand befördert (bestes 51 %), zehnmal zurückgesetzt.

**Ursachenanalyse (Oktober 2026, gemessen mit `amino-train bench/trace/diag`, je 400–600 Partien):**

1. *Die Suche funktioniert.* „AZ kurz“ gegen die Heuristik: Netz allein (Temperatur 0,5) 70 %, MCTS 64 Sim. 72 %, 200 Sim. 80 %, 800 Sim. 85 %. Suchtiefe (6 vs. 18 vs. 40) ist egal, ohne Value-Netz (reine Prior-Suche) 74 %.
2. *Das Netz allein ist nur durch Zufall stark.* Gierig (immer die wahrscheinlichste Aktion) spielt es 99 % „Zug beenden“ und gewinnt keine einzige von 400 Partien gegen die Heuristik – „Ende“ ist eine einzelne Aktion, während sich die sinnvollen Aktionen auf viele Kandidaten verteilen.
3. *Hauptursache: Der Value-Kopf hat einen Sprung an jeder Zuggrenze.* Lernbeispiele entstehen pro Entscheidung, und Verlierer treffen etwa doppelt so viele Entscheidungen wie Gewinner (lange, suchende Züge; mittlerer Ausgang über alle Entscheidungen −0,33). Das Netz lernt daher „wer mitten im Zug ist, verliert“. Die Suche dreht beim Zugwechsel das Vorzeichen um – dadurch sah „Zug beenden“ systematisch **+0,38** besser aus, als es ist (ein konsistenter Value dürfte sich im Mittel nicht ändern; nach normalen Aktionen ist die Änderung tatsächlich ≈ 0). Die Q-Ziele haben diese Vorliebe ins Netz übertragen: P(„Ende“) stieg von 0,07 (PPO) auf 0,25 in den langen Läufen → passives Spiel, schwächere Modelle. Je länger trainiert wird, desto stärker die Rückkopplung.
4. *Fix: zugkonsistente Value-Ziele* (`--vtd 1`, jetzt Standard): Die erste Entscheidung eines Zuges lernt den Spielausgang, Entscheidungen mitten im Zug lernen den Wert zu Beginn des nächsten Zuges. Sprung nach „Ende“ +0,38 → +0,03, Korrelation mit dem Ausgang 0,62 → 0,67. Nur den Value-Kopf von „AZ kurz“ so nachzutrainieren (5 Minuten, Policy unverändert, Modell `az-vtd`) macht die Suche spürbar stärker: **60 % im Direktvergleich** mit Suche gegen „AZ kurz“ (600 Partien, ±4), 83 % statt 80 % gegen die Heuristik. Gewichtung pro Spieler und Partie (`--vbalance`) hilft dagegen nicht (Sprung kippt auf −0,15, Value wird schlechter).
5. *Zweite Ursache: Das Policy-Lernsignal ist praktisch null.* Mit `--qbeta 3` unterscheiden sich die Q-Ziele kaum vom Prior (KL ≈ 0,002), obwohl die Suche das eigene Netz allein mit 70 % schlägt. Das Training ist also fast nur Rauschen – darum driften lange Läufe, statt besser zu werden. Ein 30-Minuten-Lauf mit dem Fix (`az-v2`, 36.000 Spiele, Gating mit Suche) bleibt stabil (Gating 43–55 %, statt 36–45 % vorher), wird aber nicht besser als `az-vtd` (51 %).
6. *Stärkere Ziele kippen in ein passives Gleichgewicht.* Je 10 Minuten ab `az-vtd` (600 Partien Direktvergleich mit Suche gegen `az-vtd`): `--qbeta 10` 38 %, `--qbeta 30` 34 %, geschärfte Besuchszahlen (`--qbeta 0 --ttemp 0.5`) 33 %. In allen drei Läufen wuchsen die Self-Play-Partien von ~65 auf 145–170 Züge (meist Remis an der 90-Runden-Grenze): Die Policy spielt fast nur noch „bewegen“ (q10: 79 %, kaum noch Plättchen legen) bzw. „Ende“ (Besuchszahlen: 93 %). Pro Update schieben die Ziele genau die Fortschritts-Aktionen leicht nach unten (Transport −4 %, Dreiecke ziehen −7 %, Karte −2 %, „Ende“ +3 %); mit stärkerem `qbeta` verstärkt sich das, und weil beide Self-Play-Seiten passiv werden, bestätigt sich das Gleichgewicht selbst. Gegenmaßnahmen getestet: Remis als −0,5 werten (`--draw-z -0.5`) stoppt den Kollaps bei `qbeta 10` nicht (40 %), bei `qbeta 3` gibt es aber die erste messbare Verbesserung des Netzes allein (**57 %** gegen `az-vtd`, mit Suche 48 %). Den besten Kandidaten jedes Aktionstyps immer mitsuchen (`--type-div`) und Simulationen nach Zufallsaktionen direkt mit dem Value bewerten (`--chance-leaf`, gegen die Open-Loop-Vermischung der Würfelergebnisse) ändern die Suchstärke nicht (je 48 %); beide sind als Option vorhanden, standardmäßig aus.

Fazit: Der wichtigste Fehler (Value-Sprung an der Zuggrenze) ist behoben, `az-vtd` ist mit Suche messbar stärker als „AZ kurz“. Mit diesem kleinen Netz und diesen Merkmalen liefert Self-Play-Training aber kaum ein Lernsignal, das besser ist als Rauschen: schwache Ziele ändern fast nichts, starke Ziele führen in passives Spiel. Erfolgversprechender ist vermutlich, die Suche selbst im Spiel zu nutzen (mehr Simulationen), mehr Partien gegen feste Gegner (Heuristik, frühere Modelle) einzumischen oder das Netz bzw. die Merkmale zu erweitern.

Gemessen wird besser im Direktvergleich (`amino-train bench 600 mcts:A.json:128 mcts:B.json:128` oder `node tools/h2h.mjs A.json B.json 200`) als nur gegen die Heuristik; 16 Testpartien wie im Trainingslog haben ±12 Punkte Rauschen.

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
