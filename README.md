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
* *GitHub Pages* (aktiv, kostenlos): https://utileetdulce.github.io/Amino-Webapp/ – Settings → Pages → „Deploy from a branch“ (`main`, `/`), wird bei jedem Push auf `main` neu veröffentlicht; `.nojekyll` schaltet die Jekyll-Verarbeitung ab.
* *Netlify*: „Add new site → Import from Git“, Repo wählen – `netlify.toml` setzt Publish-Verzeichnis `.` und keinen Build-Befehl.
* *Render*: „New → Static Site“ (oder Blueprint mit `render.yaml`), Build Command leer/`echo`, Publish Directory `.`.
Alle liefern HTTPS, das WebRTC-Multiplayer braucht.

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
6. *Stärkere Ziele kippen in ein passives Gleichgewicht.* Je 10 Minuten ab `az-vtd` (600 Partien Direktvergleich mit Suche gegen `az-vtd`): `--qbeta 10` 38 %, `--qbeta 30` 34 %, geschärfte Besuchszahlen (`--qbeta 0 --ttemp 0.5`) 33 %. In allen drei Läufen wuchsen die Self-Play-Partien von ~65 auf 145–170 Züge (meist Remis an der 90-Runden-Grenze): Die Policy spielt fast nur noch „bewegen“ (q10: 79 %, kaum noch Plättchen legen) bzw. „Ende“ (Besuchszahlen: 93 %). Pro Update schieben die Ziele genau die Fortschritts-Aktionen leicht nach unten (Transport −4 %, Dreiecke ziehen −7 %, Karte −2 %, „Ende“ +3 %); mit stärkerem `qbeta` verstärkt sich das, und weil beide Self-Play-Seiten passiv werden, bestätigt sich das Gleichgewicht selbst. Gegenmaßnahmen getestet: Remis als −0,5 werten (`--draw-z -0.5`) stoppt den Kollaps bei `qbeta 10` nicht (40 %), bei `qbeta 3` gab es nach 10 Minuten eine Verbesserung des Netzes allein (57 % gegen `az-vtd`, mit Suche 48 %), die ein 60-Minuten-Lauf (`az-v3`, 75 Gates, nie befördert) aber nicht bestätigte (50 % bzw. 46 %). Den besten Kandidaten jedes Aktionstyps immer mitsuchen (`--type-div`) und Simulationen nach Zufallsaktionen direkt mit dem Value bewerten (`--chance-leaf`, gegen die Open-Loop-Vermischung der Würfelergebnisse) ändern die Suchstärke nicht (je 48 %); beide sind als Option vorhanden, standardmäßig aus.

Fazit: Der wichtigste Fehler (Value-Sprung an der Zuggrenze) ist behoben, `az-vtd` ist mit Suche messbar stärker als „AZ kurz“. Mit diesem kleinen Netz und diesen Merkmalen liefert Self-Play-Training aber kaum ein Lernsignal, das besser ist als Rauschen: schwache Ziele ändern fast nichts, starke Ziele führen in passives Spiel. Erfolgversprechender ist vermutlich, die Suche selbst im Spiel zu nutzen (mehr Simulationen), mehr Partien gegen feste Gegner (Heuristik, frühere Modelle) einzumischen oder das Netz bzw. die Merkmale zu erweitern.

**Graph-Netz (Oktober 2026) – das erste Modell, das `az-vtd` messbar schlägt.** Das bisherige Netz sah nur 41 handgebaute Zahlen. Das neue (`hg > 0` in `rust-trainer/src/nn.rs`, `build_graph` in `features.rs`) bekommt zusätzlich das Brett: ein kleines Message-Passing-Netz (16 Dimensionen, 3 Schichten, 19.000 Parameter insgesamt) läuft über die Wegekomponenten (Knoten = (Dreieck, Pfadgruppe), Kanten = offene Dreiecksränder, 20 Merkmale je Knoten: Kanten, Hügel, freie Ressourcen, eigene/gegnerische Ameise, Träger, Entfernung von Ameise und Hügel, Sperren, Brücken …). Mittelwert aller Knoten, die eigene und die gegnerische Ameise und – je Kandidat – die Knoten des Zielfelds fließen in die bekannten Köpfe. Dazu kommt ein Hilfs-Kopf für die Punktedifferenz am Spielende (`--auxcoef`). Gestartet wird funktionsgleich aus einem bestehenden MLP (`--grow-graph 16,3`: alle Gewichte, die das Brett lesen, beginnen bei null), dann normales Selbstspiel. `js/gnet.js` rechnet das Netz im Browser (nur Inferenz, geprüft mit `tools/xcheck.mjs` gegen den Rust-Trainer: gleiche Partien, Summen auf 3 Nachkommastellen gleich). Training im Browser bleibt dem MLP vorbehalten.

Ergebnis (400 Partien, beide mit Suche, 128 Simulationen): nach 40 Minuten 56 % gegen `az-vtd`, nach ~90 Minuten (9.300 Spiele, Modell `graph-g4`) **63 % (±5)** bei +1,19 (±0,35) Punkten Differenz. Eine Fortsetzung um weitere 45 Minuten (`graph-g5`, 19.500 Spiele insgesamt) liegt bei 67 % gegen `az-vtd` (+1,57 Punkte), gegen `graph-g4` aber bei 98 % Remis – starke Netze patten sich am Rundenlimit, der Fortschritt ist nur noch gegen schwächere Gegner messbar. **Feature-Version 2 (`graph-h1`, Standard).** Weiteres Wissen, das das Netz vorher nicht hatte (`--grow-features` erweitert ein Graph-Netz funktionsgleich, die neuen Eingänge starten bei null): Knotenmerkmale für Erreichbarkeit und Entfernung der gegnerischen Ameise und seiner Hügel (Bedrohungskarte), zusammenhängende Wegbereiche mit Inhalt (eigene/gegnerische Ameise, Hügel, freie Ressource), offene Ränder, an die noch angebaut werden kann, und frisch gelegte Ressourcen; globale Zusatzwerte (Hand und Karten des Gegners, sein Träger und dessen Abstand zum eigenen Hügel, ob der Gegner meine Ameise erreichen kann, Wettlauf um freie Ressourcen); je Kandidat bei Platzierungen die offenen Ränder und Wegverbindungen des Plättchens, welche Bereiche es verbindet (z. B. meine Ameise mit einer Ressource oder mit der gegnerischen Ameise) und beim Transport die Würfelchance, den Hügel zu erreichen. Nach 45 Minuten Training: 71 % (±5) gegen `az-vtd` (Vorgänger 67 %), gegen die Heuristik +5,35 statt +4,75 Punkte Differenz, Partien enden früher (27 statt 32 Züge), der Policy-Verlust liegt bei ~1,05 statt ~1,18. Der Gewinn ist moderat und ohne Kontrolllauf (derselbe Zeitraum nur mit dem alten Netz) nicht sicher zuzuordnen.

Was dabei half bzw. nicht half:

- *Gradiententest* (`amino-train gradcheck`): handgeschriebene Rückwärtsrechnung stimmt mit numerischen Gradienten (0,2 % des größten Gradienten).
- *Nachahmen des Lehrers bringt nichts* (`amino-train abtest`): Auf festen Lehrer-Daten (257.000 Beispiele) sind MLP und Graph-Netz gleich gut (Policy-Fehler 1,658 vs. 1,660) – die Zielwerte stammen aus einem MLP und lassen sich vom MLP schon nachbilden. Der Nutzen entsteht erst im Selbstspiel.
- *Überanpassung bei wenig Partien* (`abtest` mit `VRAW=1 SCRATCH=1`): Von Grund auf trainiert prägt sich das Graph-Netz einzelne Partien ein (Trainingsfehler 0,24, Testfehler 0,58; MLP 0,40/0,445), weil jedes Brett ein Fingerabdruck der Partie ist. Deshalb Selbstspiel mit frischen Daten und kleinem `--ratio 1.5`, großem Puffer (150.000) – und warm starten statt von null.
- *Gating mit Direktvergleich taugte vorübergehend nicht* (behoben, siehe unten): Starke Netze spielten gegeneinander zu 60–80 % Remis (Stillstand bis zum Rundenlimit), die Quote bleibt bei 50 %. Aussagekräftig sind `bench` mit der mittleren Punktedifferenz („margin“) und die mittlere Endpunktzahl gegen die Heuristik (`amino-train trace`: `az-kurz` 5,7 · `az-vtd` 6,2 · Graph-Netz 6,4–6,5 von 7). Läufe daher mit `--gate-every 0` und anschließender Messung.
- Weitere Versuche ohne Gewinn: Heuristik-Rollouts im Blatt der Suche (`--roll`, `bench`-Option `roll=`; entspricht etwa 4× mehr Simulationen), Wert-Training mit breiterem Kopf, Daten vom eingefrorenen Champion (`--sp-champion`), mehr Simulationen, `topk`/`cpuct`-Änderungen.

**Der Patt war kein Rundenproblem, sondern Stillstand (Oktober 2026).** Partien zwischen Graph-Netzen, die das 90-Runden-Limit erreichen, sind keine langen, knappen Spiele: 0:0, keine einzige Ressource gelegt, 47 von 50 Dreiecken noch im Stapel – beide Seiten beenden einfach jeden Zug (`amino-train stall`). Mit 300 Runden ändert sich nichts (100 % Stillstand); im Selbstspiel von `graph-h1` endeten 98 % der Partien so, das Netz lernte also fast nur aus den Heuristik-Partien. Ursache (`amino-train probe`): Das Netz hat gelernt, rein reaktiv zu spielen – wer in seinen eigenen Partien zuerst baut, verliert leicht (−0,12), weil der andere die Ressourcen klaut. Einkommen hält es durchaus für gut (bester Wert nach der Aktion), probiert es aber nie (Prior 0,000), weil Selbstspiel nach den ersten 30 Entscheidungen fast nur noch die beste Aktion spielt. Erzwingen hilft (Einkommen vor „Ende“: 90 % entschiedene Partien), ist aber unnötig – es ist ein Trainingsfehler. **Fix im Training:** Remis zählt als Niederlage für beide (`--draw-z -1`) und Selbstspiel probiert über die ganze Partie (`--tau-moves 400`). Ergebnis nach 60 Minuten ab `graph-h1` (`graph-e1`, 24.900 Spiele): Selbstspiel mit fester Suche zu 78 % entschieden (vorher 2 %), **63 % (±5) im Direktvergleich gegen `graph-h1` bei nur 7 % Remis** (Punktedifferenz +1,25), 75 % gegen `az-vtd`, gegen die Heuristik 97 % und +5,95 Punkte. Der Wert von „Hügel legen“ im ersten Zug stieg von −0,13 auf 0,00, „Zug beenden“ fiel auf −0,13. Damit taugt der Direktvergleich wieder als Gating.

**Regelversion 8 (28.10.2026): keine Ameisen-Spaziergänge der KI.** Die KI machte in ihren Zügen auffallend viele sinnlose Ameisenbewegungen (Messung mit `amino-train trace`: 60 Entscheidungen je Zug, 73 % davon „bewegen“, nur 1,7 % „Zug beenden“). Jetzt gilt für die KI-Kandidaten: Nach einer Ameisenbewegung ohne Ressource ist keine zweite erlaubt, solange dazwischen keine andere Aktion (Angriff, Dreiecke legen, Überbauen, Blockade, Abladen, Aufladen, Karte …) stattfand; eine Bewegung von Hügel zu Hügel (Tunnel) zählt nicht mit. Menschen sind nicht eingeschränkt. Wirkung am bisherigen Modell, ohne Neutraining: 20 statt 60 Entscheidungen je Zug, „bewegen“ nur noch 32 %; die Stärke gegen die Heuristik sinkt vorerst von 96 % auf 92 %, weil das Modell auf lange Bewegungsketten trainiert war. Tests: `tools/test-plain-moves.mjs`, Rust/JS-Gleichheit mit `xcheck` (180 Partien).

**Regelversion 6 (27.10.2026).** Der Hügelabstand gilt jetzt auch gegenüber getragenen Ressourcen, und eine Ressource liegt auf genau einem Wegstück ihres Dreiecks. Außerdem war im Browser ein Knotenmerkmal des Graph-Netzes („Ressource in diesem Zug gelegt“) immer 0; seit der Korrektur rechnen Browser und Rust-Trainer gleich (`tools/xcheck.mjs`, 150 Partien). 60 Minuten Weitertraining ab `graph-e1` unter den neuen Regeln (`graph-r6`, Rezept wie `graph-e1`): Endstand 44 % (±5) gegen `graph-e1`, bester Zwischenstand (`graph-r6.best`, Iteration 320) 49 % (±5), also gleich stark. `graph-r6.best` ist das Standardmodell, weil es die aktuellen Regeln kennt.

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
* Hügelfelder verbinden alle ihre Wege; eine Ressource liegt auf genau einem Wegstück ihres Dreiecks (Regelversion 6); Brücken verbinden zwei *benachbarte* Dreiecke; Ressourcen werden verdeckt gezogen.
* **Hausregel (ohne Ausnahme):** Eine Ameise mit Ressource bewegt sich nur per Transport-Aktion mit Würfelwurf, nie durch normales Bewegen.
* Spielende nach 90 Runden mit Punktevergleich (Schutz gegen Endlosspiele).

Amino © Asis Hallab & Philipp Niederlag – diese Webapp ist eine unabhängige Umsetzung für private Zwecke.
