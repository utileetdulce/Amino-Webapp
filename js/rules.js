export const RULES_HTML = `
<div class="card rules">
<h2>Amino – Kurzregeln</h2>
<p>Amino ist ein Legespiel von Asis Hallab und Philipp Niederlag (<a href="https://github.com/asishallab/amino" target="_blank" rel="noopener">Original &amp; Handbuch</a>).
Du baust mit dreieckigen Plättchen ein Wegelabyrinth, führst eine Ameise zu Ressourcen und bringst sie in deinen Hügel.
<b>Wer zuerst 7 Punkte gesammelt hat, gewinnt.</b> Jeder Spieler bewertet Blume 🌸, Pilz 🍄 und Raupe 🐛 anders (1–3 Punkte, steht an der Koloniekarte).</p>
<figure class="rfig"><img src="assets/colony-123.webp" alt="Koloniekarte"><figcaption>Koloniekarte: alle Aktionen eines Zuges und rechts die Kammern mit den Punktwerten (hier Blume 1, Pilz 2, Raupe 3). Im Spiel findest du deine eigene unter „Karten“.</figcaption></figure>
<h3>Ein Zug – Aktionen in beliebiger Reihenfolge</h3>
<ul>
<li><b>Einkommen (1×):</b> 4 Dreiecke <i>oder</i> 1 Ameisenkarte <i>oder</i> einen gelegten Hügel zurück auf die Hand.</li>
<li><b>Dreieck tauschen (1×):</b> beliebiges Dreieck gegen eines aus der Hand eines Gegners.</li>
<li><b>Dreiecke legen:</b> beliebig viele. Wege müssen fortgesetzt werden, offene Kanten dürfen nicht an geschlossene stoßen. Zusammen mit einem Dreieck kannst du einen <b>Hügel</b> (ab dem ersten erscheint deine Ameise) oder eine <b>Ressource</b> legen. Zwischen deinem Hügel und jeder Ressource müssen mindestens 2 Dreiecke Platz haben (gegnerische Hügel zählen nicht) – das gilt beim Legen einer Ressource wie beim Bau eines Hügels, auch gegenüber einer Ressource, die gerade getragen wird. Eine Ressource liegt auf <i>einem</i> Wegstück ihres Dreiecks; bei getrennten Wegen kannst du wählen, auf welchem.</li>
<li><b>Ameise bewegen:</b> beliebig weit, wohin Wege führen; mit zwei Hügeln auch unterirdisch dazwischen. Trägt höchstens eine Ressource; auf-/abladen jederzeit – zum Aufladen muss sie auf dem Wegstück der Ressource stehen. <b>Trägt die Ameise eine Ressource, bewegt sie sich ausschließlich per Transport (Würfelwurf) – ohne Ausnahme.</b></li>
<li><b>Kampf (1×):</b> Steht deine Ameise (ohne Last) auf dem Feld einer Ameise mit Ressource, verbirgt der Verteidiger die Ressource in einer Faust; der Angreifer tippt. Treffer = Ressource gehört dir.</li>
<li><b>Transport (1×):</b> Würfle (0–5); so viele Dreiecke darfst du die getragene Ressource Richtung Hügel bringen. Im Hügel angekommen zählt sie sofort. Nicht erlaubt für Ressourcen, die du in diesem Zug gelegt hast.</li>
</ul>
<h3>Ameisenkarten</h3>
<ul>
<li class="rc"><img src="assets/card-bridge.webp" alt=""><span><b>Brücke</b> – verbindet zwei Wege, nur du kannst sie überqueren.</span></li>
<li class="rc"><img src="assets/card-superimpose.webp" alt=""><span><b>Überlagern</b> – lege Dreiecke über freie Felder und verändere das Labyrinth.</span></li>
<li class="rc"><img src="assets/card-blockage.webp" alt=""><span><b>Blockade</b> – sperrt ein Dreieck bis zu deinem nächsten Zug.</span></li>
<li class="rc"><img src="assets/card-wings.webp" alt=""><span><b>Flügel</b> – fliege zu einem beliebigen Dreieck (zwei beliebige Karten lassen sich zu Flügeln tauschen).</span></li>
<li class="rc"><img src="assets/card-revenge.webp" alt=""><span><b>Rache</b> – verlorenen Kampf wiederholen.</span></li>
<li class="rc"><img src="assets/card-transport.webp" alt=""><span><b>Transport</b> – zwei Würfel beim Transport.</span></li>
</ul>
<h3>Bedienung</h3>
<p>Dreieck in der Hand anklicken, mit <kbd>R</kbd> / Rechtsklick drehen, auf ein hervorgehobenes Feld klicken. Brett: ziehen = verschieben, Mausrad = Zoom.</p>
<h3>Abweichungen von der Tabletop-Vorlage</h3>
<p>Die Dreiecke zeigen die gemalten Original-Dreiecke. Die Mischung der sechs Wege-Muster ist hier anders: mehr Gabeln und getrennte Stummel (10 / 8 / 6 statt je 2 von 50), dafür weniger Y-Kreuzungen und Kurven. Ressourcen werden verdeckt vom Stapel gezogen, Einkommen/Karten/Kampf folgen dem Handbuch. Regeln dieser Umsetzung sind bewusst vereinfacht: Hügel-Felder verbinden alle ihre Wege (eine Ressource dagegen nicht); Brücken verbinden zwei <i>benachbarte</i> Dreiecke.</p>
</div>`;
