# amino-train — native AlphaZero trainer

Rust port of `tools/train-az.mjs` (engine, features, net + Adam, MCTS, self-play, eval). Same CLI flags, same model JSON
(`{version, name, kind, sims, games, iter, history, net:{arch,theta}}`), so `models/*.json` is interchangeable with the
web app and the JS tools. Self-play games and the minibatch gradients run on all cores via rayon.

```sh
cd rust-trainer
cargo build --release          # .cargo/config.toml pins the native aarch64 target (apple-m4)
./target/aarch64-apple-darwin/release/amino-train \
  --init ../models/pretrained.json --out ../models/az.json --sims 64 --games 24 --minutes 240
```

Flags: `--workers N` (default all cores) `--sims --games --steps --batch --lr --buffer --eval-every --eval-games
--opp-mix --depth --tau-moves --minutes --iters --resume --seed`.

Value / gating flags: `--vtd 1` (default; turn-consistent value targets — first decision of a turn learns the game
outcome, mid-turn decisions learn the value at the start of the next turn; fixes the value jump at turn boundaries that
made MCTS favour ending turns), `--vbalance 1` (weight value loss per player-game; tested, not helpful), `--gate-sims 64`
(gating plays with search; 0 = raw policy), `--qbeta`, `--topk`, `--cpuct`, `--pcoef/--vcoef`, `--gate-every/-games/-win/-revert`.

Graph net / experiments: `--grow-graph H,L` (grow a plain MLP into a graph net with identical outputs; GNN width H, L layers), `--auxcoef`
(aux score-margin head), `--sp-champion 1` (self-play data from the frozen champion), `--roll N --roll-w W` (heuristic-rollout leaf
evaluation; bench option `roll=N:rw=W`), `--draw-z`, `--type-div`, `--chance-leaf`. `amino-train bench` also prints the mean score margin.
Experiments: `gradcheck` (backward vs finite differences), `abtest <teacher> <games> <sims> <rounds> [lr] [hg,gl]` (does the board graph carry
information the MLP lacks? env `VRAW`, `SCRATCH`, `PCOEF`), `vtest` (value learning from rollout-averaged labels).

## Diagnostics
```sh
B=./target/aarch64-apple-darwin/release/amino-train
$B bench 600 mcts:../models/A.json:128 mcts:../models/B.json:128   # A vs B, alternating seats, score ± 95% interval
#   agents: heur | raw:<model>:<temp, 0 = greedy> | mcts:<model>:<sims>[:topk=12][:cpuct=1.25][:depth=18][:v0][:nodet]
$B trace mcts:../models/A.json:200 40                              # action-type histogram vs heuristic
$B diag ../models/A.json 64 140 [qbeta]                            # value calibration, value jump after 'end', target-vs-prior KL
```
See the main README ("Ursachenanalyse") for what these showed.

Note: the default `rustup` toolchain here is x86_64 and would run under Rosetta (~2-3x slower); the config builds for
`aarch64-apple-darwin` (`rustup target add aarch64-apple-darwin`).

## Parity with the JS engine
`node tools/xcheck.mjs models/pretrained.json 300` and `amino-train xcheck models/pretrained.json 300` print one
deterministic trace per seed (winner, turns, steps, action-count hash, feature/logit sums). The first five columns are
identical for all 300 seeds; sums differ only by float rounding.
