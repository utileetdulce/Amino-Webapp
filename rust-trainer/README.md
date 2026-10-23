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

Note: the default `rustup` toolchain here is x86_64 and would run under Rosetta (~2-3x slower); the config builds for
`aarch64-apple-darwin` (`rustup target add aarch64-apple-darwin`).

## Parity with the JS engine
`node tools/xcheck.mjs models/pretrained.json 300` and `amino-train xcheck models/pretrained.json 300` print one
deterministic trace per seed (winner, turns, steps, action-count hash, feature/logit sums). The first five columns are
identical for all 300 seeds; sums differ only by float rounding.
