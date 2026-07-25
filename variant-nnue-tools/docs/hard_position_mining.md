# Hard position mining pipeline

Hard position mining inserts a filtering step between self-play data generation
and NNUE training:

```text
generate_training_data -> transform mine -> optional transform rescore -> train.py
```

The implementation lives in `variant-nnue-tools` because that project already
owns packed-SFEN IO, `PackedSfenValue` unpacking, `Position` reconstruction, NNUE
loading/evaluation, and search. The miner is registered under the existing
`transform` command and also has a top-level `mine` alias for renamed tool
executables.

## Stage 1: mine hard positions

### `search-gap`

Compares the stored search score in each `PackedSfenValue` with a current NNUE:

```bash
./stockfish transform mine --mode search-gap \
  --input train.bin \
  --output hard.bin \
  --net current.nnue \
  --keep-count 1000000 \
  --min-gap 50
```

The largest absolute `stored_search_score - current_nnue_eval` gaps are kept.

### `eval-disagree`

Compares two NNUE files on the same input positions:

```bash
./stockfish transform mine --mode eval-disagree \
  --input train.bin \
  --output hard.bin \
  --old-net old.nnue \
  --new-net new.nnue \
  --keep-count 1000000
```

This mode uses two streaming passes. The first pass writes compact old-network
evaluations to a temporary sidecar next to the output path; the second pass loads
the new network, computes absolute disagreement, and removes the sidecar.

### Selection controls

* `--keep-count N` keeps at most `N` positions using a bounded heap.
* `--min-gap N` ignores positions with disagreement below `N`.
* `--batch-size N` controls output buffering.

Input is streamed from disk. Memory is bounded by `--keep-count`, not by the size
of the input file.

## Stage 2: deep rescore only mined positions

Rescore an already-mined subset with the existing multithreaded search-backed
transform:

```bash
./stockfish transform rescore \
  --input hard.bin \
  --output hard_rescored.bin \
  --depth 10 \
  --nodes 0 \
  --research-count 0
```

`--nodes 0` means depth-only search. Set `--nodes` to cap each position by nodes.
The command respects the engine `Threads` option and reuses the packed `.bin`
writer.

For the common mine-then-rescore case, `search-gap-deep` composes Stage 1 and
Stage 2 in one command, using a temporary mined `.bin` and writing only the final
rescored data:

```bash
./stockfish transform mine --mode search-gap-deep \
  --input train.bin \
  --output hard_rescored.bin \
  --net current.nnue \
  --keep-count 1000000 \
  --depth 10 \
  --nodes 0
```

## Stage 3: train

`variant-nnue-pytorch/train.py` already consumes packed `.bin` files, so the
cleanest integration is to keep preprocessing in `variant-nnue-tools` and pass
the mined or rescored file to the trainer:

```bash
python variant-nnue-pytorch/train.py hard_rescored.bin validation.bin --gpus 1 --threads 8
```

For an end-to-end wrapper without manual file handoff, use:

```bash
python variant-nnue-pytorch/scripts/hard_position_pipeline.py \
  --tools ./variant-nnue-tools/src/stockfish \
  --train-input train.bin \
  --val validation.bin \
  --mine-output hard.bin \
  --rescored-output hard_rescored.bin \
  --mode search-gap \
  --net current.nnue \
  --keep-count 1000000 \
  --rescore-depth 10 \
  -- --gpus 1 --threads 8 --batch-size 16384
```

Omit `--rescored-output` to train directly on mined data.

## Recommended workflows

* Fast iteration: generate -> `search-gap` -> train.
* Higher-quality refresh: generate -> `search-gap-deep` or `search-gap` +
  `rescore` -> train.
* Net comparison/regression: generate -> `eval-disagree` between old and new nets
  -> inspect statistics or train on the disagreement subset.
