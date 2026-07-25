# Hard position mining

Hard position mining filters an existing packed `.bin` training-data file down to
positions where an NNUE network most needs additional training signal. It is
implemented as a `transform mine` subcommand so it reuses the existing packed
SFEN reader/writer, packed SFEN unpacking, `Position` reconstruction, and NNUE
loader/evaluator.

The tool keeps the `--keep-count` positions with the largest disagreement and
writes them to a new packed `.bin` file. Input is streamed from disk; only the
current top-k positions are retained in memory.

## Search gap mode

Compare each entry's stored search score against a current NNUE evaluation:

```text
transform mine --mode search-gap --input training.bin --output hard.bin --net current.nnue --keep-count 1000000
```

## Eval disagreement mode

Compare two NNUE networks on the same positions:

```text
transform mine --mode eval-disagree --input training.bin --output hard.bin --old-net old.nnue --new-net new.nnue --keep-count 1000000
```

This mode uses two streaming passes. The old network pass writes a compact
temporary sidecar file next to the output path, then the new network pass computes
the final absolute gaps and removes the sidecar.

## Notes

* `--input` and `--output` are packed `.bin` paths.
* `--batch-size` controls output buffering and defaults to `100000`.
* `search-gap-deep` is intentionally not implemented here yet because it needs a
  carefully integrated deep-search scheduling policy; the existing `transform
  rescore` command remains the reusable building block for deep rescoring a mined
  subset.
