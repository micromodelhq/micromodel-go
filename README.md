# micromodel-go

Run micromodels on the CPU in pure Go. It loads any **micromodel encoder**: a `.mmenc` file holding a ModernBERT
model compressed to under 1 bit per weight (LittleBit-2), 4-bit embeddings, its tokenizer and an optional
classification head. No cgo, no GPU; the only dependency is `golang.org/x/text` for NFC.

```go
import micromodel "github.com/micromodelhq/micromodel-go"

enc, err := micromodel.Load(bytes)            // e.g. from os.ReadFile or //go:embed
logits, err := enc.Logits("my card got swallowed by the atm!!")   // in enc.Labels() order
```

- `Load(b)` parses a `.mmenc`; sign matrices and embeddings are used in place, so keep `b` unmodified.
- `Logits(text)` → `[]float32` in `Labels()` order (classify models). `Embed(text)` → the L2-normalized
  mean-pooled embedding.
- `IDs(text)` gives the model's input ids (prefix, `[bos] … [eos]`, truncated to `max_tokens`); `Tokenizer.Encode`
  is the bare tokenizer.
- `Info` is the file header. An `Encoder` is safe for concurrent use.

It is a port of `@micromodel/cpu` (micromodel-js): float32 storage with float64 sums, so the two agree to float32
rounding. A base-size 8-layer encoder takes about 160 ms for a 30-token text on one Apple M-series core.

## How it's checked

`go test` runs micromodel-spec `encoder/conformance` (in `testdata/`, copied from micromodel-spec 3505486) against
`tiny.mmenc`: token ids exactly, embeddings within 1e-4, logits within 1e-3 with equal argmax.

The format and forward pass are specified in micromodel-spec `encoder/SPEC.md`.
