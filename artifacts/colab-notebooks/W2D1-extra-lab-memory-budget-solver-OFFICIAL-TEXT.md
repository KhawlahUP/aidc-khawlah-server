# Extra Lab W2D1: the memory budget solver

*Authored by Ahmed (TA), 2026-08-02. Integrated with a machine verifier 2026-08-07.*

Start:      Morning's "numbers on disk" lecture. No GPU needed -- this is pure
            arithmetic against real model architecture specs.
Objective:  Given a GPU memory budget, a concurrent-user count, and a context
            length, compute -- from architecture fields alone, never from a
            model's marketing name -- the single largest model+precision
            combination that actually fits.

Time: about 90 minutes. On D1 this is not an extra. It is the second half of
the afternoon, scheduled by the day's own lab README, and it goes deeper on the
same formula you measured in the first half.

This is not "measure one model on one GPU." This is "solve, on paper first,
which of five real models fits a budget you're given" -- the question a
platform engineer actually gets asked before any GPU is touched.

## Predict (by hand)

Credited for handing it in, never marked right or wrong - hedged guesses teach nothing, and nothing here is graded for accuracy.

Fill this in before you write any code.

- A model's `config.json` never states its total parameter count directly.
  Which fields would you need to derive it? Name at least four.
- Two models are both called "3B" by their filenames. Would you expect their
  computed parameter counts to match exactly? Why or why not?
- If you assume a model does NOT tie its input and output embeddings, but it
  actually does, will your computed parameter count read high or low?
- Hand in the card.

## The delta

### Step 1: the catalog (given, about 5 min to read)

You are given five real models' architecture fields below -- the same fields
that live in every Hugging Face `config.json`. This is a small local catalog so
the lab runs with no network dependency; Stretch below covers pulling the real
file for a model of your choice.

```python
CATALOG = {
    "Qwen2.5-0.5B-Instruct": {"hidden_size": 896, "num_hidden_layers": 24, "num_attention_heads": 14, "num_key_value_heads": 2, "intermediate_size": 4864, "vocab_size": 151936, "tie_word_embeddings": True},
    "Qwen2.5-1.5B-Instruct": {"hidden_size": 1536, "num_hidden_layers": 28, "num_attention_heads": 12, "num_key_value_heads": 2, "intermediate_size": 8960, "vocab_size": 151936, "tie_word_embeddings": True},
    "Qwen2.5-3B-Instruct": {"hidden_size": 2048, "num_hidden_layers": 36, "num_attention_heads": 16, "num_key_value_heads": 2, "intermediate_size": 11008, "vocab_size": 151936, "tie_word_embeddings": True},
    "Llama-3.2-1B-Instruct": {"hidden_size": 2048, "num_hidden_layers": 16, "num_attention_heads": 32, "num_key_value_heads": 8, "intermediate_size": 8192, "vocab_size": 128256, "tie_word_embeddings": True},
    "Llama-3.2-3B-Instruct": {"hidden_size": 3072, "num_hidden_layers": 28, "num_attention_heads": 24, "num_key_value_heads": 8, "intermediate_size": 8192, "vocab_size": 128256, "tie_word_embeddings": True},
}
```

These fields are reconstructed from each model's real published config, not a
live download -- close enough that the derived parameter counts below match
each model's real published size almost exactly, which you will check in
Step 2.

### Step 2: derive the parameter count (about 30 min, this is the lab)

A standard Llama/Qwen-style transformer block has four attention projections
and a three-matrix gated MLP. Nothing here is given as a number; every count
below comes from multiplying two dimensions together.

```python
def count_params(cfg):
    h = cfg["hidden_size"]
    L = cfg["num_hidden_layers"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    inter = cfg["intermediate_size"]
    vocab = cfg["vocab_size"]
    tied = cfg.get("tie_word_embeddings", False)
    head_dim = h // n_heads

    # attention: q and o are full hidden_size x hidden_size; k and v are
    # narrower because of grouped-query attention (fewer kv heads than query heads)
    q = h * h
    k = h * (n_kv * head_dim)
    v = h * (n_kv * head_dim)
    o = h * h
    attn = q + k + v + o

    # gated MLP (SwiGLU-style): three matrices, not two
    gate = h * inter
    up = h * inter
    down = inter * h
    mlp = gate + up + down

    per_layer = attn + mlp
    embed_matrices = 1 if tied else 2   # tied models reuse one matrix for input+output
    total = L * per_layer + embed_matrices * vocab * h
    return total


for name, cfg in CATALOG.items():
    p = count_params(cfg)
    print(f"{name}: {p/1e9:.2f}B params")
```

Run it. Compare each result to the number in the model's own name. They should
land within a rounding error of each other -- 1.5B computes to about 1.54B, 3B
computes to about 3.09-3.21B. If yours are close, your formula is right.

**Now the gotcha, on purpose:** flip `tie_word_embeddings` to `False` for
Qwen2.5-1.5B-Instruct and rerun just that one model. You should see about
1.78B instead of 1.54B -- a 15% overestimate. This is not an error in your
formula; it is exactly what happens when a model ties its input and output
embedding matrices to save parameters and your code assumes it doesn't. Real
`config.json` files state `tie_word_embeddings` explicitly; this is why you
read it instead of guessing.

### Step 3: KV cost per token (about 10 min -- this formula is not new)

You already have this from earlier in the week: 2 (K and V) times layers times
kv-heads times head-dim times bytes-per-element.

```python
def kv_bytes_per_token(cfg, kv_dtype_bytes=2):
    h = cfg["hidden_size"]
    L = cfg["num_hidden_layers"]
    n_kv = cfg["num_key_value_heads"]
    n_heads = cfg["num_attention_heads"]
    head_dim = h // n_heads
    return 2 * L * n_kv * head_dim * kv_dtype_bytes
```

Simplifying assumption, stated up front: the KV cache is assumed fp16
regardless of what precision the weights are quantised to. Real serving stacks
sometimes quantise KV too; that is out of scope here.

### Step 4: solve the budget (about 25 min)

Given a budget, concurrent-user count, and context length, compute total GB for
every model at every precision, and pick the largest model that still fits.

```python
PRECISIONS = {"fp16": 2.0, "int8": 1.0, "int4": 0.5}
OVERHEAD_GB = 1.5   # fixed CUDA/allocator overhead, midpoint of W2D1's 1-2 GB range

def solve(budget_gb, concurrent_users, context_tokens):
    rows = []
    for name, cfg in CATALOG.items():
        params = count_params(cfg)
        kv_per_tok = kv_bytes_per_token(cfg)
        for prec, bytes_per_param in PRECISIONS.items():
            weight_gb = params * bytes_per_param / 1e9
            kv_gb = kv_per_tok * context_tokens * concurrent_users / 1e9
            total_gb = weight_gb + kv_gb + OVERHEAD_GB
            fits = total_gb <= budget_gb
            rows.append({
                "model": name, "precision": prec, "params": params,
                "weight_gb": round(weight_gb, 2), "kv_gb": round(kv_gb, 2),
                "total_gb": round(total_gb, 2), "fits": fits,
            })
    fitting = [r for r in rows if r["fits"]]
    best = max(fitting, key=lambda r: r["params"]) if fitting else None
    return rows, best

rows, best = solve(budget_gb=16, concurrent_users=4, context_tokens=4096)
for r in sorted(rows, key=lambda r: -r["params"]):
    mark = "FITS" if r["fits"] else "----"
    print(f"{mark}  {r['model']:25s} {r['precision']:5s}  total={r['total_gb']:6.2f}GB")
print("BEST:", best)
```

`best` should be the largest-parameter combination whose `total_gb` does not
exceed the budget. If no combination fits, `best` is `None` -- handle that case,
do not crash on it.

### Step 5: write budget_solution.json

```python
import json
budget_gb, concurrent_users, context_tokens = 16, 4, 4096
rows, best = solve(budget_gb, concurrent_users, context_tokens)
out = {
    "budget_gb": budget_gb, "concurrent_users": concurrent_users,
    "context_tokens": context_tokens, "best": best,
    "all_combinations": rows,
}
with open("budget_solution.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(best, indent=2))
```

## Verify (green check)

Run the verifier next to your `budget_solution.json` (or paste it into a cell):

```bash
python verify.py
```

It recomputes every combination from the same catalog independently and checks
your numbers against it, so a solver that is consistently wrong cannot pass by
being consistent. Expected final line: `GREEN CHECK: PASS`.

## Stretch

Swap the local `CATALOG` for a real download: use `huggingface_hub`'s
`hf_hub_download` to pull the actual `config.json` for a model of your choice,
parse the same fields out of it, and confirm your formula's output against the
model card's stated parameter count. This is the same computation, against a
file you did not get handed.

## Failure modes

- **Computed params are about 15-20% too high.** Cause: `tie_word_embeddings`
  was read as `False` (or defaulted) for a model that actually ties its
  embeddings. Fix: check the field is actually present and read correctly in
  your catalog entry -- this is the Step 2 gotcha, not a bug in the formula.
- **`best` is always `None`.** Cause: budget is genuinely too small for any
  catalog entry, or `OVERHEAD_GB` and KV cost are being double-counted somewhere.
  Fix: print every row's `total_gb` and sanity-check the smallest one by hand.
- **KV cost dwarfs everything else and nothing fits at high concurrency.**
  This is expected at large `concurrent_users x context_tokens` products, not a
  bug -- it is the same lesson as the KV-cache section of this week's lecture,
  just applied to many users instead of one.
- **head_dim comes out as a non-integer.** Cause: `hidden_size` is not evenly
  divisible by `num_attention_heads` in a catalog entry you edited or added.
  Fix: real models always divide evenly; check the entry against the model's
  actual config rather than adjusting the formula.
