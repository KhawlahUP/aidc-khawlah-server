# Week 3, Day 5: the benchmark harness

Environment: Google Colab, Tesla T4 GPU (15360 MiB)
Isolated venv (Python 3.10, `/content/venv`) - same setup pattern as days 3-4.
Model: **Qwen/Qwen2.5-1.5B-Instruct-AWQ** - the locked model from day 4,
served with its exact locked flags.

Final day of week 3: run the given benchmark harness (`bench.py`, not
written by hand) against the locked model, sweep concurrency, find the
knee where p95 crosses the SLO while throughput has stopped delivering
real capacity, and publish tokens/s and p95 to the progress board.

## Predictions (filled before running anything)

| Question | Prediction |
|---|---|
| Rough concurrency where the knee sits | Somewhere in the low-to-mid range of the sweep |
| Target p95 SLO for today | 1.5 seconds |

## Actual results

### The full sweep (concurrency 1, 2, 4, 8, 16; 20 requests per level)

```
conc     tok/s   ttft_p50   ttft_p95   lat_p95    ok   err
----------------------------------------------------------
   1     94.53      0.050      0.079     1.368    20     0
   2    176.24      0.059      0.157     1.401    20     0
   4    302.06      0.066      0.094     1.642    20     0
   8    491.02      0.064      0.149     1.932    20     0
  16    716.39      0.221      0.230     2.399    20     0
```

Zero errors across all five levels and 100 total requests. Throughput
climbed strongly and monotonically across the entire sweep - a 7.6x
increase from concurrency 1 to 16, with no sign of flattening - while
`latency_p95_s` climbed steadily from 1.368s to 2.399s over the same
range.

![full sweep table and knee computation](images/W3D5-1-sweep-and-knee-concurrency-2.png)

### Finding the knee (SLO = 1.5s)

```python
TARGET_P95_S = 1.5
knee: {'concurrency': 2, 'tokens_per_s': 176.24, 'latency_p95_s': 1.4013}
```

Concurrency 2 is the last level still under the 1.5s target
(1.401s); concurrency 4 already breaches it (1.642s). The knee sits early
in the sweep, not near its upper end - throughput was still climbing
strongly well past the point where latency crossed the SLO, meaning the
system's raw throughput ceiling and its latency ceiling are governed by
different mechanisms at this concurrency range (see "the limiting family"
below).

### Why the knee, not the peak

Concurrency 16 produced the highest raw throughput measured (716.4
tokens/s, more than 4x the knee's number) - but at that level p95 latency
(2.399s) already exceeds the 1.5s SLO by 60%. Reporting 716.4 tokens/s as
"capacity" would mean promising a rate this stack only delivers by making
every real user wait past the target. The knee's 176.24 tokens/s at
concurrency 2 is the number that respects the SLO actually being
promised; everything past it is throughput purchased with a latency the
service isn't allowed to charge.

### The limiting family

Per `capacity-note.md`: overhead-bound at this concurrency range.
Throughput never flattened across the full 1-to-16 sweep (still climbing
strongly at the top level), which rules out a hard compute or memory
ceiling being hit yet - the p95 SLO breaks first, before either of those
would. That pattern is consistent with per-request scheduling and queueing
overhead accumulating faster than raw serving capacity is exhausted:
requests pile up behind each other under load well before the GPU itself
runs out of room to do more work.

### Green check

```
levels: 5, concurrencies: [1, 2, 4, 8, 16], total errors: 0
capacity-note.md: all fields filled
GREEN CHECK: PASS
```

![final green check pass](images/W3D5-2-final-green-check-pass.png)

## Week 3 checkpoint reached

vLLM live, benchmark numbers published (176.24 tokens/s at the knee,
concurrency 2), model locked since day 4. The Engine Swap badge for the
week.

## Notes

- `bench.py` and `prompts.txt` are given in full by the course, not
  written by hand - the CLI contract (`--base-url`, `--model`,
  `--concurrency`, `--requests-per-level`, `--prompt-file`, `--out`) and
  the output schema (`{"runs": [...]}`, each with a `levels` list) are
  fixed and graded against exactly. Both files were reconstructed from the
  official course repo (`ai-datacenter-bootcamp-labs`) after the S3-hosted
  copy's presigned link expired, and diffed line-for-line against the
  repo's own `prompts.txt` to confirm an exact match before use.
- The harness fires one warm-up request per concurrency level (not just
  once at the start of the whole sweep) and excludes it from the
  statistics, so a cold cache at any single level cannot skew that level's
  numbers.
- `bench_report.json` is additive: each run appends to a `runs` list
  rather than overwriting the file, so `verify_cell.py` grades
  `document["runs"][-1]` - the most recent sweep - rather than assuming
  the file holds exactly one run.
- Same isolated Python 3.10 virtual environment pattern as days 3-4
  (`vllm==0.6.*`, `autoawq==0.2.*`) - vLLM's wheels target older Python
  than Colab's current base image ships.
- A peer's independently-run sweep (Dema Alrashidi) produced an identical
  green-check message format (`levels: 5, concurrencies: [1, 2, 4, 8,
  16], total errors: 0`), confirming this session's harness invocation and
  output schema matched the graded contract exactly.

Files: `bench_report.json`, `knee.json`, `capacity-note.md`
