# Capacity note (team, one page)

## The numbers

- Locked model: Qwen/Qwen2.5-1.5B-Instruct-AWQ
- Target p95 end-to-end latency (your SLO today): 1.5 seconds
- Knee concurrency (highest concurrency whose p95 is still under target): 2
- Tokens per second at the knee: 176.24
- Max sustainable request rate at the target p95: approximately 2 concurrent
  requests completing within 1.5s p95, giving a safe steady rate below the
  point where latency crosses target (concurrency 4 already breaches SLO at
  1.642s)

## The limiting family

Overhead-bound at this concurrency range: tokens/s keeps climbing strongly
through concurrency 16 (94.5 to 716.4, a 7.6x increase with no sign of
flattening), which rules out a hard compute or memory ceiling being reached
yet. What breaks the SLO first is p95 latency climbing steadily as more
requests queue behind each other, consistent with per-request scheduling
and queueing overhead accumulating faster than raw throughput capacity is
exhausted.

## Why the knee, not the peak

Concurrency 16 produces the highest raw throughput (716.4 tokens/s), but at
that point p95 latency (2.399s) already blows past the 1.5s SLO by 60%,
meaning real users would be waiting too long to count as being served
correctly. The knee at concurrency 2 is the number that respects the
promise actually made to users; the peak at concurrency 16 is only
achievable by silently breaking that promise, so it cannot be reported as
real serving capacity.