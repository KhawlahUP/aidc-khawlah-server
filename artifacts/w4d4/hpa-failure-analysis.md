# Why CPU-based HPA fails for a GPU-bound vLLM engine

Today's HPA scales on CPU utilisation, and that works because the echo
backend is CPU-bound: every request Locust sent burned actual CPU cycles
generating a response, so CPU% rose and fell exactly with load.

A real vLLM engine serving Qwen2.5-1.5B-AWQ on the shared GPU does not
share that property. The engine's Python process itself uses very little
CPU per request - the expensive work happens on the GPU (matrix multiplies,
attention, KV cache reads), not on the CPU cores. Under heavy concurrent
load, GPU utilisation and GPU memory pressure can climb toward saturation
while CPU usage barely moves, because the CPU is mostly idle, just waiting
on the GPU and shuffling small amounts of data in and out.

A CPU-based HPA watching that engine would see flat, low CPU% throughout
a real traffic spike and never scale out - exactly the failure mode this
week's own lecture material warned about: the right signal for an
echo/CPU-bound backend is the wrong signal for a GPU-bound one.

The correct signal for a real vLLM deployment would be a GPU-native
metric - GPU utilisation percentage, GPU memory pressure, or an
application-level metric like queue depth or requests-in-flight exposed
through vLLM's own metrics endpoint and a custom metrics adapter - not
`cpu` utilisation. That is beyond today's scope: today's chart and lab are
built specifically to teach the HPA's reconcile loop with a signal (CPU)
that is trivially correct to reason about, before week 5 introduces GPU
metrics through Prometheus/Grafana.
