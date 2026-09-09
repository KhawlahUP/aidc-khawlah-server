# Resource Policy
Our policy is to protect latency-sensitive serving workloads with guaranteed
resources, throttle batch workloads with explicit CPU limits, and allow
low-priority dashboard workloads to use spare capacity.
## Serving
```yaml
resources:
  requests:
    cpu: "250m"
    memory: 512Mi
  limits:
    cpu: "1"
    memory: 3Gi
```
QoS class: Burstable
## Batch
```yaml
resources:
  requests:
    cpu: 100m
    memory: 128Mi
  limits:
    cpu: 500m
    memory: 512Mi
```
QoS class: Burstable
## Dashboard
No CPU or memory requests or limits.
QoS class: BestEffort
## Noisy-neighbour measurement
Unlimited burners: n=573, p95=3ms, fails=0.
CPU-limited burners at 500m: n=574, p95=3ms, fails=0.
In this experiment, the 500m CPU limit did not measurably change p95
latency, because serving already held a guaranteed 250m CPU request from
Step 1 regardless of whether the burners were limited. The kernel weighted
serving's requested share above twenty BestEffort/unlimited loops in both
runs. Still, batch remains the first workload we choose to throttle: an
explicit limit caps how much CPU any single batch worker can ever claim,
which protects against a scenario this specific test didn't create -
fewer, hungrier neighbours instead of twenty modest ones.
