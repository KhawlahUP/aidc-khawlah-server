# Week 4, Day 2: make it self-healing

Environment: team pod t15, NVIDIA RTX A6000, k3s, namespace `khawlah`. Second
day of week 4: yesterday's bare pod becomes a Deployment behind a Service,
gated by probes, and two claims get measured instead of watched - a killed
pod comes back by itself, and a rolling update drops zero requests.

## Predictions (filled before running anything)

| Question | Prediction |
|---|---|
| Requests failed killing one of two pods under a watching client | Zero - the Service should route around the dying pod while the Deployment replaces it |
| Failures during a rolling update with correct probes | Zero, if maxUnavailable:0 and readiness are both doing their job |
| Which probe gates /health's 503-until-loaded behaviour | Readiness - liveness is a different question (is this pod beyond saving), not "can it serve right now" |
| Failures under the team's constraint card | Zero, if the card is availability-first (A) |

## Actual results

### Step 1-2: Deployment, Service, and probes

```
deployment.apps/serving created
service/serving created
deployment "serving" successfully rolled out

NAME                       READY   STATUS    RESTARTS   AGE
serving-74b794f4c6-2m9z6   1/1     Running   0          18s
serving-74b794f4c6-785pl   1/1     Running   0          17s
```

Probes confirmed in the live spec after adding them:

```
Liveness:   http-get http://:8000/health delay=20s timeout=1s period=5s #failure=3
Readiness:  http-get http://:8000/health delay=0s timeout=1s period=2s #failure=2
```

A generous 20-second `initialDelaySeconds` on liveness is deliberate: too
short and the probe kills a pod that was merely slow to start, resetting it
into a restart loop before it ever gets the chance to become ready - the
same `CrashLoopBackOff` shape as day 1's pod-c, a different cause wearing
the same costume.

### Step 3-4: in-cluster prober and the kill demo

One of two running pods was deleted mid-watch:

```
serving-6f97d84f94-fh579   deleted
serving-6f97d84f94-6pvdc   Running   0   10s   (replacement, arrived automatically)
serving-6f97d84f94-qnx5z   Running   0   10m   (untouched)
```

```
PROBE RESULT ok=880 bad=0
```

Zero failures across 880 checks at 100ms intervals, spanning the pod
deletion and its automatic replacement. The Deployment controller noticed
desired=2, observed=1, and created a replacement without anyone watching -
exactly the shift the day is about: yesterday I was the restart policy,
today the controller is.

### Step 5: the rolling update, measured both ways

**With the full spec (probes + preStop present):**

```
PROBE RESULT ok=880 bad=0
```

**With `lifecycle.preStop` deliberately removed, same update pattern:**

```
PROBE RESULT ok=877 bad=3
```

Three failed requests appeared the moment the exit-grace period vanished -
close to the lab's own reference measurement of 2 failures in 45 seconds.
Removing `preStop` doesn't change whether a pod is Running when told to
stop; it stops accepting new work immediately, but the cluster still takes
a moment to remove its IP from the Service's endpoints list, and any
request landing in that gap fails. Restoring `preStop: {sleep: {seconds:
5}}` and confirming it in the live spec (`kubectl get deployment -o yaml`
showing the lifecycle block) brought the count straight back to zero on the
next run.

### Step 6: the team's constraint card

Card **A** (availability first): "the consumer's agent retries nothing; one
5xx is a failed demo." Settings unchanged from Step 1
(`maxUnavailable: 0, maxSurge: 1`) - card A's price is exactly the headroom
and slower roll this deployment already pays.

A fresh prober, a dedicated rolling update (`APP_VERSION=v4-cardA`), run
separately from Step 5's measurement:

```
PROBE RESULT ok=882 bad=0
```

Zero dropped requests under a real, fresh measurement - the card's owner
would sign off on this number without qualification.

### Green check

`verify.sh` doesn't trust any past run: it inspects the live spec for both
probes and the preStop hook, confirms `maxUnavailable: 0`, then runs its
own dedicated update (flipping `VERIFY_STAMP`) behind its own prober and
demands zero failures.

```
PROBE RESULT ok=440 bad=0
rolling update completed with 440 requests served and none dropped
GREEN CHECK: PASS
```

Files: `deployment.yaml`, `service.yaml`, `verify.sh`
