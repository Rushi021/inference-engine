# Phase 6 — Live GKE Autoscaling Demo (captured run)

Real autoscaling run on Google Kubernetes Engine. The model server was containerized (Cloud Build),
deployed to GKE behind a LoadBalancer, and driven with sustained load to trigger the Horizontal Pod
Autoscaler and the cluster (node) autoscaler.

> GPU note: this ran on **CPU nodes** (`e2-standard-4`) because the project's GPU quota was not yet
> granted (new account, no usage history). The autoscaling mechanism — HPA scaling pods on CPU
> utilization and the cluster autoscaler adding nodes — is identical regardless of CPU vs GPU. The
> model is small (Qwen2.5-0.5B), so it serves fine on CPU.

## Setup

- **Image:** built on Cloud Build (`deploy/Dockerfile.cpu`) → Artifact Registry.
- **Cluster:** `infer-cpu`, zonal (`us-central1-a`), node pool `e2-standard-4`, autoscaling 1–3 nodes.
- **Workload:** `deploy/gke-cpu/` — Deployment (1 CPU request/pod), LoadBalancer Service, HPA
  (target 60% CPU, min 1, max 4).
- **Load:** `scripts/gke_load.py` — 24 concurrent clients, 128-token generations, sustained.

## What happened (sampled every 20s)

```
time      hpa cpu      replicas  running_pods  ready_nodes
00:20:00  0%/60%       1         1             1     <- idle baseline
00:20:22  4%/60%       1         1             1
00:20:43  199%/60%     4         3             1     <- load hits; HPA jumps to max replicas
00:21:25  105%/60%     4         3             2     <- cluster autoscaler adds a 2nd node
00:21:46  91%/60%      4         4             2     <- 4th pod scheduled on the new node
00:22:29  124%/60%     4         4             2
00:24:36  147%/60%     4         4             2     <- steady state, capped at maxReplicas=4
```

**Pods spread across two nodes:**
```
inference-engine-...-7pb2l   Running   gke-infer-cpu-default-pool-...-dnr6   <- new node
inference-engine-...-nks46   Running   gke-infer-cpu-default-pool-...-6jd8
inference-engine-...-pzjrq   Running   gke-infer-cpu-default-pool-...-6jd8
inference-engine-...-z72bj   Running   gke-infer-cpu-default-pool-...-6jd8
```

**HPA event:**
```
Normal  SuccessfulRescale  horizontal-pod-autoscaler
  New size: 4; reason: cpu resource utilization (percentage of request) above target
```

## What this demonstrates

1. **Two-layer autoscaling.** The HPA scaled *pods* 1→4 when CPU crossed 60% of request; the cluster
   autoscaler then added a *node* because 4 pods (4 CPU requested) didn't fit on one `e2-standard-4`.
2. **The maxReplicas cap held** — replicas stopped at 4 even as CPU stayed >100%, which is the
   intended cost guardrail (more demand would queue rather than scale unbounded).
3. **Zero-downtime scale-out** — new pods joined the LoadBalancer rotation via readiness probes while
   the original pod kept serving.

## Production note

This scales on **CPU%** for simplicity (no extra metrics plumbing). A production LLM server would
scale on a serving signal — request queue depth or tokens/sec — exposed as a custom metric via the
Prometheus Adapter or Cloud Monitoring, because GPU-bound work doesn't always track CPU. The HPA
manifest is one `metric:` block away from that change.

## Teardown

After capturing this, every resource was deleted (cluster, node pools, LoadBalancer) so nothing
keeps billing — see `deploy/teardown.sh`.
