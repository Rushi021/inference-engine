#!/usr/bin/env bash
# Delete every billable GCP resource from the GKE deploy. Run after any cloud session.
set -euo pipefail

ZONE="${ZONE:-us-central1-a}"
REGION="${REGION:-us-central1}"
CLUSTER="${CLUSTER:-infer-cpu}"

echo "1/3 deleting k8s workloads (releases the LoadBalancer)..."
kubectl delete -f deploy/gke-cpu/ --ignore-not-found || true   # frees the external IP / forwarding rules

echo "2/3 deleting the GKE cluster (removes node pools too)..."
gcloud container clusters delete "$CLUSTER" --zone "$ZONE" --quiet

echo "3/3 verifying nothing is left running..."
gcloud container clusters list
gcloud compute instances list
gcloud compute forwarding-rules list

# Optional (kept by default — negligible storage cost, lets you redeploy without rebuilding):
#   gcloud artifacts repositories delete inference --location "$REGION" --quiet
echo "done. If the lists above are empty, nothing is billing."
