#!/usr/bin/env bash
set -euo pipefail

# Create Argo Workflows namespace
kubectl create namespace argo-workflows

# Install via Helm (lightweight defaults)
helm repo add argo https://argoproj.github.io/argo-helm >/dev/null
helm repo update >/dev/null
helm upgrade --install argo-workflows argo/argo-workflows \
  --namespace argo-workflows \
  --set server.enabled=true \
  --set server.authModes\[0\]=server \
  --set workflow.rbac.create=true \
  --set server.serviceType=NodePort \
  --set server.serviceNodePort=30600 

# Validate 
kubectl get all -n argo-workflows

# Print the NodePort for the UI
echo "Argo Workflows UI : http://127.0.0.1:30600"
