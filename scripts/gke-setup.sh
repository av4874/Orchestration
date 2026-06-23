#!/usr/bin/env bash
# Adversarial Guardrail — GKE + Jenkins + Argo one-time setup
# Run from a machine with gcloud CLI authenticated.
# Prereqs: gcloud auth login && gcloud auth application-default login

set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-adversarial-guardrail}"
CLUSTER_NAME="guardrail-cluster"
ZONE="us-central1-a"
GITHUB_REPO="${GITHUB_REPO:-}"           # owner/repo  e.g. Builder117/enterprise
HF_TOKEN="${HF_TOKEN:-}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
ARGO_TOKEN="${ARGO_TOKEN:-}"

# ── 1. GCP project + APIs ─────────────────────────────────────────────────────
echo "==> Configuring GCP project: $PROJECT_ID"
gcloud projects create "$PROJECT_ID" --name="Adversarial Guardrail" 2>/dev/null || true
gcloud config set project "$PROJECT_ID"

gcloud services enable \
  container.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com

# ── 2. GKE cluster ────────────────────────────────────────────────────────────
echo "==> Creating GKE cluster: $CLUSTER_NAME"
gcloud container clusters create "$CLUSTER_NAME" \
  --zone "$ZONE" \
  --num-nodes 2 \
  --machine-type e2-standard-4 \
  --disk-size 50GB \
  --enable-autoscaling --min-nodes 1 --max-nodes 4

gcloud container clusters get-credentials "$CLUSTER_NAME" --zone "$ZONE"

echo "==> Cluster nodes:"
kubectl get nodes

# ── 3. Namespaces ─────────────────────────────────────────────────────────────
kubectl create namespace jenkins 2>/dev/null || true
kubectl create namespace argo 2>/dev/null || true

# ── 4. K8s secrets ───────────────────────────────────────────────────────────
# Requires: HF_TOKEN, GITHUB_TOKEN, GITHUB_REPO, ARGO_TOKEN set in env
# ARGO_TOKEN is set after Argo install (step 6); re-run this block then.

_b64() { echo -n "$1" | base64 -w0; }

ARGO_URL="${ARGO_URL:-http://argo-server.argo.svc.cluster.local:2746}"
STAGING_URL="${STAGING_URL:-http://guardrail-staging.jenkins.svc.cluster.local:7860}"
JENKINS_ADMIN_PASSWORD="${JENKINS_ADMIN_PASSWORD:-$(openssl rand -hex 16)}"

echo "Jenkins admin password: $JENKINS_ADMIN_PASSWORD"
echo "(save this — not stored elsewhere)"

kubectl -n jenkins create secret generic pipeline-secrets \
  --from-literal=hf-token="$HF_TOKEN" \
  --from-literal=github-token="$GITHUB_TOKEN" \
  --from-literal=github-repo="$GITHUB_REPO" \
  --from-literal=argo-token="${ARGO_TOKEN:-placeholder}" \
  --from-literal=argo-url="$ARGO_URL" \
  --from-literal=staging-url="$STAGING_URL" \
  --from-literal=jenkins-admin-password="$JENKINS_ADMIN_PASSWORD" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl -n argo create secret generic pipeline-secrets \
  --from-literal=github-token="$GITHUB_TOKEN" \
  --from-literal=github-repo="$GITHUB_REPO" \
  --from-literal=staging-url="$STAGING_URL" \
  --dry-run=client -o yaml | kubectl apply -f -

# ── 5. Jenkins deploy ─────────────────────────────────────────────────────────
echo "==> Deploying Jenkins"

# ConfigMap for JCasC
kubectl -n jenkins create configmap jenkins-casc \
  --from-file=jenkins.yaml=jenkins/casc/jenkins.yaml \
  --dry-run=client -o yaml | kubectl apply -f -

# Patch deployment to mount JCasC configmap + env vars
cat <<'PATCH' > /tmp/jenkins-casc-patch.yaml
spec:
  template:
    spec:
      containers:
        - name: jenkins
          env:
            - name: CASC_JENKINS_CONFIG
              value: /var/jenkins_home/casc_configs/
            - name: JENKINS_ADMIN_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: pipeline-secrets
                  key: jenkins-admin-password
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: pipeline-secrets
                  key: hf-token
            - name: GITHUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: pipeline-secrets
                  key: github-token
            - name: GITHUB_REPO
              valueFrom:
                secretKeyRef:
                  name: pipeline-secrets
                  key: github-repo
            - name: ARGO_TOKEN
              valueFrom:
                secretKeyRef:
                  name: pipeline-secrets
                  key: argo-token
          volumeMounts:
            - name: casc-config
              mountPath: /var/jenkins_home/casc_configs
      volumes:
        - name: casc-config
          configMap:
            name: jenkins-casc
PATCH

kubectl apply -f jenkins/k8s/jenkins-deployment.yaml
kubectl apply -f jenkins/k8s/jenkins-service.yaml
kubectl -n jenkins patch deployment jenkins --patch-file /tmp/jenkins-casc-patch.yaml

echo "==> Waiting for Jenkins pod..."
kubectl -n jenkins rollout status deployment/jenkins --timeout=300s

JENKINS_IP=$(kubectl -n jenkins get svc jenkins -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "Jenkins URL: http://$JENKINS_IP"
echo "Login: admin / $JENKINS_ADMIN_PASSWORD"

# ── 6. Argo Workflows ─────────────────────────────────────────────────────────
echo "==> Installing Argo Workflows"
ARGO_VERSION="v3.5.8"
kubectl apply -n argo -f "https://github.com/argoproj/argo-workflows/releases/download/${ARGO_VERSION}/install.yaml"
kubectl -n argo rollout status deployment/argo-server --timeout=300s

# Get Argo token and update secret
ARGO_TOKEN=$(kubectl -n argo exec deploy/argo-server -- argo auth token 2>/dev/null || echo "")
if [ -n "$ARGO_TOKEN" ]; then
  kubectl -n jenkins create secret generic pipeline-secrets \
    --from-literal=argo-token="$ARGO_TOKEN" \
    --dry-run=client -o yaml | kubectl apply -f -
  echo "Argo token updated in pipeline-secrets"
else
  echo "WARNING: Could not retrieve Argo token — update pipeline-secrets manually"
fi

# Patch Argo server to disable auth for internal cluster access
kubectl -n argo patch deployment argo-server \
  --type='json' \
  -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--auth-mode=server"}]' \
  2>/dev/null || true

ARGO_IP=$(kubectl -n argo get svc argo-server -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
echo "Argo UI: http://$ARGO_IP:2746"

# ── 7. GitHub webhook ─────────────────────────────────────────────────────────
echo "==> Configuring GitHub webhook"
if [ -n "$JENKINS_IP" ] && [ -n "$GITHUB_TOKEN" ] && [ -n "$GITHUB_REPO" ]; then
  WEBHOOK_SECRET=$(openssl rand -hex 20)
  curl -s -X POST \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github.v3+json" \
    "https://api.github.com/repos/${GITHUB_REPO}/hooks" \
    -d "{
      \"name\": \"web\",
      \"active\": true,
      \"events\": [\"push\", \"pull_request\"],
      \"config\": {
        \"url\": \"http://${JENKINS_IP}/github-webhook/\",
        \"content_type\": \"json\",
        \"secret\": \"${WEBHOOK_SECRET}\"
      }
    }" | python3 -c "import json,sys; r=json.load(sys.stdin); print('Webhook ID:', r.get('id', r.get('message', 'error')))"
else
  echo "SKIP: Set GITHUB_TOKEN and GITHUB_REPO to auto-configure webhook"
  echo "Manual: GitHub repo → Settings → Webhooks → Add:"
  echo "  Payload URL: http://<JENKINS_IP>/github-webhook/"
  echo "  Content type: application/json"
  echo "  Events: Pushes + Pull requests"
fi

# ── 8. Apply Argo workflow templates ─────────────────────────────────────────
echo "==> Applying Argo workflow templates"
kubectl apply -n argo -f argo/workflows/

# ── 9. Staging scorer deployment ──────────────────────────────────────────────
echo "==> Deploying staging scorer (STAGING_URL endpoint)"
kubectl apply -f jenkins/k8s/staging-deployment.yaml
kubectl -n jenkins rollout status deployment/guardrail-staging --timeout=120s || echo "WARNING: staging deploy pending"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Segment 6 Setup Complete"
echo "=========================================="
echo "Jenkins:  http://$JENKINS_IP  (admin / $JENKINS_ADMIN_PASSWORD)"
echo "Argo UI:  http://$ARGO_IP:2746"
echo ""
echo "Next: push to attacks/ branch → GHA runs agents → merge PR → Jenkins triggers"
echo "=========================================="
