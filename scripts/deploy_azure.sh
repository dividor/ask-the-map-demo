#!/usr/bin/env bash
# Deploy the demo to Azure Container Apps (consumption plan, scales to zero when idle).
#
# Needs: the Azure CLI logged in (`az login`), the containerapp extension, a .env with ASKMAP_DATA_ROOT and
# HF_HOME (where notebook 13's demo data and the text encoder are), and the Hugging Face token in HF_TOKEN or the
# local login. Nothing in this script is specific to one subscription: set the variables below in your shell.
#
#   AZ_RESOURCE_GROUP   resource group to create or reuse        (default askmap-demo)
#   AZ_LOCATION         region                                   (default eastus2)
#   AZ_REGISTRY         container registry name, globally unique (default askmap + 6 random characters)
#   AZ_APP              container app name                       (default askmap-demo)
#   AZ_CPU / AZ_MEMORY  per replica                              (default 2 / 4Gi)
#   AZ_DATA_ROOT        data folder, if not the one in .env
#
# Steps: stage the data under deploy/ (ignored by git), build the image in the registry (no local Docker
# needed), create the environment and the app (or update the app's image if it exists), store the token as a
# secret, set a startup probe long enough for the cold start, and print the URL. Running it again rebuilds the
# image and rolls the app to it.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
# AZ_DATA_ROOT overrides ASKMAP_DATA_ROOT, for deploying from a data folder other than the one .env names
DATA_ROOT="${AZ_DATA_ROOT:-${ASKMAP_DATA_ROOT:-}}"
: "${DATA_ROOT:?set ASKMAP_DATA_ROOT in .env or AZ_DATA_ROOT}"
: "${HF_HOME:?set HF_HOME in .env}"

RG="${AZ_RESOURCE_GROUP:-askmap-demo}"
LOC="${AZ_LOCATION:-eastus2}"
APP="${AZ_APP:-askmap-demo}"
CPU="${AZ_CPU:-2}"
MEM="${AZ_MEMORY:-4Gi}"
ENVNAME="$APP-env"
if [ -z "${AZ_REGISTRY:-}" ]; then
  AZ_REGISTRY="askmap$(LC_ALL=C tr -dc a-z0-9 </dev/urandom | head -c 6)"
  echo "registry name: $AZ_REGISTRY (set AZ_REGISTRY to reuse it next time)"
fi
IMAGE="$AZ_REGISTRY.azurecr.io/askmap-demo:latest"
TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
: "${TOKEN:?no Hugging Face token: set HF_TOKEN or run hf auth login}"

echo "== staging the data under deploy/ (not committed)"
mkdir -p deploy/data deploy/hf_cache
rsync -a --delete "$DATA_ROOT/demo/" deploy/data/demo/
rsync -a --delete --include='*.nc' --include='manifest.json' --include='*.json' --exclude='*' \
  "$DATA_ROOT/power/" deploy/data/power/
# the encoder's snapshot with its links resolved (the cache's blobs/ folder would double it), weights as
# safetensors only (the cache may also hold .bin and .h5 copies of the same weights)
rsync -aL --delete --exclude='blobs/' --exclude='*.bin' --exclude='*.h5' --exclude='*.msgpack' --exclude='*.ot' \
  "$HF_HOME/models--openai--clip-vit-large-patch14/" deploy/hf_cache/models--openai--clip-vit-large-patch14/
du -sh deploy/data deploy/hf_cache

echo "== resource group, registry, environment"
az group create -n "$RG" -l "$LOC" -o none
az acr create -n "$AZ_REGISTRY" -g "$RG" --sku Basic -o none 2>/dev/null || true
az containerapp env create -n "$ENVNAME" -g "$RG" -l "$LOC" -o none 2>/dev/null || true

echo "== building the image in the registry"
az acr build -r "$AZ_REGISTRY" -g "$RG" -t askmap-demo:latest --platform linux/amd64 .

echo "== the app"
# The app pulls the image with its own managed identity (given AcrPull on the registry), so no registry password
# is stored anywhere. The token is a Container Apps secret.
if az containerapp show -n "$APP" -g "$RG" -o none 2>/dev/null; then
  az containerapp update -n "$APP" -g "$RG" --image "$IMAGE" --revision-suffix "r$(date +%H%M%S)" -o none
else
  az containerapp create -n "$APP" -g "$RG" --environment "$ENVNAME" --image "$IMAGE" \
    --registry-server "$AZ_REGISTRY.azurecr.io" --registry-identity system \
    --secrets "hf-token=$TOKEN" --env-vars "HF_TOKEN=secretref:hf-token" \
    --cpu "$CPU" --memory "$MEM" --min-replicas 0 --max-replicas 1 \
    --ingress external --target-port 8765 -o none
fi

# Loading the tiles and the text encoder takes a few minutes on a cold start, longer than the default startup
# probe allows, so the probes are set from a template (the CLI has no flags for them). No secret is in it.
PROBES=$(mktemp)
{
  echo "properties:"
  echo "  template:"
  echo "    scale: {minReplicas: 0, maxReplicas: 1}"
  echo "    containers:"
  echo "      - name: $APP"
  echo "        image: $IMAGE"
  echo "        env:"
  echo "          - {name: HF_TOKEN, secretRef: hf-token}"
  echo "        resources: {cpu: $CPU, memory: $MEM}"
  echo "        probes:"
  echo "          - {type: Startup, tcpSocket: {port: 8765}, initialDelaySeconds: 20, periodSeconds: 10, failureThreshold: 90, timeoutSeconds: 5}"
  echo "          - {type: Liveness, httpGet: {path: /api/meta, port: 8765}, periodSeconds: 30, failureThreshold: 5, timeoutSeconds: 10}"
} > "$PROBES"
az containerapp update -n "$APP" -g "$RG" --yaml "$PROBES" -o none
rm -f "$PROBES"
echo "== deployed: https://$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
