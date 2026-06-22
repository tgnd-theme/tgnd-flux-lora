#!/bin/bash
#
# deploy-and-train.sh — Full training pipeline with safety checks.
#
# Enforces the correct order:
# 1. Git push (if there are changes)
# 2. Wait for Docker build to complete
# 3. Cycle workers (force fresh image pull)
# 4. Run pre-flight checks
# 5. Submit training job
# 6. Poll until complete
#
# Usage:
#   ./deploy-and-train.sh <escort_user_id> <trigger_word> [steps]
#
# Example:
#   ./deploy-and-train.sh 6 agatha_model 2000
#

set -euo pipefail

# ─── Config ───
# Set these env vars or they'll be fetched from staging wp-config.php
RUNPOD_API_KEY="${TGND_RUNPOD_API_KEY:-}"
HF_TOKEN="${TGND_HF_TOKEN:-}"
ANTHROPIC_KEY="${TGND_ANTHROPIC_API_KEY:-}"
ENDPOINT_ID="zcapgpo7w622dj"
WORKFLOW_NAME="Build & Push Training Docker Image"
SSH_TARGET="the-girl-next-doorcom@thegis.ssh.transip.me"
SSH_KEY="$HOME/.ssh/id_ed25519"

# ─── Args ───
ESCORT_USER_ID="${1:-}"
TRIGGER_WORD="${2:-}"
STEPS="${3:-2000}"

if [[ -z "$ESCORT_USER_ID" || -z "$TRIGGER_WORD" ]]; then
    echo "Usage: $0 <escort_user_id> <trigger_word> [steps]"
    echo "Example: $0 6 agatha_model 2000"
    exit 1
fi

echo "═══════════════════════════════════════════════════"
echo "  TGND LoRA Training Pipeline"
echo "  Escort: $ESCORT_USER_ID | Trigger: $TRIGGER_WORD | Steps: $STEPS"
echo "═══════════════════════════════════════════════════"

# ─── Fetch missing keys from staging ───
if [[ -z "$RUNPOD_API_KEY" || -z "$HF_TOKEN" || -z "$ANTHROPIC_KEY" ]]; then
    echo ""
    echo "▶ Fetching API keys from staging wp-config..."
    KEYS=$(ssh -i "$SSH_KEY" "$SSH_TARGET" "cd staging && wp eval \"
        echo 'RUNPOD=' . (defined('TGND_RUNPOD_API_KEY') ? TGND_RUNPOD_API_KEY : '') . PHP_EOL;
        echo 'HF=' . (defined('TGND_HF_TOKEN') ? TGND_HF_TOKEN : '') . PHP_EOL;
        echo 'ANTHROPIC=' . (defined('TGND_ANTHROPIC_API_KEY') ? TGND_ANTHROPIC_API_KEY : '') . PHP_EOL;
    \"" 2>/dev/null)
    [[ -z "$RUNPOD_API_KEY" ]] && RUNPOD_API_KEY=$(echo "$KEYS" | grep '^RUNPOD=' | cut -d= -f2)
    [[ -z "$HF_TOKEN" ]] && HF_TOKEN=$(echo "$KEYS" | grep '^HF=' | cut -d= -f2)
    [[ -z "$ANTHROPIC_KEY" ]] && ANTHROPIC_KEY=$(echo "$KEYS" | grep '^ANTHROPIC=' | cut -d= -f2)

    if [[ -z "$RUNPOD_API_KEY" ]]; then
        echo "  ✗ Could not get RunPod API key"
        exit 1
    fi
    echo "  ✓ Keys loaded from staging"
fi

# ─── Step 0: Pre-flight checks BEFORE anything else ───
echo ""
echo "▶ Step 0: Pre-flight checks..."

# Check HF token
HF_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $HF_TOKEN" "https://huggingface.co/api/whoami-v2")
if [[ "$HF_STATUS" != "200" ]]; then
    echo "  ✗ HuggingFace token invalid (HTTP $HF_STATUS)"
    exit 1
fi
echo "  ✓ HuggingFace token valid"

# Check Anthropic key
if [[ -n "$ANTHROPIC_KEY" ]]; then
    ANT_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -H "x-api-key: $ANTHROPIC_KEY" -H "anthropic-version: 2023-06-01" "https://api.anthropic.com/v1/models")
    if [[ "$ANT_STATUS" != "200" ]]; then
        echo "  ✗ Anthropic API key invalid (HTTP $ANT_STATUS)"
        exit 1
    fi
    echo "  ✓ Anthropic API key valid"
else
    echo "  ⚠ No Anthropic key set (captions will use fallback)"
fi

# Check RunPod API
RP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $RUNPOD_API_KEY" "https://api.runpod.ai/v2/$ENDPOINT_ID/health")
if [[ "$RP_STATUS" != "200" ]]; then
    echo "  ✗ RunPod endpoint unreachable (HTTP $RP_STATUS)"
    exit 1
fi
echo "  ✓ RunPod endpoint reachable"

# ─── Step 1: Push code if needed ───
echo ""
echo "▶ Step 1: Check for unpushed changes..."

if [[ -n "$(git status --porcelain handler_training.py Dockerfile.training 2>/dev/null)" ]]; then
    echo "  Uncommitted changes in handler files. Commit first!"
    exit 1
fi

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main 2>/dev/null || echo "none")

if [[ "$LOCAL" != "$REMOTE" ]]; then
    echo "  Pushing local changes to origin/main..."
    git push origin main
    echo "  ✓ Pushed"
    NEED_BUILD=true
else
    echo "  ✓ Already up to date with origin/main"
    NEED_BUILD=false
fi

# ─── Step 2: Wait for Docker build ───
if [[ "$NEED_BUILD" == "true" ]]; then
    echo ""
    echo "▶ Step 2: Waiting for Docker build..."

    # Wait for workflow to start
    sleep 10

    # Get latest run
    RUN_ID=$(gh run list --workflow="$WORKFLOW_NAME" --limit 1 --json databaseId --jq '.[0].databaseId')
    echo "  Build run: $RUN_ID"

    # Poll until done
    while true; do
        STATUS=$(gh run view "$RUN_ID" --json status,conclusion --jq '.status')
        if [[ "$STATUS" == "completed" ]]; then
            CONCLUSION=$(gh run view "$RUN_ID" --json conclusion --jq '.conclusion')
            if [[ "$CONCLUSION" == "success" ]]; then
                echo "  ✓ Docker build succeeded"
                break
            else
                echo "  ✗ Docker build failed ($CONCLUSION)"
                exit 1
            fi
        fi
        echo "  ... building ($STATUS)"
        sleep 30
    done
else
    echo ""
    echo "▶ Step 2: No code changes, skipping Docker build"
fi

# ─── Step 3: Cycle workers ───
if [[ "$NEED_BUILD" == "true" ]]; then
    echo ""
    echo "▶ Step 3: Cycling workers for fresh Docker image..."

    # Scale to 0
    curl -s -H "Content-Type: application/json" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        "https://api.runpod.io/graphql" \
        -d "{\"query\":\"mutation { saveEndpoint(input: { id: \\\"$ENDPOINT_ID\\\", name: \\\"tgnd-lora-trainer\\\", workersMin: 0, workersMax: 0 }) { id } }\"}" > /dev/null

    sleep 5

    # Scale back to max 1
    curl -s -H "Content-Type: application/json" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        "https://api.runpod.io/graphql" \
        -d "{\"query\":\"mutation { saveEndpoint(input: { id: \\\"$ENDPOINT_ID\\\", name: \\\"tgnd-lora-trainer\\\", workersMin: 0, workersMax: 1 }) { id } }\"}" > /dev/null

    echo "  ✓ Workers cycled"
else
    echo ""
    echo "▶ Step 3: No build needed, skipping worker cycle"
fi

# ─── Step 4: Get training ZIP and webhook info from WP ───
echo ""
echo "▶ Step 4: Preparing training job..."

# Check if training ZIP exists on HF
ZIP_NAME="${TRIGGER_WORD}_training.zip"
# We need the escort to already have a ZIP on HF or WP
# For now, we check HF
ZIP_URL="https://huggingface.co/JulioIglesiass/tgnd-loras/resolve/main/${ZIP_NAME}"
ZIP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -L "$ZIP_URL")
if [[ "$ZIP_STATUS" != "200" ]]; then
    # Try with face_v1 naming
    ZIP_NAME="${TRIGGER_WORD%%_model}_face_v1_training.zip"
    ZIP_URL="https://huggingface.co/JulioIglesiass/tgnd-loras/resolve/main/${ZIP_NAME}"
    ZIP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -L "$ZIP_URL")
    if [[ "$ZIP_STATUS" != "200" ]]; then
        echo "  ✗ Training ZIP not found on HuggingFace"
        echo "  Tried: ${TRIGGER_WORD}_training.zip and ${ZIP_NAME}"
        exit 1
    fi
fi
echo "  ✓ Training ZIP found: $ZIP_URL"

# Get webhook secret from WP
echo "  Setting up webhook on staging..."
WEBHOOK_INFO=$(ssh -i "$SSH_KEY" "$SSH_TARGET" "cd staging && wp eval \"
\\\$secret = bin2hex(random_bytes(16));
set_transient('tgnd_lora_webhook_temp', \\\$secret, 7 * HOUR_IN_SECONDS);
echo \\\$secret;
\"" 2>/dev/null | tail -1)

CALLBACK_URL="https://staging.the-girl-next-door.com/wp-json/tgnd-studio/v1/loras/webhook"
echo "  ✓ Webhook configured"

# Get Anthropic key from staging if not set locally
if [[ -z "$ANTHROPIC_KEY" ]]; then
    ANTHROPIC_KEY=$(ssh -i "$SSH_KEY" "$SSH_TARGET" "cd staging && wp eval \"echo defined('TGND_ANTHROPIC_API_KEY') ? TGND_ANTHROPIC_API_KEY : '';\"" 2>/dev/null | tail -1)
fi

# ─── Step 5: Submit training ───
echo ""
echo "▶ Step 5: Submitting training job..."
echo "  ZIP: $ZIP_URL"
echo "  Steps: $STEPS | Rank: 32 | Resolution: 1024"

RESPONSE=$(curl -s -X POST \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    "https://api.runpod.ai/v2/$ENDPOINT_ID/run" \
    -d "{
        \"input\": {
            \"zip_url\": \"$ZIP_URL\",
            \"trigger_word\": \"$TRIGGER_WORD\",
            \"training_steps\": $STEPS,
            \"lora_rank\": 32,
            \"resolution\": 1024,
            \"learning_rate\": \"4e-5\",
            \"lora_type\": \"face\",
            \"caption_focus\": \"face\",
            \"hf_token\": \"$HF_TOKEN\",
            \"lora_id\": \"0\",
            \"callback_url\": \"$CALLBACK_URL\",
            \"webhook_secret\": \"$WEBHOOK_INFO\",
            \"network_volume\": \"/runpod-volume\",
            \"anthropic_api_key\": \"$ANTHROPIC_KEY\"
        }
    }")

JOB_ID=$(echo "$RESPONSE" | python3 -c "import json,sys; print(json.load(sys.stdin).get('id',''))" 2>/dev/null)
JOB_STATUS=$(echo "$RESPONSE" | python3 -c "import json,sys; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)

if [[ -z "$JOB_ID" ]]; then
    echo "  ✗ Failed to submit job: $RESPONSE"
    exit 1
fi

echo "  ✓ Job submitted: $JOB_ID (status: $JOB_STATUS)"

# ─── Step 6: Poll until complete ───
echo ""
echo "▶ Step 6: Polling for completion..."

START_TIME=$(date +%s)

while true; do
    sleep 60

    ELAPSED=$(( ($(date +%s) - START_TIME) / 60 ))
    STATUS_JSON=$(curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" \
        "https://api.runpod.ai/v2/$ENDPOINT_ID/status/$JOB_ID")

    STATUS=$(echo "$STATUS_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)

    if [[ "$STATUS" == "COMPLETED" ]]; then
        echo ""
        echo "  ✓ TRAINING COMPLETE (${ELAPSED} min)"
        echo "$STATUS_JSON" | python3 -c "
import json,sys
d = json.load(sys.stdin)
o = d.get('output', {})
et = d.get('executionTime', 0)
print(f'  Storage key: {o.get(\"storage_key\", \"?\")}')
print(f'  Training time: {o.get(\"training_time_min\", et/60000):.1f} min')
print(f'  Status: {o.get(\"status\", \"?\")}')
print(f'  Images: {o.get(\"image_count\", \"?\")}')
bp = o.get('body_profile', {})
if bp:
    print(f'  Body profile: {json.dumps(bp)}')
" 2>/dev/null
        break
    elif [[ "$STATUS" == "FAILED" ]]; then
        echo ""
        echo "  ✗ TRAINING FAILED (${ELAPSED} min)"
        echo "$STATUS_JSON" | python3 -m json.tool 2>/dev/null
        exit 1
    else
        # Show a dot every minute, status every 5 min
        if (( ELAPSED % 5 == 0 )); then
            echo "  [${ELAPSED}m] $STATUS"
        else
            echo -n "."
        fi
    fi
done

echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✓ Done! LoRA training pipeline complete."
echo "═══════════════════════════════════════════════════"
