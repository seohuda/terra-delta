#!/usr/bin/env bash
# Run only later, on an already provisioned EC2 host. No provisioning/start/stop calls.
set -euo pipefail
if [[ ${1:-} != --execute ]]; then
  cat <<'PLAN'
Preparation script: no actions without --execute.
Recommended future host: EC2 g5.2xlarge (A10G), Ubuntu/NVIDIA GPU image.
1. Clone terra-delta and create Python venv.
2. Install reviewed dependencies and verify torch CUDA availability.
3. Optionally sync approved datasets/checkpoints from S3 using instance role.
4. Run training manually after license and geographic-split checks.
5. Back up checkpoints to a user-specified S3 URI; stop instance manually.
PLAN
  exit 0
fi
REPO_URL=${TERRADELTA_REPO_URL:-https://github.com/seohuda/terra-delta.git}
SCRIPT_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR=${TERRADELTA_REPO_DIR:-"$SCRIPT_REPO_DIR"}
if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
# Install a CUDA-enabled torch wheel appropriate to your selected NVIDIA image first.
python -m pip install -r requirements-dev.txt
python - <<'PY'
import torch
print('torch', torch.__version__, 'CUDA build', torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable. Check driver/image/wheel before future GPU training.')
print(torch.cuda.get_device_name(0))
PY
if [[ -n ${TERRADELTA_DATA_S3_URI:-} ]]; then
  aws s3 sync "$TERRADELTA_DATA_S3_URI" data/approved/
fi
if [[ -n ${TERRADELTA_CHECKPOINT_S3_URI:-} ]]; then
  aws s3 sync "$TERRADELTA_CHECKPOINT_S3_URI" checkpoints/
fi
cat <<'NEXT'
Setup finished; no training was started.
Review config/data licenses, then manually run:
  python scripts/train.py --config configs/train_short.yaml --dry-run
  python scripts/train.py --config configs/train_short.yaml
Checkpoint backup example (replace YOUR_BUCKET):
  aws s3 sync checkpoints/ s3://YOUR_BUCKET/terradelta/checkpoints/
After backup, stop your instance manually in the EC2 console or AWS CLI.
NEXT
