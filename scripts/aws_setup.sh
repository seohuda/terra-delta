#!/usr/bin/env bash
# Run on an already provisioned EC2 host. No provisioning or training calls.
set -euo pipefail
EXECUTE=false
MODE=cpu
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=true ;;
    --cpu) MODE=cpu ;;
    --gpu) MODE=gpu ;;
    *) printf 'Unknown argument: %s\n' "$arg" >&2; exit 2 ;;
  esac
done
if [[ "$EXECUTE" != true ]]; then
  cat <<'PLAN'
Preparation script: no actions without --execute.
Default: CPU dataset builder on an existing Ubuntu host (t3.medium, 30 GB gp3).
1. Clone terra-delta and create Python venv.
2. Install CPU-only torch/torchvision, project and geospatial dependencies.
3. Optionally sync approved datasets/checkpoints from S3 using instance role.
4. Build/review/audit real datasets; no training command is run or suggested.
5. Preserve data and stop the instance after the task.
Future GPU setup requires explicit --gpu on an already authorized GPU host.
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
if [[ "$MODE" == cpu ]]; then
  python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
fi
python -m pip install --no-cache-dir -r requirements.txt
python - "$MODE" <<'PY'
import sys
import torch
import rasterio
import scipy
import shapely
print('torch', torch.__version__, 'CUDA build', torch.version.cuda)
if sys.argv[1] == 'cpu' and torch.version.cuda is not None:
    raise SystemExit('CPU setup resolved a CUDA wheel; do not proceed with this environment.')
if sys.argv[1] == 'gpu' and not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable. Check driver/image/wheel before future GPU training.')
if sys.argv[1] == 'gpu':
    print(torch.cuda.get_device_name(0))
print('geospatial dependencies:', rasterio.__version__, scipy.__version__, shapely.__version__)
PY
if [[ -n ${TERRADELTA_DATA_S3_URI:-} ]]; then
  aws s3 sync "$TERRADELTA_DATA_S3_URI" data/approved/
fi
if [[ -n ${TERRADELTA_CHECKPOINT_S3_URI:-} ]]; then
  aws s3 sync "$TERRADELTA_CHECKPOINT_S3_URI" checkpoints/
fi
cat <<'NEXT'
Setup finished; no training was started.
Review source licenses, build real temporal pairs and run the data audit.
Preserve the dataset on EBS or an explicitly approved backup target.
Stop your instance after the dataset task; do not terminate unbacked data.
NEXT
