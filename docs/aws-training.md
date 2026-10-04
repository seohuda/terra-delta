# Future AWS workflow with hotspot protection

Current dataset verdict is **NOT READY**: four mock pairs, zero real samples.
No command creating/starting EC2, transferring S3 data or training was run.
Future AWS operations need separate authorization. This document records those
future commands; it does not authorize or execute them.

## Data path and where commands run

The PC is an SSH/Git console. Data must flow **provider → EC2 → EBS/S3**.
Do not download rasters/dataset ZIPs to the PC, SCP multi-GB data/checkpoints,
pull local Docker images or rebuild local PyTorch/CUDA. EC2 performs git clone,
pip installation, source download, preprocessing, training and checkpoint creation.
Use the existing local environment for inspection only. Retrieve only metrics,
small logs/JPEGs and the final submission ZIP when needed.

A previously selected g5.2xlarge/A10G is the preparation plan, not a running
resource. Pick/review instance, AMI/driver, disk, subnet/security group and
least-privilege instance role before any paid operation. Existing ULM instances
are not implied targets. Review current availability/pricing separately; this
workflow creates no instance, bucket, spot request or account defaults.

## Next step: SSH and remote-only preparation

Set your own region/profile/instance/SSH target on the PC. No account, bucket,
region or instance ID is hardcoded. Do not start anything in the current audit.
For a **later authorized start** of the selected existing instance:

```sh
# PC: set real values first; these remain placeholders here.
export AWS_PROFILE=YOUR_PROFILE
export AWS_REGION=YOUR_REGION
export TERRADELTA_INSTANCE_ID=YOUR_PROJECT_INSTANCE_ID
export TERRADELTA_SSH_TARGET=YOUR_SSH_USER_AT_HOST
export TERRADELTA_S3_BUCKET=YOUR_EXISTING_REVIEWED_BUCKET
# Only later, if explicitly authorized to start this instance:
aws ec2 start-instances --instance-ids "$TERRADELTA_INSTANCE_ID"
aws ec2 wait instance-running --instance-ids "$TERRADELTA_INSTANCE_ID"
aws ec2 wait instance-status-ok --instance-ids "$TERRADELTA_INSTANCE_ID"
ssh "$TERRADELTA_SSH_TARGET"
```

The next commands run **inside EC2**, never through local dataset download:

```sh
# EC2: clone or update code; use a mounted writable EBS /data volume.
git clone https://github.com/seohuda/terra-delta.git
cd terra-delta
# Existing checkout instead: git pull --ff-only
export TERRADELTA_DATA_ROOT=/data/terradelta
export TERRADELTA_S3_BUCKET=YOUR_EXISTING_REVIEWED_BUCKET
export AWS_REGION=YOUR_REGION
bash scripts/aws_setup.sh                    # plan only
bash scripts/aws_setup.sh --execute          # dependency setup on EC2; no training
source .venv/bin/activate
mkdir -p "$TERRADELTA_DATA_ROOT"/{datasets,manifests,checkpoints,outputs,submissions}

# Bounded live discovery/HEAD on EC2, still no raster/footprint download.
python scripts/download_naip.py --bounds -76.61 39.29 -76.60 39.30 \
  --years 2021 --region md-pilot --state md --max-items 2 --inspect-sizes \
  --dry-run --output "$TERRADELTA_DATA_ROOT/datasets/naip"
python scripts/download_fema.py --bounds -76.61 39.29 -76.60 39.30 \
  --dry-run --output "$TERRADELTA_DATA_ROOT/datasets/fema/structures.geojson"
python scripts/download_hansen.py --bounds -76.61 39.29 -76.60 39.30 \
  --layers lossyear --max-tiles 1 --inspect-sizes --dry-run \
  --output "$TERRADELTA_DATA_ROOT/datasets/hansen"
```

Example AOI is a planning probe, not an approved training region. Current NAIP
downloader resolves Planetary Computer COGs on Azure; do not describe that backend
as S3. If an equivalent, verified AWS Open Data asset is selected, access its S3
object directly **from EC2**, using its documented region/access mode (for a
public asset, `aws s3 cp "$TERRADELTA_NAIP_OPEN_DATA_URI" /data/naip/ --no-sign-request`).
Verify matching acquisition date/bands/resolution/license before substituting.
FEMA and Hansen also download directly to EC2 after rights/size review. Never
stage them on the PC. Current download guards require explicit `--download`,
full discovery and byte limits; no large-download command is run in this phase.

Original official weights also go directly to EC2 using baseline/README.md or a
previously authorized S3 copy, not PC-mediated checkpoint uploads. Check original
SHA-256 on EC2. Dataset/code files do not contain credentials or signed URLs.

## Offline local downloader checks

These commands replay existing metadata and perform no new HTTP request:

```sh
.venv/bin/python scripts/download_naip.py --bounds -76.61 39.29 -76.60 39.30 \
  --years 2021 --metadata-plan outputs/naip-dry-run.json --dry-run --output /data/naip
.venv/bin/python scripts/download_fema.py --bounds -76.61 39.29 -76.60 39.30 \
  --metadata-plan outputs/fema-dry-run.json --dry-run --output /data/fema/structures.geojson
.venv/bin/python scripts/download_hansen.py --bounds -76.61 39.29 -76.60 39.30 \
  --dry-run --output /data/hansen
```

Historical plans do not authorize acquisition. Unknown wire sizes remain unknown;
uncompressed/AOI estimates are not whole-source transfer sizes. All downloader
paths accept CLI arguments and resolve relative to the invoking EC2 workspace.

## EBS and existing S3 layout

Use an existing approved bucket; no bucket is created. The AWS CLI owns these
transfers; no additional storage service or embedded credentials are required.

```text
s3://<TERRADELTA_S3_BUCKET>/terradelta/
  datasets/
  manifests/
  checkpoints/
  outputs/
  submissions/
```

Later backups run **EC2 → S3** using the instance role:

```sh
# EC2 only, after separate transfer authorization:
: "${TERRADELTA_S3_BUCKET:?Set an existing reviewed bucket}"
aws s3 sync "$TERRADELTA_DATA_ROOT/datasets/" "s3://$TERRADELTA_S3_BUCKET/terradelta/datasets/"
aws s3 sync "$TERRADELTA_DATA_ROOT/manifests/" "s3://$TERRADELTA_S3_BUCKET/terradelta/manifests/"
aws s3 sync outputs/train_first250/ "s3://$TERRADELTA_S3_BUCKET/terradelta/checkpoints/" \
  --exclude '*' --include '*.pt'
aws s3 sync outputs/train_first250/ "s3://$TERRADELTA_S3_BUCKET/terradelta/outputs/" \
  --exclude '*.pt'
aws s3 cp outputs/submission.zip "s3://$TERRADELTA_S3_BUCKET/terradelta/submissions/submission.zip"
```

Optional setup input sync uses explicit `TERRADELTA_DATA_S3_URI` and
`TERRADELTA_CHECKPOINT_S3_URI` on EC2 only. Keep source dataset provenance and
source-specific redistribution restrictions when transferring/retaining inputs.
Encrypted EBS persists independently; instance scratch storage is not a backup.

## Before eventual training

Prepare/review actual labels and at least two disconnected geographies on EC2.
Save paths relative to manifests, source rasters/AOI/bounds/CRS, real acquisition
years/resolution, all input sources, license evidence and reviewer decisions.
Audit must pass; present NOT READY fixtures must not be repurposed as real labels.
Run unchanged official weights on fully reviewed real validation before fitting.

```sh
# EC2: reviewed local files only; exact output path must be new.
python scripts/audit_data.py \
  --manifest "$TERRADELTA_DATA_ROOT/manifests/reviewed.csv" \
  --train-manifest "$TERRADELTA_DATA_ROOT/manifests/train.csv" \
  --val-manifest "$TERRADELTA_DATA_ROOT/manifests/val.csv" \
  --checkpoint "$TERRADELTA_DATA_ROOT/checkpoints/official.pt" \
  --output "$TERRADELTA_DATA_ROOT/outputs/pretraining_audit"
# Copy/configure real manifest + official checkpoint paths in an experiment YAML.
python scripts/train.py --config configs/train_first250.yaml --dry-run
# Only after later explicit training authorization and satisfactory real audit:
python scripts/train.py --config configs/train_first250.yaml
```

The checked-in training config still uses repository-relative placeholder data
and baseline paths. Set `data.train_manifest`, `data.val_manifest`,
`training.init_checkpoint` and output path on EC2 before the commands above.
The first planned checkpoint steps are 25/50/75/100/150/250, encoder LR 1e-5,
head LR 1e-4. Compare approximate score, both class scores, FP/FN, negative FP
rate and confidence distribution. Inspect 25–100 first; 500+ is not a priority.

Only small results return to the hotspot PC, for example an explicit single-file
`scp "$TERRADELTA_SSH_TARGET:/data/terradelta/outputs/pretraining_audit/report.json" ./`
and the selected JPEG preview. Do not copy `/data/`, model directories or entire
probability/checkpoint trees. The existing final ZIP is about 53 MB; retrieve it
only when needed, not as part of routine monitoring.

After future work, verify EC2→S3 backups before a separately authorized stop:
`aws ec2 stop-instances --instance-ids "$TERRADELTA_INSTANCE_ID"`, then
`aws ec2 wait instance-stopped --instance-ids "$TERRADELTA_INSTANCE_ID"`.
Persistent EBS/S3 can incur charges while EC2 is stopped.

Official command references:
[start-instances](https://docs.aws.amazon.com/cli/latest/reference/ec2/start-instances.html),
[stop-instances](https://docs.aws.amazon.com/cli/latest/reference/ec2/stop-instances.html).
