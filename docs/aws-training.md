# Future AWS training (not executed in preparation)

Recommended starting point: EC2 g5.2xlarge with one NVIDIA A10G, 8 vCPU and
32 GiB host memory. Review current regional availability and pricing before
provisioning. See [AWS G5 specification](https://aws.amazon.com/ec2/instance-types/g5/).
Use an Ubuntu GPU image with a supported NVIDIA driver and matching PyTorch
CUDA wheel. Use a dedicated new project instance or an explicitly selected
existing instance; no recorded ULM instance is reused implicitly.

No EC2 create/start/stop calls, S3 transfer, GPU training, or AWS credentials
were used while building this repository. `scripts/aws_setup.sh` prints a plan
by default. It runs dependency setup only with `--execute`, on a host you have
already provisioned. It never starts training or modifies shell startup files.

## Commands for a later authorized run

Choose your AWS profile, region, existing project instance id and SSH settings.
These placeholders are deliberately not executable account defaults:

```sh
export AWS_PROFILE=YOUR_PROFILE
export AWS_REGION=YOUR_REGION
export TERRADELTA_INSTANCE_ID=YOUR_PROJECT_INSTANCE_ID
aws ec2 start-instances --instance-ids "$TERRADELTA_INSTANCE_ID"
aws ec2 wait instance-running --instance-ids "$TERRADELTA_INSTANCE_ID"
aws ec2 wait instance-status-ok --instance-ids "$TERRADELTA_INSTANCE_ID"
# SSH to your host, then:
git clone https://github.com/seohuda/terra-delta.git
cd terra-delta
bash scripts/aws_setup.sh                 # plan only
TERRADELTA_REPO_DIR="$PWD" bash scripts/aws_setup.sh --execute
source .venv/bin/activate
```

If creating a new instance, select reviewed AMI, subnet, security group, SSH
key, least-privilege instance role and EBS storage in EC2 first. The repo contains
no provisioning command because these cost/security choices depend on your
account. Use encrypted EBS for persistent assets; ephemeral instance storage is
not a checkpoint backup. No credentials or account IDs belong in code/configs.

Optional S3 input sync is available through `TERRADELTA_DATA_S3_URI` and
`TERRADELTA_CHECKPOINT_S3_URI` during setup; it is absent by default. Use an
instance role instead of hardcoded keys. Download and review licensed data
separately, create approved labeled manifests and geographic splits, then
inspect `configs/train_short.yaml` and paths before the first run.

```sh
python scripts/train.py --config configs/train_short.yaml --dry-run
# Only after reviewing the plan/data/license status:
python scripts/train.py --config configs/train_short.yaml
# Resume and additional evaluation commands are documented in experiments.md.
aws s3 sync checkpoints/ s3://YOUR_BUCKET/terradelta/checkpoints/
aws s3 sync outputs/ s3://YOUR_BUCKET/terradelta/outputs/
# Back up completed and latest checkpoints; confirm the backup before stopping:
aws ec2 stop-instances --instance-ids "$TERRADELTA_INSTANCE_ID"
aws ec2 wait instance-stopped --instance-ids "$TERRADELTA_INSTANCE_ID"
```

Official command references: [start-instances](https://docs.aws.amazon.com/cli/latest/reference/ec2/start-instances.html),
[stop-instances](https://docs.aws.amazon.com/cli/latest/reference/ec2/stop-instances.html).
Check CUDA availability, free disk and VRAM, CPU loader throughput, and correct
encoder/head LRs before paying for a longer run. A checkpoint every 25–100
optimizer steps provides early exit options. Stop only after backups finish;
verify EC2 is stopped, and budget persistent EBS/S3 charges separately.
