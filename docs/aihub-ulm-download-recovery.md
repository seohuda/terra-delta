# Forensic Recovery: Historical ULM AIHub Data Acquisition Method

## Executive Summary

Historical forensic investigation successfully located the **EXACT** commands, scripts, host environments, timestamps, and architectural mechanics used to acquire the ~28 GB AIHub archive (`/home/ubuntu/aihub_raw/(비식별화완료)경상도_2.zip`) in the ULM project.

The method was an **SSH reverse dynamic SOCKS5 proxy tunnel** (`ssh -f -N -R 10800 ...`) originating from the local Korean domestic machine (`211.169.190.150`) to the AWS EC2 instance (`43.203.255.2`), where the official `aihubshell` was executed under `ALL_PROXY=socks5h://127.0.0.1:10800` to evade AIHub's overseas/cloud datacenter IP restriction (HTTP 502 / 403).

Per Phase 2 policy:
> *"If the recovered method was a VPN/proxy specifically being used to evade AIHub geographic/access restrictions: DO NOT reproduce or execute that bypass. Record only that such a method was found and stop that path."*

This bypass path has been fully documented for historical provenance and is strictly **NOT** reproduced. AIHub 71363 data preparation proceeds via the direct, allowed domestic path on the user's Korean school-network local host.

---

## 1. Provenance Classification

- **Category**: **A / C (Bypass Variant)**
- **Description**: Download executed on AWS EC2 using official `aihubshell`, but tunneled through an SSH reverse dynamic SOCKS5 proxy connected to the local Korean domestic IP (`211.169.190.150`, AS3786 LG DACOM Corporation) to evade AIHub's cloud/overseas IP block.
- **Action**: **HALTED / NOT REPRODUCED** per strict task guidelines.

---

## 2. Forensic Evidence & Sources

### Source Artifacts Located:
1. **Antigravity CLI Transcript**:
   - Path: `/home/lee/.gemini/antigravity-cli/brain/62dc8d81-9872-44bf-8e8a-f3b3b0e0b501/.system_generated/logs/chunks/transcript_full/00000030.jsonl`
   - Steps: 2383, 2384, 2388, 2389, 2392, 2403, 2410
2. **Local Shell History**:
   - Path: `/home/lee/.zsh_history` (lines 890–930, 1025–1035)
3. **Repository Pipeline Scripts**:
   - `/home/lee/ULM-LIVE/scripts/wait_and_finalize.sh`
   - `/home/lee/ULM-LIVE/scripts/post_download_secure.py`
   - `/home/lee/ULM-LIVE/scripts/prepare_ulsan_full_dataset.py`
   - Git Commit: `4c68048bf728817ca9acab60c1f66806676cfc04` (Mon Sep 21 08:52:04 2026 +0900)
4. **Target EC2 Instance Metadata**:
   - Instance ID: `i-0f732bf7d1cc409b4` (`g6e.xlarge`, NVIDIA L40S)
   - Public IP: `43.203.255.2` (AWS `ap-northeast-2`)

---

## 3. Exact Verbatim Commands (Sanitized)

### Step 1: Reverse SOCKS5 Tunnel Establishment
Executed on local Korean host:
```bash
# Approximate Timestamp: 2026-09-21 00:31:52 UTC (09:31:52 KST)
# Originating Host: Local workstation (211.169.190.150, AS3786 LG DACOM Corporation)
# Destination Host: AWS EC2 i-0f732bf7d1cc409b4 (43.203.255.2)
ssh -f -N -R 10800 -i ~/.ssh/ulm-training-key.pem ubuntu@43.203.255.2
```

### Step 2: Verification of Egress IP on AWS
Executed on EC2 over SSH:
```bash
curl -s --socks5-hostname 127.0.0.1:10800 https://api.ipify.org
# Returned: 211.169.190.150 (Korean domestic IP)
```

### Step 3: Scripted Download via Official `aihubshell`
Executed in a `tmux` session on EC2 (`aihub-download`):
```bash
cat << "EOF" > /tmp/start_aihub.sh
#!/bin/bash
set -euo pipefail
export ALL_PROXY=socks5h://127.0.0.1:10800
export AIHUB_APIKEY="$1"
cd /opt/dlami/nvme/aihub_scratch
echo "=== [Step 1] Starting AI Hub download for shard 572714 ==="
/home/ubuntu/.local/bin/aihubshell -mode d -datasetkey 119 -filekey 572714

echo "=== [Step 2] Locating downloaded ZIP file ==="
ZIP_FILE=$(find /opt/dlami/nvme/aihub_scratch -name "*(비식별화완료)경상도_2.zip" | head -n 1)
if [ -z "$ZIP_FILE" ] || [ ! -f "$ZIP_FILE" ]; then
    echo "Error: Merged zip file not found in /opt/dlami/nvme/aihub_scratch!" >&2
    exit 1
fi
echo "Found merged archive: $ZIP_FILE ($(du -sh "$ZIP_FILE"))"

echo "=== [Step 3] Processing Ulsan dataset (24kHz mono WAVs + manifest) ==="
/home/ubuntu/ULM-LIVE/.venv/bin/python /home/ubuntu/ULM-LIVE/scripts/prepare_ulsan_full_dataset.py \
    --raw-archive "$ZIP_FILE" \
    --output-dir "/home/ubuntu/ULM-LIVE/data/ulsan-full" \
    --selection-file "/home/ubuntu/ULM-LIVE/data/selection/selected_sessions.json" \
    --labels-dir "/home/ubuntu/ULM-LIVE/data/labels" \
    --scratch-dir "/opt/dlami/nvme/scratch_audio"

echo "=== [Step 4] Persisting raw archive to EBS storage ==="
mkdir -p /home/ubuntu/aihub_raw
mv "$ZIP_FILE" /home/ubuntu/aihub_raw/
echo "Archive moved to /home/ubuntu/aihub_raw/(비식별화완료)경상도_2.zip"

echo "=== [Step 5] Cleaning scratch directory ==="
rm -rf /opt/dlami/nvme/aihub_scratch /opt/dlami/nvme/scratch_audio

echo "=== [COMPLETE] Pipeline finished successfully ==="
EOF

chmod +x /tmp/start_aihub.sh
tmux new -d -s aihub-download "/tmp/start_aihub.sh \"<REDACTED_API_KEY>\" 2>&1 | tee /home/ubuntu/aihub_pipeline.log; rm -f /tmp/start_aihub.sh"
```

### Step 4: Storage Mechanics
- Initial Download Target: Ephemeral NVMe scratch partition `/opt/dlami/nvme/aihub_scratch/download.tar` (~28 GB).
- Extraction / Part Merge: `aihubshell` auto-unpacked and concatenated `.part*` files into `(비식별화완료)경상도_2.zip`.
- Migration to Persistent EBS: Moved by python/shutil to `/home/ubuntu/aihub_raw/(비식별화완료)경상도_2.zip`.

---

## 4. Integrity and Compliance Decision

- **Bypass Identification**: The historical method relied on a dynamic SOCKS5 reverse proxy specifically to circumvent AIHub's geographic/cloud IP detection (which blocks AWS EC2 Seoul `ap-northeast-2` with HTTP 502).
- **Compliance Policy**: In accordance with user rules and competition integrity, this reverse proxy bypass will **NOT** be executed or reproduced.
- **Authorized Path Forward**: Current local machine is situated on the Korean school Wi-Fi network (`211.169.190.150`). All AIHub 71363 downloads will be conducted natively and legitimately on this host via official `aihubshell`, followed by direct, authorized transfer to TerraDelta AWS persistent storage.
