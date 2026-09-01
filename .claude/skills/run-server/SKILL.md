---
name: run-server
description: This skill should be used when the user asks to "run on server", "execute on GPU", "train remotely", "run experiment on lab machine", "check on my training", "check running job", mentions "128.2.204.110", "128.2.204.116", "spiderman", "superman", or discusses "Isaac Sim execution". Provides guidance for executing experiments on lab GPU servers via SSH and Docker.
---

# Remote Server Execution

Execute experiments on the lab GPU server with proper setup, job tracking, and results retrieval.

---

## Server Configuration

| Server | IP | SSH User | Purpose |
|--------|-----|----------|---------|
| Spiderman | 128.2.204.110 | rnagabhi | Isaac Sim, training, experiments |
| Superman | 128.2.204.116 | rohan | Lego training, overflow experiments |

Spiderman has 4x RTX A6000 (49GB each).

**Post-rebuild (March 2026):** Spiderman was rebuilt. Username changed from `rohan` to `rnagabhi` (Andrew ID). Home is `/home/rnagabhi` (3.4TB volume, 3.0TB free). Old data at `/sata1/data/rohan/` needs permission fix from Peiqi Yu. Docker may need to be installed/configured — check with `docker info` before using Docker commands.

**GPU selection rule:** Always select the **highest numbered free GPUs** first (e.g., if GPUs 1-3 are free, use 3, then 2, then 1). Free = 0% utilization and ~5MiB memory. Always run `nvidia-smi` before launching to verify.

### Server Selection

- **Default**: Spiderman (128.2.204.110)
- **Use Superman** if: plan/config specifies Superman, user says "superman", or user mentions "128.2.204.116"
- Use `$PERSEVE_SERVER` variable if set, otherwise default to Spiderman
- Superman has 8x RTX A4000 (16GB each) — **leave 2 GPUs free for other users**

---

## Prerequisites: SSH Authentication

All SSH/SCP commands use `sshpass` with the `$PERSEVE_SERVER_PASSWORD` environment variable.

**Setup (one-time):**
```bash
# Install sshpass
brew install hudochenkov/sshpass/sshpass

# Add to ~/.zshrc (already configured)
export PERSEVE_SERVER_PASSWORD="your_password"
```

**Command pattern:**
```bash
# SSH (use appropriate server IP)
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "<command>"   # Spiderman
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rohan@128.2.204.116 "<command>"   # Superman

# SCP
sshpass -p "$PERSEVE_SERVER_PASSWORD" scp <source> rnagabhi@128.2.204.110:<dest>   # Spiderman
sshpass -p "$PERSEVE_SERVER_PASSWORD" scp <source> rohan@128.2.204.116:<dest>   # Superman
```

**Important:** The password is stored in `~/.zshrc` (not committed to git). Never put the password in skill files or CLAUDE.md.

---

## Phase 1: Determine Workflow Mode

Ask the user what they want to do:

---

**What would you like to do?**

1. **START** a new long-running job (training, SDG, multi-hour tasks)
2. **CHECK** on a running job (see status, output, progress)
3. **RETRIEVE** results from a completed job
4. **CLEANUP** server after retrieval
5. **QUICK** command (< 5 min, wait for result)

---

Based on selection, go to the appropriate section:
- Option 1 → Phase 2 (Start Job)
- Option 2 → Phase 6 (Check Job)
- Option 3 → Phase 7 (Retrieve Results)
- Option 4 → Phase 8 (Cleanup)
- Option 5 → Phase 5 (Quick Command)

---

## Phase 2: Pre-Flight Safety Checks

**CRITICAL: These checks MUST pass before starting any long-running job.**

### Step 1: Verify Connection
```bash
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh -o ConnectTimeout=5 rnagabhi@128.2.204.110 "echo 'Connected' && hostname"
```

### Step 2: Run Status Checks
```bash
# Replace SERVER_IP with 128.2.204.110 (Spiderman) or 128.2.204.116 (Superman)
SERVER_IP="128.2.204.110"

# Check disk space
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rohan@$SERVER_IP "df -h /home/rohan"

# Check GPU availability
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rohan@$SERVER_IP "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv"

# Check existing tmux sessions
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rohan@$SERVER_IP "tmux list-sessions 2>/dev/null || echo 'No active sessions'"

# Check running containers (Spiderman only)
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rohan@$SERVER_IP "docker ps"
```

### Step 3: Evaluate Thresholds

**Spiderman (128.2.204.110) thresholds:**

| Resource | Threshold | Action |
|----------|-----------|--------|
| Disk space | < 10GB free | BLOCK - Refuse to run, ask user to clean up |
| Disk space | 10-20GB free | WARN - Proceed but alert user |
| GPU memory | > 90% used | BLOCK - Something else is running |
| GPU memory | > 50% used | WARN - Ask user if they want to proceed |

**Superman (128.2.204.116) thresholds:**

| Resource | Threshold | Action |
|----------|-----------|--------|
| Disk space | < 250GB free | BLOCK - Must preserve 250GB minimum |
| Disk usage | > 100GB ours | WARN - Ask user to confirm |
| GPUs in use | > 6 of 8 | BLOCK - Leave 2 GPUs free for other users |
| GPU memory | > 90% used on target GPU | BLOCK - Pick a different GPU |

### Step 4: Block or Proceed

**If blocked**, present options:
```
BLOCKED: [reason]

Options:
1. Run cleanup (Phase 8)
2. Check what's running (Phase 6)
3. Abort
```

**If clear**, proceed to Phase 3.

---

## Phase 3: Start Long-Running Job

All long-running jobs use **tmux** for persistence and easy monitoring.

### Step 1: Choose Job Name
Generate a descriptive name: `<type>-<date>-<id>`
- Example: `training-20260128-v1`
- Example: `sdg-20260128-batch1`

### Step 2: Start Job in tmux

**For Training:**
```bash
JOB_NAME="training-$(date +%Y%m%d)-v1"
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "tmux new-session -d -s $JOB_NAME 'cd /home/rohan/perseve && source ~/.bashrc && conda activate PERSEVE && python src/perseve/fine_tuning/train.py --config config.yaml 2>&1 | tee logs/${JOB_NAME}.log'"
```

**For Isaac Sim SDG:**
```bash
JOB_NAME="sdg-$(date +%Y%m%d)-batch1"
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "tmux new-session -d -s $JOB_NAME 'cd /home/rohan/perseve/src/perseve/synthetic_data_generation && docker exec -i isaac-sim bash -c \"cd /workspace/perseve/src/perseve/synthetic_data_generation && bash run_scene_based_sdg.sh --start-idx 0 --end-idx 100 --gpu-id 0\" 2>&1 | tee /home/rohan/logs/${JOB_NAME}.log'"
```

**For Custom Command:**
```bash
JOB_NAME="job-$(date +%Y%m%d)-custom"
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "tmux new-session -d -s $JOB_NAME '<command> 2>&1 | tee /home/rohan/logs/${JOB_NAME}.log'"
```

### Step 3: Save Job Metadata
```bash
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "echo '{\"name\":\"$JOB_NAME\",\"started\":\"$(date -Iseconds)\",\"type\":\"<type>\",\"output_path\":\"<expected_output>\"}' >> /home/rohan/.claude-jobs.jsonl"
```

### Step 4: Verify Job Started
```bash
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "tmux has-session -t $JOB_NAME 2>/dev/null && echo 'Job started successfully' || echo 'ERROR: Job failed to start'"
```

### Step 5: Report to User
```
Job started successfully!

Session name: <JOB_NAME>
Log file: /home/rohan/logs/<JOB_NAME>.log
Expected output: <path>

To check on this job later:
- Say "check on my training" or "check job <JOB_NAME>"
- Or run: /run-server and select "CHECK on a running job"

You can now disconnect. The job will continue running on the server.
```

---

## Phase 4: Task-Specific Setup

### For Isaac Sim Tasks

**Ensure Docker container is running:**
```bash
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "docker ps | grep isaac-sim || echo 'Container not running'"
```

**Start container if needed:**
```bash
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "cd /home/rohan/perseve/src/perseve/synthetic_data_generation && bash docker/isaac_sim_docker.sh"
```

**Execute in container (for quick commands):**
```bash
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "docker exec -i isaac-sim bash -c 'cd /workspace/perseve/src/perseve/synthetic_data_generation && <command>'"
```

Note: Use `-i` not `-it` when running via SSH with command string.

### For Training Tasks

**Activate environment:**
```bash
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "source ~/.bashrc && conda activate PERSEVE && <command>"
```

---

## Phase 5: Quick Command (< 5 min)

For short commands that complete quickly, run synchronously:

```bash
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "<command>"
```

Wait for result and report back to user.

---

## Phase 6: Check on Running Job

### Step 1: List Active Jobs
```bash
# List tmux sessions
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "tmux list-sessions 2>/dev/null || echo 'No active sessions'"

# Show saved job metadata
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "cat /home/rohan/.claude-jobs.jsonl 2>/dev/null | tail -10"
```

### Step 2: Check Specific Job Status
```bash
JOB_NAME="<job-name>"

# Check if still running
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "tmux has-session -t $JOB_NAME 2>/dev/null && echo 'RUNNING' || echo 'COMPLETED/STOPPED'"

# Get recent output (last 100 lines)
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "tmux capture-pane -t $JOB_NAME -p -S -100 2>/dev/null || echo 'Session not found'"

# Check log file
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "tail -50 /home/rohan/logs/$JOB_NAME.log 2>/dev/null || echo 'Log not found'"
```

### Step 3: Check Resource Usage
```bash
# GPU status
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "nvidia-smi"

# Disk space
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "df -h /home/rohan"
```

### Step 4: Report Status
```
Job Status: <JOB_NAME>

Status: RUNNING / COMPLETED
Runtime: X hours Y minutes
GPU Usage: X%
Disk Free: X GB

Recent output:
<last 20 lines>

Options:
1. Keep monitoring (check again later)
2. Retrieve results (if complete)
3. Kill job (if stuck)
```

### Step 5: Kill Job (if requested)
```bash
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "tmux kill-session -t $JOB_NAME"
```

---

## Phase 7: Retrieve Results

### Step 1: Verify Job Completed
```bash
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "tmux has-session -t $JOB_NAME 2>/dev/null && echo 'Still running!' || echo 'Completed'"
```

### Step 2: Check Output Exists
```bash
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "ls -la <expected_output_path>"
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "du -sh <expected_output_path>"
```

### Step 3: Copy Results Locally
```bash
# Ask user for local destination
LOCAL_PATH=~/Downloads/<job-name>

# Copy results
sshpass -p "$PERSEVE_SERVER_PASSWORD" scp -r rnagabhi@128.2.204.110:<remote_path> $LOCAL_PATH

# Copy log file
sshpass -p "$PERSEVE_SERVER_PASSWORD" scp rnagabhi@128.2.204.110:/home/rohan/logs/$JOB_NAME.log $LOCAL_PATH/
```

### Step 4: Verify Transfer
```bash
# Check local size matches remote
LOCAL_SIZE=$(du -sh $LOCAL_PATH | awk '{print $1}')
REMOTE_SIZE=$(sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "du -sh <remote_path> | awk '{print \$1}'")
echo "Local: $LOCAL_SIZE, Remote: $REMOTE_SIZE"
```

### Step 5: Report
```
Results retrieved successfully!

Local path: <LOCAL_PATH>
Size: <SIZE>
Files: <count>

Ready for cleanup? (Phase 8)
```

---

## Phase 8: Smart Cleanup

### Safety Gate

**Before ANY cleanup, verify results were copied:**

```
Results verification:
- Local path: <path>
- Local size: X GB
- Remote size: X GB
- Match: Yes/No

Can you confirm the results were copied successfully?
```

**DO NOT proceed until user confirms.**

### Step 1: Analyze Server Data
```bash
# List data directories with sizes
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "du -sh /home/rohan/perseve/data/* 2>/dev/null | sort -h"

# Find large files
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "find /home/rohan/perseve -type f -size +100M -exec ls -lh {} \; 2>/dev/null | head -20"

# Check old tmux sessions
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "tmux list-sessions 2>/dev/null"
```

### Step 2: Categorize into Buckets

| Bucket | Description | Examples |
|--------|-------------|----------|
| SAFE | Temp files, caches | `__pycache__`, `.tmp`, old logs |
| PROBABLY | Can be regenerated | Extracted archives, intermediate outputs |
| MAYBE | Expensive to recreate | Downloaded archives (`.7z`, `.zip`) |
| KEEP | Primary outputs | Final CSVs, trained models, configs |

**Classification patterns:**
- SAFE: `__pycache__`, `*.tmp`, `*.pyc`, `.DS_Store`, `*.bak`, old tmux sessions
- PROBABLY: extracted STL directories (if metrics already extracted)
- MAYBE: `*.7z`, `*.zip`, `downloads/` (expensive to re-download)
- KEEP: `*.csv`, `*.log`, `*.json`, `*.pt`, `*.pth`, `config/`, `checkpoints/`

### Step 3: Present to User
```
Cleanup analysis:

SAFE TO DELETE (recommended):
- __pycache__/ (2MB)
- *.tmp files (100MB)
- Old tmux sessions: <list>

PROBABLY DELETABLE:
- data/abc_eval/stls/ (12GB) - can re-extract from 7z

MAYBE KEEP:
- data/downloads/*.7z (20GB) - avoid re-downloading

KEEPING (not deletable):
- checkpoints/*.pt - trained models
- logs/*.log - experiment records

Which buckets should I delete? [SAFE / SAFE+PROBABLY / SAFE+PROBABLY+MAYBE / custom]
```

### Step 4: Execute Cleanup
```bash
# Clean pycache
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "find /home/rohan/perseve -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null"

# Clean tmp files
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "find /home/rohan/perseve -name '*.tmp' -delete 2>/dev/null"

# Kill old tmux sessions
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "tmux kill-session -t <old-session>"

# Remove job from tracking
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "sed -i '/<JOB_NAME>/d' /home/rohan/.claude-jobs.jsonl"
```

### Step 5: Verify Cleanup
```bash
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "df -h /home/rohan"
```

---

## Quick Reference: tmux Commands

| Action | Command |
|--------|---------|
| Start job | `tmux new-session -d -s <name> '<command>'` |
| List sessions | `tmux list-sessions` |
| Check if running | `tmux has-session -t <name>` |
| See output | `tmux capture-pane -t <name> -p -S -100` |
| Attach live | `tmux attach -t <name>` |
| Kill session | `tmux kill-session -t <name>` |

---

## Error Handling

| Error | Solution |
|-------|----------|
| Connection refused/timeout | Check CMU VPN, verify server online |
| Container not found | Start Docker container first |
| CUDA out of memory | Check GPU usage, kill stuck processes |
| Permission denied | Check file ownership, may need sudo |
| tmux session not found | Job may have completed or crashed - check logs |

---

## Safety Notes

- Always run pre-flight checks before long jobs
- Use tmux for anything > 5 minutes
- Verify results are copied before cleanup
- Keep logs of all experiments
- Don't leave zombie processes running
