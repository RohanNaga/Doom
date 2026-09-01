# Server Configuration & Error Handling

Detailed server configuration and troubleshooting guide.

---

## Server Details

### Spiderman (Primary)

| Property | Value |
|----------|-------|
| Server Name | Spiderman |
| IP Address | 128.2.204.110 |
| SSH User | rnagabhi (Andrew ID — changed from rohan post-rebuild March 2026) |
| GPUs | 4x NVIDIA RTX A6000 (49GB each) |
| CUDA Driver | 12.8 |
| Home | /home/rnagabhi (3.4TB volume, 3.0TB free) |
| Old Data | /sata1/data/rohan/ (permission-locked, need Peiqi to chown) |
| Docker | Needs install/config — check `docker info` before use |
| Purpose | Isaac Sim, training, experiments |
| Network | CMU VPN required |

### Superman (Backup / Overflow)

| Property | Value |
|----------|-------|
| Server Name | Superman |
| IP Address | 128.2.204.116 |
| SSH User | rohan |
| GPUs | 8x NVIDIA RTX A4000 (16GB each) |
| CUDA Driver | 12.4 |
| Disk | ~3.5TB (320GB free as of 2026-02-19) |
| Purpose | Backup when Spiderman is busy, overflow projects |
| Network | CMU VPN required |
| Project Path | /home/rohan/lego_project/ |

**Superman resource constraints (shared server):**

| Resource | Limit | Action |
|----------|-------|--------|
| GPUs in use | Max 6 of 8 | Leave 2 GPUs free for other users |
| Disk usage (ours) | Warn at 100GB | Ask user to confirm |
| Disk free | Min 250GB | BLOCK if available disk < 250GB |
| Per-GPU VRAM | 16GB | Use batch size 2 + gradient accumulation |

**GPU selection rule (ALL servers):** Always select the **highest numbered free GPUs** first (e.g., if GPUs 1-7 are free, use 7, then 6, then 5, etc.). Free = 0% utilization and ~5MiB memory. When using multiple GPUs, take the last N free GPUs. Always run `nvidia-smi` before launching to verify. This avoids conflicts with other users who typically start from GPU 0.

---

## Connection Setup

### SSH with sshpass

All commands use `sshpass` with the `$PERSEVE_SERVER_PASSWORD` environment variable.

**One-time setup:**
```bash
# Install sshpass (macOS)
brew install hudochenkov/sshpass/sshpass

# Add to ~/.zshrc
export PERSEVE_SERVER_PASSWORD="your_password"
```

**Usage pattern:**
```bash
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "<command>"
sshpass -p "$PERSEVE_SERVER_PASSWORD" scp <source> rnagabhi@128.2.204.110:<dest>
```

### VPN Requirement
The server requires CMU VPN connection. Verify connectivity with:
```bash
ping 128.2.204.110
```

---

## Docker Configuration

### Isaac Sim Container

| Property | Value |
|----------|-------|
| Container Name | isaac-sim |
| Workspace Mount | /workspace/perseve |
| Host Directory | /home/rohan/perseve |

**Start Container:**
```bash
cd /home/rnagabhi/perseve/src/perseve/synthetic_data_generation && bash docker/isaac_sim_docker.sh
```

**Check Container:**
```bash
docker ps | grep isaac-sim
```

**Execute in Container:**
```bash
docker exec -i isaac-sim bash -c '<command>'
```

---

## Error Handling

### "Connection refused" or timeout

**Causes:**
- Not on CMU VPN
- Server offline
- Network issues

**Solutions:**
1. Check VPN connection: `ping 128.2.204.110`
2. Verify server is online (contact lab if needed)
3. Try manual SSH: `sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110`

---

### "Container not found"

**Causes:**
- Docker container not running
- Docker daemon stopped

**Solutions:**
1. Start the Docker container:
   ```bash
   sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "cd /home/rnagabhi/perseve/src/perseve/synthetic_data_generation && bash docker/isaac_sim_docker.sh"
   ```
2. Check Docker daemon:
   ```bash
   sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "sudo systemctl status docker"
   ```
3. Restart Docker if needed:
   ```bash
   sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "sudo systemctl restart docker"
   ```

---

### "CUDA out of memory"

**Causes:**
- Other processes using GPU
- Batch size too large
- Memory leak from previous run

**Solutions:**
1. Check what's using GPU:
   ```bash
   sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "nvidia-smi"
   ```
2. Kill stuck processes:
   ```bash
   sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "pkill -f <process_name>"
   ```
3. Reduce batch size in config
4. Clear GPU memory:
   ```bash
   sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "nvidia-smi --gpu-reset"
   ```

---

### "Permission denied"

**Causes:**
- File ownership issues
- Missing sudo permissions
- Protected directory

**Solutions:**
1. Check file ownership:
   ```bash
   sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "ls -la /path/to/file"
   ```
2. Fix ownership:
   ```bash
   sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "sudo chown rohan:rohan /path/to/file"
   ```
3. Use sudo for protected operations

---

## Monitoring Commands

### GPU Status
```bash
# One-time check
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "nvidia-smi"

# Continuous monitoring
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "watch -n 5 nvidia-smi"

# Formatted output
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv"
```

### Process Management
```bash
# Find process
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "ps aux | grep <name>"

# Kill process
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "pkill -f <name>"

# Kill by PID
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "kill -9 <pid>"
```

### Disk Usage
```bash
# Check home directory
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "df -h /home/rohan"

# Find large files
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "du -sh /home/rohan/* | sort -h"
```

---

## Long-Running Tasks

### Using nohup
```bash
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "nohup <command> > /home/rohan/logs/job_$(date +%Y%m%d_%H%M%S).log 2>&1 &"
```

### Using screen
```bash
# Start screen session
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "screen -S experiment"

# Detach: Ctrl+A, D

# Reattach
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "screen -r experiment"

# List sessions
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "screen -ls"
```

### Using tmux
```bash
# Start tmux session
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "tmux new -s experiment"

# Detach: Ctrl+B, D

# Reattach
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "tmux attach -t experiment"

# List sessions
sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh rnagabhi@128.2.204.110 "tmux ls"
```
