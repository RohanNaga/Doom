#!/bin/bash
# Model Training Workflow
# Usage: bash training.sh <config_file> [experiment_name]

CONFIG=${1:-"config.yaml"}
EXPERIMENT=${2:-"experiment_$(date +%Y%m%d_%H%M%S)"}

SERVER="rohan@128.2.204.110"
REMOTE_DIR="/home/rohan/perseve"
LOG_DIR="/home/rohan/logs"
LOCAL_RESULTS="$HOME/Downloads/$EXPERIMENT"

echo "=== Model Training ==="
echo "Server: $SERVER"
echo "Config: $CONFIG"
echo "Experiment: $EXPERIMENT"
echo ""

# Step 1: Pull latest code
echo "[1/6] Pulling latest code..."
ssh $SERVER "cd $REMOTE_DIR && git pull"

# Step 2: Check GPU availability
echo "[2/6] Checking GPU status..."
ssh $SERVER "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv"

# Step 3: Create log directory
echo "[3/6] Setting up logging..."
ssh $SERVER "mkdir -p $LOG_DIR"

# Step 4: Run training
echo "[4/6] Starting training..."
LOG_FILE="$LOG_DIR/${EXPERIMENT}.log"
ssh $SERVER "cd $REMOTE_DIR && source ~/.bashrc && conda activate PERSEVE && python src/perseve/fine_tuning/train.py --config $CONFIG 2>&1 | tee $LOG_FILE"

# Step 5: Check results
echo "[5/6] Checking training results..."
ssh $SERVER "ls -la $REMOTE_DIR/checkpoints/ | tail -10"

# Step 6: Copy results locally
echo "[6/6] Copying results to $LOCAL_RESULTS..."
mkdir -p "$LOCAL_RESULTS"
scp -r $SERVER:$REMOTE_DIR/checkpoints/* "$LOCAL_RESULTS/"
scp $SERVER:$LOG_FILE "$LOCAL_RESULTS/"

echo ""
echo "=== Complete ==="
echo "Results saved to: $LOCAL_RESULTS"

# Alternative: Run in background with nohup
# Uncomment below to run training in background (survives disconnection)
# echo "Running in background mode..."
# ssh $SERVER "cd $REMOTE_DIR && nohup bash -c 'source ~/.bashrc && conda activate PERSEVE && python src/perseve/fine_tuning/train.py --config $CONFIG' > $LOG_FILE 2>&1 &"
# echo "Training started in background. Check progress with:"
# echo "  ssh $SERVER 'tail -f $LOG_FILE'"
