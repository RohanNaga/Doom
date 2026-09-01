#!/bin/bash
# Isaac Sim Synthetic Data Generation Workflow
# Usage: bash isaac-sim.sh [start_idx] [end_idx] [gpu_id]

START_IDX=${1:-0}
END_IDX=${2:-10}
GPU_ID=${3:-0}

SERVER="rohan@128.2.204.110"
REMOTE_DIR="/home/rohan/perseve"
LOCAL_RESULTS="$HOME/Downloads/sdg_results_$(date +%Y%m%d_%H%M%S)"

echo "=== Isaac Sim Synthetic Data Generation ==="
echo "Server: $SERVER"
echo "Range: $START_IDX to $END_IDX"
echo "GPU: $GPU_ID"
echo ""

# Step 1: Pull latest code
echo "[1/5] Pulling latest code..."
ssh $SERVER "cd $REMOTE_DIR && git pull"

# Step 2: Check if container is running
echo "[2/5] Checking Isaac Sim container..."
CONTAINER_STATUS=$(ssh $SERVER "docker ps | grep isaac-sim" 2>/dev/null)
if [ -z "$CONTAINER_STATUS" ]; then
    echo "Container not running. Starting..."
    ssh $SERVER "cd $REMOTE_DIR/src/perseve/synthetic_data_generation && bash docker/isaac_sim_docker.sh"
    sleep 10  # Wait for container startup
else
    echo "Container already running."
fi

# Step 3: Run synthetic data generation
echo "[3/5] Running synthetic data generation..."
ssh $SERVER "docker exec -i isaac-sim bash -c 'cd /workspace/perseve/src/perseve/synthetic_data_generation && bash run_scene_based_sdg.sh --start-idx $START_IDX --end-idx $END_IDX --gpu-id $GPU_ID'"

# Step 4: Check results
echo "[4/5] Checking generated data..."
ssh $SERVER "ls -la $REMOTE_DIR/src/perseve/synthetic_data_generation/_out_multi_object/ | tail -20"

# Step 5: Copy results locally
echo "[5/5] Copying results to $LOCAL_RESULTS..."
mkdir -p "$LOCAL_RESULTS"
scp -r $SERVER:$REMOTE_DIR/src/perseve/synthetic_data_generation/_out_multi_object/* "$LOCAL_RESULTS/"

echo ""
echo "=== Complete ==="
echo "Results saved to: $LOCAL_RESULTS"
