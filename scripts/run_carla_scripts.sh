#!/bin/bash
# scripts/run_carla_scripts.sh

set -euo pipefail

# Parse command line arguments
RESUME_FROM=1
while [[ $# -gt 0 ]]; do
  case $1 in
    --resume-from)
      RESUME_FROM="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--resume-from RUN_NUMBER]"
      exit 1
      ;;
  esac
done

# Set up log file and directory
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/carla_data_collection_$(date +'%Y%m%d_%H%M%S').log"

# Configure logging so set -e still works with tee, and tqdm bars are visible in log
{
  :
} > /dev/null
exec > >(tee "${LOGFILE}") 2>&1

# Graceful failure handler
trap 'echo ""; echo "=============================================================="; echo "CARLA DATA COLLECTION FAILED"; echo "End Time: $(date)"; echo "Log: ${LOGFILE}"; echo "=============================================================="; exit 1' ERR

echo "=============================================================="
echo "STARTING CARLA DATA COLLECTION"
echo "Start Time: $(date)"
echo "Resuming from run: ${RESUME_FROM}"
echo "Logging to: ${LOGFILE}"
echo "=============================================================="
echo ""

export CARLA_DATA_PATH=/home/ubuntu/self-driving-model/datasets/carla/raw

# Function to run a command only if we haven't passed the resume point
run_if_not_passed() {
    local run_number=$1
    shift
    if [[ $run_number -ge $RESUME_FROM ]]; then
        echo "Running command for run ${run_number}: $@"
        "$@"
    else
        echo "Skipping run ${run_number} (resume-from is ${RESUME_FROM})"
    fi
}

# Multi-run jobs (runs 1-4)
run_if_not_passed 1 python3 scripts/collect_autopilot_data.py --runs 4 --continue-from 1 --vehicles 10 --walkers 10 --duration 600

# Single-run jobs (runs 5-16)
run_if_not_passed 5 python3 scripts/collect_autopilot_data.py --single-run --continue-from 5 --vehicles 25 --walkers 15
run_if_not_passed 6 python3 scripts/collect_autopilot_data.py --single-run --continue-from 6 --vehicles 30 --walkers 18
run_if_not_passed 7 python3 scripts/collect_autopilot_data.py --single-run --continue-from 7 --vehicles 5 --walkers 2
run_if_not_passed 8 python3 scripts/collect_autopilot_data.py --single-run --continue-from 8 --vehicles 40 --walkers 25
run_if_not_passed 9 python3 scripts/collect_autopilot_data.py --single-run --continue-from 9 --vehicles 45 --walkers 28
run_if_not_passed 10 python3 scripts/collect_autopilot_data.py --single-run --continue-from 10 --vehicles 35 --walkers 20
run_if_not_passed 11 python3 scripts/collect_autopilot_data.py --single-run --continue-from 11 --vehicles 50 --walkers 30
run_if_not_passed 12 python3 scripts/collect_autopilot_data.py --single-run --continue-from 12 --duration 900 --vehicles 25 --walkers 15
run_if_not_passed 13 python3 scripts/collect_autopilot_data.py --single-run --continue-from 13 --duration 1200 --vehicles 30 --walkers 18
run_if_not_passed 14 python3 scripts/collect_autopilot_data.py --single-run --continue-from 14 --duration 900 --vehicles 35 --walkers 20
run_if_not_passed 15 python3 scripts/collect_autopilot_data.py --single-run --continue-from 15 --save-every 3 --vehicles 20 --walkers 12
run_if_not_passed 16 python3 scripts/collect_autopilot_data.py --single-run --continue-from 16 --save-every 2 --vehicles 15 --walkers 8 --duration 300

# Multi-run jobs (runs 17-30)
run_if_not_passed 17 python3 scripts/collect_autopilot_data.py --runs 5 --continue-from 17 --vehicles 25 --walkers 15
run_if_not_passed 22 python3 scripts/collect_autopilot_data.py --runs 3 --continue-from 22 --vehicles 35 --walkers 20 --duration 900
run_if_not_passed 25 python3 scripts/collect_autopilot_data.py --runs 4 --continue-from 25 --vehicles 40 --walkers 25 --duration 600
run_if_not_passed 29 python3 scripts/collect_autopilot_data.py --runs 2 --continue-from 29 --vehicles 20 