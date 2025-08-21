#!/bin/bash

# NOTE: Update this script to run however you see fit.

# Multi-run jobs (runs 1-4)
python3 scripts/collect_autopilot_data.py --runs 4 --continue-from 1 --vehicles 10 --walkers 10 --duration 600

# Single-run jobs (runs 5-16)
python3 scripts/collect_autopilot_data.py --single-run --continue-from 5 --vehicles 25 --walkers 15
python3 scripts/collect_autopilot_data.py --single-run --continue-from 6 --vehicles 30 --walkers 18
python3 scripts/collect_autopilot_data.py --single-run --continue-from 7 --vehicles 5 --walkers 2
python3 scripts/collect_autopilot_data.py --single-run --continue-from 8 --vehicles 40 --walkers 25
python3 scripts/collect_autopilot_data.py --single-run --continue-from 9 --vehicles 45 --walkers 28
python3 scripts/collect_autopilot_data.py --single-run --continue-from 10 --vehicles 35 --walkers 20
python3 scripts/collect_autopilot_data.py --single-run --continue-from 11 --vehicles 50 --walkers 30
python3 scripts/collect_autopilot_data.py --single-run --continue-from 12 --duration 900 --vehicles 25 --walkers 15
python3 scripts/collect_autopilot_data.py --single-run --continue-from 13 --duration 1200 --vehicles 30 --walkers 18
python3 scripts/collect_autopilot_data.py --single-run --continue-from 14 --duration 900 --vehicles 35 --walkers 20
python3 scripts/collect_autopilot_data.py --single-run --continue-from 15 --save-every 3 --vehicles 20 --walkers 12
python3 scripts/collect_autopilot_data.py --single-run --continue-from 16 --save-every 2 --vehicles 15 --walkers 8 --duration 300

# Multi-run jobs (runs 17-30)
python3 scripts/collect_autopilot_data.py --runs 5 --continue-from 17 --vehicles 25 --walkers 15
python3 scripts/collect_autopilot_data.py --runs 3 --continue-from 22 --vehicles 35 --walkers 20 --duration 900
python3 scripts/collect_autopilot_data.py --runs 4 --continue-from 25 --vehicles 40 --walkers 25 --duration 600
python3 scripts/collect_autopilot_data.py --runs 2 --continue-from 29 --vehicles 20 --walkers 10 --duration 600
