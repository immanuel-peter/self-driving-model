## Inference (Overview)

For full arguments and environment variables, open each file.

> Note: Feel free to modify these scripts to align with your development workflow.

---

### `run_automoe.sh`
- Purpose: Shell wrapper for closed-loop CARLA inference with AutoMoE. Sets defaults, logging, optional frame/GIF export, then launches the Python runner.
- Run:
    ```bash
    bash inference/run_automoe.sh
    ```

---

### `run_automoe.py`
- Purpose: Python runner that connects to CARLA, loads AutoMoE (experts + gating), and executes closed-loop driving with optional recording.
- Run:
    ```bash
    python3 inference/run_automoe.py
    ```


