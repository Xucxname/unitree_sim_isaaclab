# Unitree Benchmark Flow

This project runs migrated GenieSim benchmark tasks through Unitree G129/Dex1 IsaacLab environments.

## Scope

The flow adapts GenieSim's benchmark structure to the current Unitree runtime:

```text
GenieSim selected task table
  -> Unitree Gym task id
  -> Genie background USD + task foreground USD
  -> G129 Dex1 fixed-base env cfg
  -> policy action loop
  -> JSONL episode result + summary
```

Robot adaptation:

- Robot: `G1RobotPresets.g1_29dof_dex1_base_fix`
- Hand: Dex1 gripper
- Base height: fixed to `z = 0.55`
- Task assets: loaded from `genie/benchmark/config/llm_task/<task>/<instance>/scene.usda`
- Backgrounds: loaded from the migrated `genie/assets/background/...` tree
- Cameras: enabled automatically by `benchmark_main.py` because Unitree task configs spawn wrist/front cameras.

## Entry Points

List migrated benchmark task ids:

```bash
python3 benchmark_main.py --headless --list_tasks
```

Run all migrated tasks, instance `0`, one episode each:

```bash
scripts/run_unitree_benchmark.sh
```

Run one task:

```bash
python3 benchmark_main.py \
  --headless \
  --device cpu \
  --tasks Isaac-Scoop-Popcorn-G129-Dex1-Joint \
  --instances 0 \
  --episodes 1 \
  --policy zero
```

Task names may be either Genie task keys, such as `scoop_popcorn`, or full Gym ids.

## Output

Results are written under:

```text
output/unitree_benchmark/<timestamp>/
  episodes.jsonl
  summary.json
```

Each episode record includes task id, instance id, instruction, policy name, step count, reward totals, done status, and error text if the episode failed.

## Policy Adapters

Current adapters:

- `zero`: hold default Unitree joint targets. This validates task loading, reset, stepping, and result logging.
- `replay`: replay a JSON action list through `--replay_path`.

Real benchmark policies should be added in `g1bench/policies.py` by implementing the same `reset(env, instruction)` and `act(env, observation, step)` interface.

## GenieSim Parity Notes

This flow keeps GenieSim's task/instance/instruction/problem files and StepOut limits, but it does not reuse GenieSim's G2 robot, APICore, ROS data courier, or VLM scoring stack. Those parts are intentionally replaced by the Unitree IsaacLab environment and policy loop.

The current score is simulation-loop level telemetry, not VLM task success. To reproduce GenieSim's final VLM success score, add a Unitree-compatible evaluator that consumes camera observations and the instruction/problem metadata after each episode.
