# YAM Ultra deformable-ball manipulation with Isaac Lab Newton

This project builds a YAM Ultra 2 tabletop manipulation scene with Isaac Lab and Newton. The robot grasps a VBD deformable ball through physical fingertip contact and friction, then optionally returns it to the table or places it on a dynamic rigid block.

The validated software baseline is Isaac Lab `v3.0.0-beta2.patch1` with Isaac Sim `6.0.1.0`, Python `3.12.14`, Newton `1.2.1`, and Warp `1.13.0`. See [`VERSIONS.md`](VERSIONS.md) for the complete environment record.

[简体中文](README_zh-CN.md)

## Current structure

There is one task runtime entry point:

```text
scripts/yam_pick_ball.py
```

Do not add parallel copies of the task runner. New task modes should be added as command-line options or internal functions in `yam_pick_ball.py`.

Other important files are:

- `source/yam_ultra_deformable_place/yam_ultra_deformable_place/yam_kinematics.py` — project-local PyTorch FK, geometric Jacobian, and DLS IK;
- `scripts/convert_yam_ultra_2.py` — asset conversion utility only; it is not a second task runner;
- `docs/yam_pick_ball_reference.md` — parameters, functions, variables, and core math;
- `docs/yam_pick_ball_workflow.md` — run modes, state machine, validation, debugging, and maintenance conventions;
- `VERSIONS.md` — validated environment versions and current known limitations.

## Task pipeline

```text
sample/reset
  -> wait for block and deformable ball to settle
  -> pre-grasp IK
  -> approach
  -> contact-sensitive descent
  -> close gripper
  -> physical grasp check
  -> lift
  -> optional polar transfer
  -> lower/release
  -> retreat
  -> final settling check
```

The controller is deterministic scripted control, not reinforcement learning. The ball is not attached to the gripper: lifting depends on the original fingertip geometry, collision, deformation, normal contact, and friction.

## Current workspace layout

The learning task uses a table-centered robot and annular object sampling:

```python
TABLE_SIZE = (1.50, 1.50, 0.08)
TABLE_CENTER = (0.0, 0.0, 0.0)
ROBOT_BASE_XY = TABLE_CENTER[:2]
OBJECT_RADIUS_RANGE = (0.30, 0.43)
OBJECT_BEARING_RANGE = (-2.40, 3.00)
MIN_OBJECT_PLANAR_DISTANCE = 0.18
MAX_TRANSFER_BEARING_DELTA = math.pi
```

Candidate positions are sampled with

```text
x = robot_x + r cos(theta)
y = robot_y + r sin(theta)
```

and then filtered for table containment, minimum object separation, and transfer-bearing constraints. This is still a constrained test workspace; sampling inside the region does not prove that every full end-effector pose is reachable.

## One-time package setup

```bash
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/env_isaaclab/bin/python \
-m pip install -e source/yam_ultra_deformable_place
```

Run the commands below from the repository root. They use the matching Isaac Lab launcher and its `env_isaaclab` Python environment.

## Run

### Reset and settling only

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/yam_pick_ball.py --visualizer none \
--episodes 3 --steps-per-episode 60 --seed 7
```

### Gripper open/close diagnostic

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/yam_pick_ball.py --visualizer kit --keep-open \
--gripper-check --episodes 1 --steps-per-episode 60 --seed 7
```

### Physical grasp and lift

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/yam_pick_ball.py --visualizer none \
--pick-lift --episodes 1 --steps-per-episode 60 --lift-height 0.03 --seed 7
```

### Grasp, lift, and return to the original position

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/yam_pick_ball.py --visualizer none \
--pick-lift --put-back --episodes 1 --steps-per-episode 60 --lift-height 0.03 --seed 7
```

### Grasp, lift, and place on the rigid block

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/yam_pick_ball.py --visualizer kit --keep-open \
--pick-lift --place-on-block --episodes 3 --steps-per-episode 60 --lift-height 0.03 --seed 7
```

For a large batch, increase `--episodes`, for example:

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/yam_pick_ball.py --visualizer none \
--pick-lift --place-on-block --episodes 50 --steps-per-episode 60 \
--motion-steps 240 --lift-height 0.03 --seed 7
```

`--motion-steps` is a time-scale baseline, not a fixed per-stage step count. The current script requires values `>=240`; use the named speed constants in the source when intentionally tuning motion speed.

Each episode samples new object positions, resets the robot, block, and deformable ball, waits for settling, and runs the selected task. A recoverable task failure is reported as `EPISODE_FAILED` and the next episode starts automatically. The batch summary reports `complete_ok`, `failed`, `settle_ok`, `pick_ok`, and `placement_ok`. Numerical corruption such as NaN/Inf or a CUDA illegal-memory error stops the process because the simulator state cannot be trusted afterward.

### Latest robustness validation

The latest logged batch used the current annular workspace and ran 200 randomized episodes:

```text
mode=PICK_LIFT_ON_BLOCK  episodes=200  seed=7
radius=(0.30, 0.43)      bearing=(-2.40, 3.00)
```

| Check | Result |
|---|---:|
| Scene settling | 200/200 |
| Physical pick and lift | 199/200 |
| Ball remains on the block after release and settling | 199/200 |
| Completed episodes | 199/200 |
| Overall rate | 99.5% |

The only failed episode stopped at `pick_and_lift` with `QUICK_GRASP_TIMEOUT`: the stability counter reached `1/12`. The failure snapshot still showed the ball enveloped by the fingers, a ball-to-gripper offset of about `0.84 mm`, no excessive deformation, and no joint-limit saturation. This is therefore classified as a grasp-stability-check timeout, not confirmed ball loss.

There were 42 `RETREAT_INCOMPLETE` / `MOTION_TIMEOUT` warnings during post-release retreat. They were recoverable and did not cause a placement failure; the ball passed the final support check in every otherwise completed episode. The final placement criterion is that the ball remains supported by the block after release settling and observation, rather than requiring the ball center to exactly coincide with the block center.

This result supports good robustness for the tested object material, robot pose, workspace annulus, controller parameters, and seed. It is not yet a statistical guarantee of general-purpose manipulation: more seeds, wider workspace/material ranges, and repeated runs should be used before making that claim. The full raw record is [the 200-episode log](logs/yam_pick_ball_20260916_135206.log).

## VS Code

The repository launch/tasks configurations point task modes to `scripts/yam_pick_ball.py`.

- F5: `Task: Place ball on block GUI`, 3 episodes, with `--keep-open` after the batch;
- Ctrl+Shift+B: the default build task, 50 GUI episodes, with `--keep-open` after the batch;
- other launch/tasks entries cover reset/settle, pick/lift, gripper diagnostics, and asset conversion.

If a GUI batch contains recoverable failures, the terminal continues to the next episode. After the final summary, close the GUI window manually.

## Asset regeneration

Regenerate the robot USD only when the vendored URDF or meshes change:

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/convert_yam_ultra_2.py
```

## Important success markers

Useful terminal markers include:

```text
RUN_CONFIG
EPISODE_START
EPISODE_COMPLETE
EPISODE_FAILED
EPISODE_SUMMARY
RUN_SUCCESS
RUN_COMPLETE
GRASP_FAILURE
```

`EPISODE_COMPLETE` means that episode passed its configured validation checks. `RUN_SUCCESS` means every requested episode completed. `RUN_COMPLETE` means the batch finished while one or more recoverable episodes failed; inspect each `EPISODE_FAILED` line and the summary counts. `GRASP_FAILURE` indicates a top-level fatal exception. Passing these scripted checks is not evidence of general-purpose grasping across arbitrary objects or the full robot workspace.

## Documentation

Use these two documents only:

1. [`docs/yam_pick_ball_reference.md`](docs/yam_pick_ball_reference.md) — parameters and functions;
2. [`docs/yam_pick_ball_workflow.md`](docs/yam_pick_ball_workflow.md) — workflow, validation, running, and debugging.

The source code remains the final authority when a value changes.
