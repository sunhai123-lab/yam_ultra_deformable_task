# YAM Ultra deformable-ball manipulation with Isaac Lab Newton

This project builds a YAM Ultra 2 tabletop manipulation scene with Isaac Lab and Newton. The robot grasps a VBD deformable ball through physical fingertip contact and friction, then optionally returns it to the table or places it on a dynamic rigid block.

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
OBJECT_RADIUS_RANGE = (0.30, 0.45)
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

`--motion-steps` is a time-scale baseline, not a fixed per-stage step count. The current script requires values `>=240`; use the named speed constants in the source when intentionally tuning motion speed.

## VS Code

The repository launch/tasks configurations point task modes to `scripts/yam_pick_ball.py`. The default build task runs the GUI place-on-block mode. Asset conversion remains a separate utility task.

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
GRIPPER_CHECK_OK
QUICK_GRASP_OK
YAM_DEFORMABLE_BALL_LIFT_OK
YAM_DEFORMABLE_BALL_PUT_BACK_OK
YAM_DEFORMABLE_BALL_ON_BLOCK_OK
EPISODE_COMPLETE
EPISODE_SUMMARY
YAM_DEFORMABLE_TASK_INTEGRATION_OK
```

A success marker means the requested scripted mode passed its configured geometric, motion, support, settling, and lift/placement checks. It is not evidence of general-purpose grasping across arbitrary objects or the full robot workspace.

## Documentation

Use these two documents only:

1. [`docs/yam_pick_ball_reference.md`](docs/yam_pick_ball_reference.md) — parameters and functions;
2. [`docs/yam_pick_ball_workflow.md`](docs/yam_pick_ball_workflow.md) — workflow, validation, running, and debugging.

The source code remains the final authority when a value changes.
