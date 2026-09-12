# YAM Ultra deformable-ball placement

This project builds a YAM Ultra 2 tabletop scene directly with Isaac Lab and uses Newton for coupled rigid/deformable simulation. It does not copy a task or policy from another robotics project.

## Main files

- `scripts/smoke_test_integrated_task.py`: scene creation, randomized reset, hard-coded grasp/lift sequence, Newton stepping, and success checks.
- `source/yam_ultra_deformable_place/yam_ultra_deformable_place/yam_kinematics.py`: project-local forward kinematics, geometric Jacobian, and damped least-squares IK.
- `scripts/convert_yam_ultra_2.py`: converts the vendored YAM URDF to the USD consumed by the task.
- `assets/vendor/i2rt`: original robot descriptions, meshes, attribution, and license.
- `assets/generated/yam_ultra_2`: converted runtime USD asset.
- `.vscode`: F5 and Ctrl+Shift+B launch configurations.

The expected Isaac Lab installation and exact versions are recorded in `VERSIONS.md`.

## One-time package setup

```bash
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/env_isaaclab/bin/python \
-m pip install -e source/yam_ultra_deformable_place
```

## Run

Open this directory as the VS Code workspace. Ctrl+Shift+B runs the GUI physical-grasp milestone. F5 exposes GUI, headless reset, and headless physical-grasp configurations.

Equivalent GUI command:

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/smoke_test_integrated_task.py --visualizer kit --keep-open \
--episodes 1 --steps-per-episode 60 --pick-lift \
--motion-steps 240 --lift-height 0.05 --seed 7
```

Headless 0.05 m physical grasp-and-lift check:

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/smoke_test_integrated_task.py --visualizer none \
--episodes 1 --steps-per-episode 60 --pick-lift \
--motion-steps 240 --lift-height 0.05 --seed 7
```

Headless reset check:

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/smoke_test_integrated_task.py --visualizer none \
--episodes 3 --steps-per-episode 60 --seed 7
```

Regenerate the robot USD only when the vendored URDF or meshes change:

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/convert_yam_ultra_2.py
```

## Current boundary

The first physical-grasp milestone is implemented: randomized rigid/deformable reset, deterministic IK from a reachable top-down branch, open approach, descent, material-dependent finger closure, and vertical lift under Newton two-way coupling.

The generated robot's high-complexity convex finger colliders are disabled because simultaneous contact with the Newton/VBD body is unstable in this Isaac Lab beta. Two simple box collision pads are attached to the real tip_left and tip_right rigid links instead. All soft-body nodes remain free (free_flag=1) throughout the run; there is no node binding, kinematic attachment, or scripted ball following. The ball moves only through simulated normal contact and Coulomb friction.

The current default command is a 0.05 m lift. This first-stage "clear the table" check requires the measured soft-ball center to rise by at least 70% of the commanded end-effector displacement, allowing for elastic compression, mesh variation, and small physical slip. The latest headless run raised the center by 0.03784 m and maintained it above the table. A collision-safe 0.4 m transport path, release above the rigid block, and placement-success evaluation remain future milestones.
