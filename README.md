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

Open this directory as the VS Code workspace. Ctrl+Shift+B runs the GUI grasp-and-lift configuration. F5 exposes GUI, headless reset, and headless grasp-and-lift configurations.

Equivalent GUI command:

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/smoke_test_integrated_task.py --visualizer kit --keep-open \
--episodes 1 --steps-per-episode 60 --pick-lift \
--motion-steps 240 --lift-height 0.4 --seed 7
```

Headless 0.4 m grasp-and-lift check:

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/smoke_test_integrated_task.py --visualizer none \
--episodes 1 --steps-per-episode 60 --pick-lift \
--motion-steps 240 --lift-height 0.4 --seed 7
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

The scene, randomized rigid/deformable reset, Newton two-way coupling, hard-coded inverse-kinematics grasp sequence, and 0.4 m lift check are implemented. The current check requires the deformable ball's measured center to rise by at least 85% of the commanded distance.

Isaac Lab 3.0.0 beta2 currently exits natively when this robot's two convex finger colliders simultaneously contact the Newton/VBD soft body. The stable workaround disables robot collisions and pre-declares two small surface patches of the ball as kinematic nodes during reset. When the fingers close, those patches follow the gripper frame; the remaining soft-body nodes continue to deform under Newton/VBD. Thus the current milestone validates hard-coded IK, gripper motion, soft-body deformation, and lift tracking, but it is not yet a pure frictional-contact grasp. Transport to the block, release, and placement-success evaluation remain future work.
