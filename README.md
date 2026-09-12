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

Open this directory as the VS Code workspace. Ctrl+Shift+B runs the isolated original-gripper diagnostic. F5 exposes GUI, headless reset, and headless physical-grasp configurations.

Equivalent GUI command:

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/smoke_test_integrated_task.py --visualizer kit --keep-open \
--episodes 1 --steps-per-episode 60 --gripper-check \
--motion-steps 240 --lift-height 0.03 --seed 7
```

Headless 0.03 m physical grasp-and-lift check:

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/smoke_test_integrated_task.py --visualizer none \
--episodes 1 --steps-per-episode 60 --pick-lift \
--motion-steps 240 --lift-height 0.03 --seed 7
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

The previous physical-grasp success claim is withdrawn: it used added collision plates with an incorrect open/close convention. The script now uses the original robot collision meshes and no added plates. q=-0.04695 opens the fingers; q=0 closes them.

Ctrl+Shift+B and the first F5 entry now run the isolated original-gripper open/close diagnostic, not a grasp. The headless grasp entry commands 3 cm. Contact stability and lift with the original fingers remain unverified.

See [the revised plan](docs/grasp_revision_plan.md) for geometric evidence, staged commands, acceptance criteria, and limitations. For close/hold only use --pick-lift --stop-after-hold --episodes 1; for the isolated open/close diagnostic use --gripper-check --episodes 1.
