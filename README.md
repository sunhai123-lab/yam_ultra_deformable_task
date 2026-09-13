# YAM Ultra deformable-ball placement

This project builds a YAM Ultra 2 tabletop scene directly with Isaac Lab and uses Newton for coupled rigid/deformable simulation. It does not copy a task or policy from another robotics project.

## Current status

The integrated task now runs end to end in the project Isaac Lab/Newton environment. The validated control path includes:

- randomized tabletop reset for one dynamic rigid block and one deformable ball;
- scene settling checks for both objects before manipulation;
- pre-grasp IK, approach, contact-sensitive descent, gripper closure, and short physical-grasp verification;
- lifting the deformable ball clear of the table;
- optional return to the original ball position with `--put-back`;
- optional transfer to the dynamic rigid block and release with `--place-on-block`;
- post-release retreat and settling checks;
- multi-episode execution with per-episode success markers and timing output.

The robot asset is not modified on disk. At runtime the original fingertip mesh collisions are made uninstanceable and their collision approximation is set to `convexDecomposition`; no extra collision plates or node binding are added.

A successful run is therefore evidence that the scripted task works inside the **constrained test scene described below**. It is not evidence of general grasping robustness over the full robot workspace or arbitrary object configurations.

## Main files

- `scripts/smoke_test_integrated_task.py`: scene creation, constrained randomized reset, grasp/lift/place state machine, Newton stepping, diagnostics, and success checks.
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

Open this directory as the VS Code workspace. Ctrl+Shift+B runs the isolated original-gripper diagnostic. F5 exposes GUI/headless task configurations.

### Isolated gripper open/close check

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/smoke_test_integrated_task.py --visualizer kit --keep-open \
--episodes 1 --steps-per-episode 60 --gripper-check \
--motion-steps 240 --lift-height 0.03 --seed 7
```

### Physical grasp and lift

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/smoke_test_integrated_task.py --visualizer none \
--episodes 1 --steps-per-episode 60 --pick-lift \
--motion-steps 240 --lift-height 0.03 --seed 7
```

### Grasp, lift, and put the ball back

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/smoke_test_integrated_task.py --visualizer none \
--episodes 1 --steps-per-episode 60 --pick-lift --put-back \
--motion-steps 240 --lift-height 0.03 --seed 7
```

### Grasp, lift, and place on the dynamic block

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/smoke_test_integrated_task.py --visualizer none \
--episodes 3 --steps-per-episode 60 --pick-lift --place-on-block \
--motion-steps 240 --lift-height 0.03 --seed 7
```

### Reset/settling-only check

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

## Test-scene limits and constrained randomization

The randomized episodes are intentionally **not** unconstrained samples over the full tabletop or full robot workspace. The sampler rejects configurations that do not match the scripted task assumptions.

Current XY sampling ranges are:

| Object | X range (m) | Y range (m) |
| --- | ---: | ---: |
| Dynamic block | `[-0.02, 0.28]` | `[0.06, 0.26]` |
| Deformable ball | `[-0.02, 0.28]` | `[-0.26, -0.06]` |

Additional sampling constraints are:

- robot base XY is fixed at `(-0.22, 0.0)`;
- each sampled object must have a horizontal base distance between `0.20 m` and `0.40 m`;
- the block and ball must be at least `0.18 m` apart in the XY plane;
- the block and ball intentionally occupy different Y-side bands, so their sampling regions do not represent arbitrary relative placement;
- sampling retries at most 100 times before failing;
- only XY is randomized. Object orientation, geometry, materials, table pose, robot base pose, and nominal Z placement are fixed;
- the dynamic block always starts upright and is placed only `0.5 mm` above its expected support height before settling;
- the ball is positioned from its actual lowest VBD node plus particle radius and reset clearance rather than from an arbitrary drop height;
- after the first naturally settled episode, the code may reuse the settled deformable nodal shape as the reset template for later episodes, translating it to the newly sampled position and zeroing nodal velocity. Later episodes therefore do not all begin from the original undeformed mesh state.

These restrictions make the randomized test useful for regression testing of the current scripted manipulation pipeline, but they substantially reduce the problem compared with unrestricted object randomization.

## Manipulation and acceptance limits

The test is also specialized in several other ways:

- there is exactly one 4 cm-radius deformable sphere and one 7 cm rigid cube;
- the hard-coded controller assumes a downward gripper orientation and uses project-local DLS IK rather than a general motion planner;
- the approach, descent, lift, transfer, placement, and release speeds are tuned for this scene;
- the grasp target is derived from a fixed nominal compression heuristic rather than measured contact force;
- `soft_contact_mu`, deformable stiffness/damping, rigid friction, solver iterations, and substeps are experimental simulation parameters, not material calibration results;
- success is based on geometry, motion, support, settling, and lift/placement thresholds; there are no fingertip force/torque sensors in the acceptance logic;
- `--place-on-block` assumes the dynamic block remains nearly horizontal and within the configured displacement tolerance while the ball is released;
- passing a finite number of random episodes does not establish success outside the configured ranges, for different object shapes/sizes/materials, or under arbitrary rotations.

## Important success markers

Depending on the selected mode, useful terminal markers include:

- `GRIPPER_CHECK_OK`
- `QUICK_GRASP_OK lifted=False`
- `QUICK_GRASP_OK lifted=True`
- `YAM_DEFORMABLE_BALL_LIFT_OK`
- `YAM_DEFORMABLE_BALL_PUT_BACK_OK`
- `YAM_DEFORMABLE_BALL_ON_BLOCK_OK`
- `EPISODE_SUMMARY`
- `YAM_DEFORMABLE_TASK_INTEGRATION_OK`

`YAM_DEFORMABLE_TASK_INTEGRATION_OK` means the requested scripted mode completed its configured acceptance checks. Interpret it together with the mode-specific marker and the test-scene limitations above.

See [the revised grasp/validation notes](docs/grasp_revision_plan.md) for the geometry, staged acceptance criteria, implementation history, and remaining limits.
