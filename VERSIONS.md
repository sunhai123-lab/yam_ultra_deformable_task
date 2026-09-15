# Reproducible environment

## Environment baseline

The project environment baseline was validated with:

- Isaac Lab source tag: `v3.0.0-beta2.patch1`
- Isaac Sim: `6.0.1.0`
- Python: `3.12.14`
- PyTorch: `2.10.0+cu128`
- Newton: `1.2.1`
- Warp: `1.13.0`
- I2RT YAM Ultra 2 asset commit: `5b72c47239bd056d0fa6c1a39edeb0537c89443c`

Environment path used by the repository launch configurations:

```text
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/env_isaaclab
```

## Current task architecture

The repository now has one task runtime entry point:

```text
scripts/yam_pick_ball.py
```

The project-local kinematics implementation remains in:

```text
source/yam_ultra_deformable_place/yam_ultra_deformable_place/yam_kinematics.py
```

`scripts/convert_yam_ultra_2.py` is only an asset conversion utility and is not a task-runner variant.

The current task uses:

- Newton `CoupledMJWarpVBDSolverCfg` with MJWarp rigid/articulation dynamics and VBD deformable dynamics;
- `two_way` rigid/deformable coupling;
- project-local URDF-derived PyTorch FK and geometric Jacobian;
- DLS position/pose IK instead of the simulator articulation Jacobian in the coupled VBD path;
- runtime fingertip `convexDecomposition` collision approximation without modifying the vendored robot asset on disk;
- table-centered robot placement and annular randomized object sampling;
- radial gripper orientation, joint1-compatible IK seeding, and continuous polar transfer for opposite-side object configurations;
- staged grasp, lift, placement, release, and settling acceptance checks.

## Known implementation constraints

- The vendored YAM URDF does not provide authored collision meshes for all links; runtime USD collision generation and fingertip convex decomposition are part of the current simulation setup.
- Robot actuator gains, deformable material parameters, rigid/deformable contact parameters, and acceptance thresholds are task simulation parameters, not calibrated real-world measurements.
- The built-in articulation Jacobian was avoided in the coupled MJWarp/VBD beta path because earlier testing showed native instability; the project therefore keeps its own URDF-derived FK/Jacobian implementation.
- Object positions are obtained from simulation ground truth; perception noise and camera-based pose estimation are outside the current task.
- The controller is a scripted state machine with local DLS IK, not a general collision-aware motion planner.
- Passing the configured randomized episodes does not imply full-workspace or arbitrary-object grasp robustness.

## Documentation

Use only:

- `docs/yam_pick_ball_reference.md` for parameters, functions, variables, and math;
- `docs/yam_pick_ball_workflow.md` for run modes, state machine, validation, and debugging.

The source code is the final authority for current numeric parameter values.
