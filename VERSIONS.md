# Reproducible environment

Validated on 2026-09-11:

- Isaac Lab source tag: `v3.0.0-beta2.patch1`
- Isaac Sim: `6.0.1.0`
- Python: `3.12.14`
- PyTorch: `2.10.0+cu128`
- Newton: `1.2.1`
- Warp: `1.13.0`
- I2RT YAM Ultra 2 asset commit: `5b72c47239bd056d0fa6c1a39edeb0537c89443c`

Environment path:

`/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/env_isaaclab`

Validated checks:

- Isaac Sim/Kit launch and five simulation steps.
- Newton MJWarp CUDA solver initialization and one physics step.
- Newton VBD tetrahedral deformable initialization, nodal reset, CUDA graph capture, and physics stepping.
- YAM Ultra 2 URDF conversion to USD with a fixed base and collisions generated from visual meshes.
- YAM Ultra 2 eight-joint articulation loading and 20 position-control steps with Newton MJWarp.

Known limitations:

- The vendor URDF has no authored collision geometries; current collisions are generated from visual meshes and require visual/performance validation.
- Vendor URDF effort and velocity limits are placeholders; runtime limits in the smoke test are provisional.
- Arm/gripper gains require task-specific tuning before deformable grasp testing.


## 2026-09-11 task integration milestone

- Added a two-way coupled Newton scene using `CoupledMJWarpVBDSolverCfg`, MJWarp, and VBD.
- Added deterministic block and deformable-ball sampling plus complete rigid, nodal, and robot reset.
- Added project-local YAM URDF-derived PyTorch FK, geometric Jacobian, and DLS position IK.
- Verified three randomized reset episodes with 60 physics steps each.
- Verified one randomized pre-grasp run at 5.6 mm gripper error and zero printed FK to simulation alignment error.
- Confirmed a beta limitation: the built-in articulation Jacobian under the coupled MJWarp and VBD manager causes a native early exit, while the same Jacobian works under standalone MJWarp.
- Open issue: a second reset after controlled robot motion can trigger a native early exit. Multi-episode control evaluation remains pending safe coupled-state reset handling.
