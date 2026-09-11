"""Load the YAM Ultra 2 articulation in Newton and execute a few position-control steps."""

from pathlib import Path

from isaaclab.app import AppLauncher


app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import torch
from isaaclab_newton.assets import Articulation
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationCfg, SimulationContext


PROJECT_ROOT = Path(__file__).resolve().parents[1]
YAM_USD = PROJECT_ROOT / "assets/generated/yam_ultra_2/yam_ultra/yam_ultra.usda"


def main() -> None:
    physics_cfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=40,
            nconmax=80,
            ls_iterations=20,
            integrator="implicitfast",
        ),
        num_substeps=2,
        use_cuda_graph=False,
    )
    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device="cuda:0", physics=physics_cfg))

    sim_utils.create_prim("/World/env_0", "Xform")
    robot_cfg = ArticulationCfg(
        prim_path="/World/env_0/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(YAM_USD),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(max_force=80.0, max_joint_velocity=3.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint1": 0.0,
                "joint2": 1.0,
                "joint3": 1.4,
                "joint4": -0.8,
                "joint5": 0.0,
                "joint6": 0.0,
                "joint7": -0.02,
                "joint8": -0.02,
            }
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]"],
                effort_limit_sim=80.0,
                velocity_limit_sim=3.0,
                stiffness=100.0,
                damping=10.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["joint[7-8]"],
                effort_limit_sim=40.0,
                velocity_limit_sim=1.0,
                stiffness=200.0,
                damping=20.0,
            ),
        },
    )
    robot = Articulation(robot_cfg)
    sim.reset()

    if robot.num_joints != 8:
        raise RuntimeError(f"Expected 8 joints, got {robot.num_joints}: {robot.joint_names}")

    target = robot.data.default_joint_pos.torch.clone()
    for _ in range(20):
        robot.set_joint_position_target_index(target=target)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim.get_physics_dt())

    if not torch.isfinite(robot.data.joint_pos.torch).all():
        raise RuntimeError("Non-finite joint state after Newton stepping")

    gripper_body_idx = robot.body_names.index("gripper")
    gripper_jacobian_idx = gripper_body_idx - 1 if robot.is_fixed_base else gripper_body_idx
    jacobian = robot.data.body_link_jacobian_w.torch[:, gripper_jacobian_idx, :, :6]
    if not torch.isfinite(jacobian).all():
        raise RuntimeError("Non-finite YAM gripper Jacobian")
    print(f"YAM_GRIPPER_JACOBIAN_SHAPE={tuple(jacobian.shape)}")

    print(f"YAM_JOINT_NAMES={robot.joint_names}")
    print(f"YAM_JOINT_LIMITS={robot.data.joint_pos_limits[0].cpu().tolist()}")
    print(f"YAM_JOINT_POS={robot.data.joint_pos[0].cpu().tolist()}")
    print("YAM_ULTRA_2_NEWTON_SMOKE_OK")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
