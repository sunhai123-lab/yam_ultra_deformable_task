"""Newton integration smoke test for the YAM Ultra deformable-ball task.

This is the first executable milestone of the task.  It deliberately contains no
policy or task code copied from another robotics project: the scene, reset logic,
and checks are built directly with Isaac Lab APIs.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episodes", type=int, default=3, help="Number of randomized resets to exercise.")
parser.add_argument("--steps-per-episode", type=int, default=30, help="Physics steps after each reset.")
parser.add_argument("--seed", type=int, default=7, help="Seed for deterministic object randomization.")
parser.add_argument("--ik-smoke-steps", type=int, default=0, help="Move the gripper above the ball for this many steps.")
parser.add_argument("--keep-open", action="store_true", help="Keep the Kit GUI open after the checks finish.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab_newton.assets import Articulation, RigidObject
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from yam_ultra_deformable_place.yam_kinematics import (
    damped_least_squares_position_step,
    forward_kinematics_and_jacobian,
)
from isaaclab_contrib.deformable import CoupledMJWarpVBDSolverCfg, DeformableObject, VBDSolverCfg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
YAM_USD = PROJECT_ROOT / "assets/generated/yam_ultra_2/yam_ultra/yam_ultra.usda"

TABLE_SIZE = (1.10, 0.75, 0.08)
TABLE_CENTER = (0.12, 0.0, 0.40)
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_SIZE[2] / 2.0
BLOCK_SIZE = (0.07, 0.07, 0.05)
BALL_RADIUS = 0.04


def make_sim() -> SimulationContext:
    """Create a two-way coupled MJWarp rigid + VBD deformable Newton simulation."""
    solver_cfg = CoupledMJWarpVBDSolverCfg(
        rigid_solver_cfg=MJWarpSolverCfg(
            njmax=80,
            nconmax=240,
            ls_iterations=20,
            cone="pyramidal",
            integrator="implicitfast",
        ),
        soft_solver_cfg=VBDSolverCfg(
            iterations=5,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=False,
            particle_collision_detection_interval=-1,
        ),
        coupling_mode="two_way",
    )
    return SimulationContext(
        SimulationCfg(
            dt=1.0 / 120.0,
            device=args_cli.device,
            physics=NewtonCfg(solver_cfg=solver_cfg, num_substeps=4, use_cuda_graph=False),
        )
    )


def spawn_scene() -> tuple[Articulation, RigidObject, DeformableObject]:
    """Spawn the robot, static workspace, rigid target block, and soft ball."""
    if not YAM_USD.is_file():
        raise FileNotFoundError(f"Converted YAM asset is missing: {YAM_USD}")

    sim_utils.create_prim("/World/env_0", "Xform")
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    table_cfg = sim_utils.CuboidCfg(
        size=TABLE_SIZE,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.34, 0.22, 0.12)),
    )
    table_cfg.func("/World/env_0/Table", table_cfg, translation=TABLE_CENTER)

    robot_cfg = ArticulationCfg(
        prim_path="/World/env_0/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(YAM_USD),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(max_force=80.0, max_joint_velocity=3.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.22, 0.0, TABLE_TOP_Z),
            joint_pos={
                "joint1": 0.0,
                "joint2": 1.0,
                "joint3": 1.4,
                "joint4": -0.8,
                "joint5": 0.0,
                "joint6": 0.0,
                "joint7": -0.02,
                "joint8": -0.02,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]"],
                effort_limit_sim=120.0,
                velocity_limit_sim=3.0,
                stiffness=500.0,
                damping=40.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["joint[7-8]"],
                effort_limit_sim=60.0,
                velocity_limit_sim=1.0,
                stiffness=500.0,
                damping=30.0,
            ),
        },
    )
    robot = Articulation(robot_cfg)

    block = RigidObject(
        RigidObjectCfg(
            prim_path="/World/env_0/TargetBlock",
            spawn=sim_utils.CuboidCfg(
                size=BLOCK_SIZE,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.12),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.35, 0.9)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.20, 0.16, TABLE_TOP_Z + BLOCK_SIZE[2] / 2.0)),
        )
    )

    youngs_modulus = 5.0e4
    poissons_ratio = 0.35
    ball = DeformableObject(
        DeformableObjectCfg(
            prim_path="/World/env_0/DeformableBall",
            spawn=sim_utils.MeshSphereCfg(
                radius=BALL_RADIUS,
                deformable_props=NewtonDeformableBodyPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.18, 0.12)),
                physics_material=NewtonDeformableBodyMaterialCfg(
                    density=500.0,
                    k_mu=youngs_modulus / (2.0 * (1.0 + poissons_ratio)),
                    k_lambda=(
                        youngs_modulus
                        * poissons_ratio
                        / ((1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio))
                    ),
                    particle_radius=0.006,
                ),
            ),
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.15, -0.16, TABLE_TOP_Z + BALL_RADIUS + 0.01)),
        )
    )
    return robot, block, ball


def sample_object_positions(rng: random.Random) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Sample separated start positions from two reachable tabletop regions."""
    block_pos = (
        rng.uniform(0.02, 0.16),
        rng.uniform(0.08, 0.18),
        TABLE_TOP_Z + BLOCK_SIZE[2] / 2.0 + 0.004,
    )
    ball_pos = (
        rng.uniform(0.02, 0.16),
        rng.uniform(-0.18, -0.08),
        TABLE_TOP_Z + BALL_RADIUS + 0.012,
    )
    planar_distance = ((block_pos[0] - ball_pos[0]) ** 2 + (block_pos[1] - ball_pos[1]) ** 2) ** 0.5
    if planar_distance < 0.13:
        raise RuntimeError(f"Sampling regions unexpectedly overlap: distance={planar_distance:.3f} m")
    return block_pos, ball_pos


def reset_episode(
    robot: Articulation,
    block: RigidObject,
    ball: DeformableObject,
    block_pos: tuple[float, float, float],
    ball_pos: tuple[float, float, float],
) -> None:
    """Reset every dynamic state, including all deformable nodes."""
    joint_pos = robot.data.default_joint_pos.torch.clone()
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel)
    robot.set_joint_position_target_index(target=joint_pos)

    block_pose = block.data.default_root_pose.torch.clone()
    block_pose[0, :3] = torch.tensor(block_pos, device=block_pose.device)
    block_pose[0, 3:] = torch.tensor((0.0, 0.0, 0.0, 1.0), device=block_pose.device)
    block.write_root_pose_to_sim_index(root_pose=block_pose)
    block.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=block_pose.device))

    nodal_state = ball.data.default_nodal_state_w.torch.clone()
    current_center = nodal_state[..., :3].mean(dim=1)
    desired_center = torch.tensor(ball_pos, device=nodal_state.device).unsqueeze(0)
    nodal_state[..., :3] += (desired_center - current_center).unsqueeze(1)
    nodal_state[..., 3:] = 0.0
    ball.write_nodal_state_to_sim_index(nodal_state)

    kinematic_target = ball.data.nodal_kinematic_target.torch.clone()
    kinematic_target[..., :3] = nodal_state[..., :3]
    kinematic_target[..., 3] = 1.0
    ball.write_nodal_kinematic_target_to_sim_index(kinematic_target)

    robot.reset()
    block.reset()
    ball.reset()



def move_gripper_above_ball(
    sim: SimulationContext,
    robot: Articulation,
    block: RigidObject,
    ball: DeformableObject,
    ball_pos: tuple[float, float, float],
    num_steps: int,
) -> tuple[float, float]:
    """Move to a pre-grasp waypoint using the local URDF-derived DLS Jacobian."""
    ee_body_idx = robot.body_names.index("gripper")
    target_pos_w = torch.tensor(
        ((ball_pos[0], ball_pos[1], ball_pos[2] + 0.14),), device=robot.device, dtype=torch.float32
    )

    robot.set_joint_position_target_index(target=robot.data.joint_pos.torch.clone())
    robot.write_data_to_sim()
    sim.step(render=not args_cli.headless)
    robot.update(sim.get_physics_dt())
    block.update(sim.get_physics_dt())
    ball.update(sim.get_physics_dt())

    limits = robot.data.joint_pos_limits.torch[0, :6]
    initial_fk_error = 0.0
    for step in range(num_steps):
        root_pose_w = robot.data.root_link_pose_w.torch
        arm_pos = robot.data.joint_pos.torch[:, :6]
        fk_pos_w, _, geometric_jacobian_w = forward_kinematics_and_jacobian(root_pose_w, arm_pos)
        if step == 0:
            simulated_pos_w = robot.data.body_link_pose_w.torch[:, ee_body_idx, :3]
            initial_fk_error = torch.linalg.vector_norm(fk_pos_w - simulated_pos_w, dim=-1).item()

        joint_delta = damped_least_squares_position_step(
            fk_pos_w, target_pos_w, geometric_jacobian_w[:, :3, :], damping=0.05
        )
        arm_target = arm_pos + torch.clamp(joint_delta, min=-0.025, max=0.025)
        arm_target = torch.clamp(arm_target, min=limits[:, 0], max=limits[:, 1])

        full_target = robot.data.joint_pos.torch.clone()
        full_target[:, :6] = arm_target
        full_target[:, 6:] = -0.04
        robot.set_joint_position_target_index(target=full_target)
        robot.write_data_to_sim()
        sim.step(render=not args_cli.headless)
        robot.update(sim.get_physics_dt())
        block.update(sim.get_physics_dt())
        ball.update(sim.get_physics_dt())

    final_pos = robot.data.body_link_pose_w.torch[:, ee_body_idx, :3]
    final_error = torch.linalg.vector_norm(final_pos - target_pos_w, dim=-1).item()
    return final_error, initial_fk_error



def main() -> None:
    if args_cli.episodes < 1 or args_cli.steps_per_episode < 1:
        raise ValueError("--episodes and --steps-per-episode must both be positive")

    sim = make_sim()
    sim.set_camera_view(eye=(1.25, -1.20, 1.15), target=(0.12, 0.0, TABLE_TOP_Z))
    robot, block, ball = spawn_scene()
    sim.reset()

    if robot.num_joints != 8:
        raise RuntimeError(f"Expected 8 YAM joints, got {robot.num_joints}: {robot.joint_names}")

    rng = random.Random(args_cli.seed)
    results: list[dict[str, object]] = []
    ik_errors: list[float] = []
    fk_errors: list[float] = []
    for episode in range(args_cli.episodes):
        block_pos, ball_pos = sample_object_positions(rng)
        reset_episode(robot, block, ball, block_pos, ball_pos)

        if args_cli.ik_smoke_steps > 0:
            ik_error, fk_error = move_gripper_above_ball(
                sim, robot, block, ball, ball_pos, args_cli.ik_smoke_steps
            )
            ik_errors.append(ik_error)
            fk_errors.append(fk_error)

        for _ in range(args_cli.steps_per_episode):
            robot.set_joint_position_target_index(target=robot.data.default_joint_pos.torch)
            robot.write_data_to_sim()
            sim.step(render=not args_cli.headless)
            robot.update(sim.get_physics_dt())
            block.update(sim.get_physics_dt())
            ball.update(sim.get_physics_dt())

        tensors = (robot.data.joint_pos.torch, block.data.root_pos_w.torch, ball.data.nodal_pos_w.torch)
        if not all(torch.isfinite(value).all() for value in tensors):
            raise RuntimeError(f"Non-finite simulation state in episode {episode}")

        measured_block = block.data.root_pos_w.torch[0].cpu().tolist()
        measured_ball = ball.data.nodal_pos_w.torch[0].mean(dim=0).cpu().tolist()
        results.append(
            {
                "episode": episode,
                "sampled_block": [round(v, 4) for v in block_pos],
                "sampled_ball": [round(v, 4) for v in ball_pos],
                "measured_block": [round(v, 4) for v in measured_block],
                "measured_ball_center": [round(v, 4) for v in measured_ball],
            }
        )

    print(f"NEWTON_SOLVER={type(sim.cfg.physics.solver_cfg).__name__}")
    print(f"YAM_JOINT_NAMES={robot.joint_names}")
    print(f"YAM_BODY_NAMES={robot.body_names}")
    gripper_index = robot.body_names.index("gripper")
    print(f"YAM_GRIPPER_POSE={robot.data.body_link_pose_w.torch[0, gripper_index].cpu().tolist()}")
    print("YAM_KINEMATICS=project URDF-derived Torch FK/Jacobian")
    for result in results:
        print(f"RESET_RESULT={result}")
    for episode, error in enumerate(ik_errors):
        print(f"IK_PREGRASP_ERROR episode={episode} error_m={error:.6f}")
        print(f"URDF_FK_ALIGNMENT_ERROR episode={episode} error_m={fk_errors[episode]:.6f}")
    if fk_errors and max(fk_errors) > 0.002:
        raise RuntimeError(f"URDF FK does not align with simulated gripper within 2 mm: {fk_errors}")
    if ik_errors and max(ik_errors) > 0.06:
        raise RuntimeError(f"IK pre-grasp did not converge within 6 cm: {ik_errors}")
    print("YAM_DEFORMABLE_TASK_INTEGRATION_OK")

    if args_cli.keep_open:
        if args_cli.headless:
            raise ValueError("--keep-open requires the Kit visualizer; use --visualizer kit.")
        print("GUI_READY_CLOSE_WINDOW_TO_EXIT", flush=True)
        hold_target = robot.data.joint_pos.torch.clone()
        while simulation_app.is_running():
            robot.set_joint_position_target_index(target=hold_target)
            robot.write_data_to_sim()
            sim.step(render=True)
            robot.update(sim.get_physics_dt())
            block.update(sim.get_physics_dt())
            ball.update(sim.get_physics_dt())


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
