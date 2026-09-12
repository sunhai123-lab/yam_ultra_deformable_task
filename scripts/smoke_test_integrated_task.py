"""YAM Ultra 抓取并抬升可变形球的 Isaac Lab/Newton 集成测试。

脚本直接使用 Isaac Lab API 构建桌面、刚体方块、VBD 可变形球和 YAM Ultra
机械臂，不依赖其他机器人任务或策略代码。任务控制不是强化学习策略，而是一个
确定性的硬编码流程：

1. 在桌面可达区域随机化方块和球的位置；
2. 使用项目内基于 URDF 的正运动学、雅可比矩阵和 DLS 逆运动学移动夹爪；
3. 夹爪依次执行预抓取、下降、闭合和抬升；
4. 使用简化盒形夹指碰撞垫与软球产生法向接触和 Coulomb 摩擦；
5. 通过有限值、FK 对齐误差和球心实际抬升量判断测试是否成功。

原始 YAM USD 的碰撞体由高面数视觉网格自动生成；当前 Isaac Lab/Newton beta 在两个
复杂凸包同时接触软体时不稳定。因此保留原始可视网格，但关闭其碰撞，改用固定在左右
夹指 link 下的简单盒形碰撞垫。球体的全部节点始终是自由动力学节点，不使用节点绑定。
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from isaaclab.app import AppLauncher

# 命令行参数在 AppLauncher 参数之前定义；AppLauncher 随后补充 --device、--headless、
# --visualizer 等 Isaac Lab 通用参数。
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episodes", type=int, default=10, help="随机复位并测试的 episode 数量。")
parser.add_argument("--steps-per-episode", type=int, default=30, help="每次复位后用于稳定场景的物理步数。")
parser.add_argument("--seed", type=int, default=7, help="物体位置随机化使用的可复现随机种子。")
parser.add_argument("--pick-lift", action="store_true", help="执行硬编码抓取和抬升流程。")
parser.add_argument("--motion-steps", type=int, default=240, help="每个机械臂运动阶段使用的物理步数。")
parser.add_argument("--lift-height", type=float, default=0.05, help="球心期望沿世界 Z 轴抬升的距离，单位 m。")
parser.add_argument("--keep-open", action="store_true", help="测试结束后保持 Kit GUI，直到手动关闭窗口。")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 必须先启动 SimulationApp，再导入依赖 Kit/Omniverse 运行时的 Isaac Lab 模块。
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
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply

# 导入 YAM Ultra 机器人运动学工具
from yam_ultra_deformable_place.yam_kinematics import (
    damped_least_squares_position_step,
    damped_least_squares_pose_step,
    forward_kinematics_and_jacobian,
)
from isaaclab_contrib.deformable import CoupledMJWarpVBDSolverCfg, DeformableObject, VBDSolverCfg
from isaaclab_contrib.deformable.newton_manager_cfg import NewtonModelCfg

# ------------------------------ 场景与控制常量 ------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 由 scripts/convert_yam_ultra_2.py 从项目内 URDF 转换得到的运行时 USD。
YAM_USD = PROJECT_ROOT / "assets/generated/yam_ultra_2/yam_ultra/yam_ultra.usda"

# 所有位置和尺寸使用 SI 单位：长度 m、角度 rad、时间 s。
TABLE_SIZE = (1.10, 0.75, 0.08)
TABLE_CENTER = (0.12, 0.0, 0.40)
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_SIZE[2] / 2.0
BLOCK_SIZE = (0.07, 0.07, 0.05)
BALL_RADIUS = 0.04
BALL_DENSITY = 500.0
BALL_YOUNGS_MODULUS = 5.0e4
BALL_POISSONS_RATIO = 0.35
# 受控抓取点位于 gripper link 坐标系的 -Z 方向 0.10 m，即两根夹指之间。
GRASP_POINT_LOCAL = (0.0, 0.0, -0.10)
# 预抓取/抓取阶段的世界系末端朝向：夹爪从球体正上方向下接近。
GRASP_ROTATION_W = ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
# 简化碰撞垫只替代原高面数碰撞凸包；它们固定在真实 tip_left/tip_right link 上。
# 夹持轴方向厚度仅 4 mm，使最大张开时的净间隙约 85 mm，大于离散软球约
# 81--82 mm 的实际宽度；这样下降阶段不会在闭合命令之前擦碰并推走球。
GRIPPER_PAD_SIZE = (0.040, 0.004, 0.065)
# USD 转换器把 tip prim 的原点烘焙到了夹指网格坐标系，因此下面是 ContactPad
# 相对“转换后 tip prim”的坐标，不能直接填写 URDF link 坐标。它们经 tip
# 的固定变换后，在夹爪坐标系中分别位于 (0, -0.0445119, -0.10) 和
# (0, +0.0445119, -0.10)；即张开时位于球的两侧，而不是球上方。
LEFT_PAD_LOCAL_POS = (0.0140000, 0.0914614, -0.1730911)
RIGHT_PAD_LOCAL_POS = (0.0140000, 0.0013375, -0.1731001)
LEFT_TIP_PATH = "/World/env_0/Robot/Geometry/base/link1/link2/link3/link4/link5/gripper/tip_left"
RIGHT_TIP_PATH = "/World/env_0/Robot/Geometry/base/link1/link2/link3/link4/link5/gripper/tip_right"

# joint7、joint8 的 URDF 范围为 [-0.04695, 0]；0 表示最大张开，负值向球体闭合。
GRIPPER_OPEN_POSITION = 0.0
# q=0 时两夹指关节中心相距 2*0.0445119 m；减去两碰撞垫的总半厚度得到净开口。
GRIPPER_OPEN_GAP = 2.0 * 0.0445119 - GRIPPER_PAD_SIZE[1]
# 用简化线弹性关系 epsilon=sigma/E_eff 决定压缩率。8 kPa 对当前材料约为 14% 压缩；
# 该应力量级相对 50 kPa 杨氏模量仍处于小变形估算范围。
NOMINAL_GRIP_STRESS = 8.0e3
BALL_EFFECTIVE_MODULUS = BALL_YOUNGS_MODULUS / (1.0 - BALL_POISSONS_RATIO**2)
BALL_COMPRESSION_RATIO = min(0.18, NOMINAL_GRIP_STRESS / BALL_EFFECTIVE_MODULUS)
GRIPPER_TARGET_GAP = 2.0 * BALL_RADIUS * (1.0 - BALL_COMPRESSION_RATIO)
# 两根对称夹指各承担一半开口变化；负号来自 YAM prismatic joint 的闭合方向。
GRIPPER_GRASP_POSITION = -0.5 * (GRIPPER_OPEN_GAP - GRIPPER_TARGET_GAP)
# 可变形球在夹持中存在压缩和微滑移；第一里程碑要求球心实际位移达到末端命令的
# 70%，并以真实节点位移而非夹爪命令作为判据。该阈值用于判定“已抓离桌面”，
# 不代表后续 0.4 m 运输阶段的位置精度标准。
MIN_LIFT_RATIO = 0.70
# 该初值位于“夹爪竖直向下、肘部远离桌面”的可达 IK 分支。它不是固定动作结果；
# 每个 episode 仍会根据随机球坐标重新迭代求解关节角。
APPROACH_IK_SEED = (-0.48, 1.97, 1.63, -1.23, 0.0, -0.48)


@configclass
class DeformableNewtonCfg(NewtonCfg):
    """为耦合管理器补充软体接触模型参数。"""

    model_cfg: NewtonModelCfg | None = None


def make_sim() -> SimulationContext:
    """创建 MJWarp 刚体/关节系统与 VBD 软体双向耦合的 Newton 仿真。

    MJWarp 负责桌子、方块和机械臂，VBD 负责四面体软球；``two_way`` 表示两类
    求解器交换作用力。每个 1/120 s 控制步包含 10 个 Newton 子步，以提高接触稳定性。
    """
    solver_cfg = CoupledMJWarpVBDSolverCfg(
        rigid_solver_cfg=MJWarpSolverCfg(
            # 预分配关节和接触容量，避免复杂 articulation 超出缓冲区。
            njmax=256,
            nconmax=2048,
            # 刚体约束线性求解器参数。
            ls_iterations=20,
            ls_parallel=False,
            impratio=1,
            cone="pyramidal",
            integrator="implicitfast",
            ccd_iterations=100,
        ),
        soft_solver_cfg=VBDSolverCfg(
            # 增加软体迭代次数，以解析闭合夹爪产生的双侧接触和弹性压缩。
            iterations=10,
            integrate_with_external_rigid_solver=True,
            # 球体形变较小，关闭昂贵的粒子自碰撞。
            particle_enable_self_contact=False,
            particle_collision_detection_interval=-1,
        ),
        coupling_mode="two_way",
    )
    return SimulationContext(
        SimulationCfg(
            dt=1.0 / 120.0,
            device=args_cli.device,
            physics=DeformableNewtonCfg(
                solver_cfg=solver_cfg,
                model_cfg=NewtonModelCfg(
                    soft_contact_ke=1.0e4,
                    soft_contact_kd=1.0e-5,
                    soft_contact_mu=5.0,
                    shape_material_ke=4.0e4,
                    shape_material_kd=1.0e-5,
                    shape_material_mu=5.0,
                ),
                num_substeps=10,
                use_cuda_graph=False,
            ),
        )
    )


def spawn_scene() -> tuple[Articulation, RigidObject, DeformableObject]:
    """生成机械臂、静态工作台、刚体目标方块和 VBD 可变形球。

    Returns:
        ``(robot, block, ball)``：Newton articulation、刚体对象和软体对象。
    """
    if not YAM_USD.is_file():
        raise FileNotFoundError(f"Converted YAM asset is missing: {YAM_USD}")

    # /World/env_0 是当前单环境的根节点。
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

    # 前 6 个关节属于机械臂，joint7/8 对称驱动左右夹指。
    robot_cfg = ArticulationCfg(
        prim_path="/World/env_0/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(YAM_USD),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(max_force=80.0, max_joint_velocity=3.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.22, 0.0, TABLE_TOP_Z),
            joint_pos={
                # 初始构型使末端悬在桌面上方，避免启动时穿透工作台。
                "joint1": 0.0,
                "joint2": 1.0,
                "joint3": 1.4,
                "joint4": -0.8,
                "joint5": 0.0,
                "joint6": 0.0,
                # 启动时保持最大开口，保证动作顺序确实是“先张开，再下降，再闭合”。
                "joint7": GRIPPER_OPEN_POSITION,
                "joint8": GRIPPER_OPEN_POSITION,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]"],
                effort_limit_sim=120.0,
                velocity_limit_sim=3.0,
                # 提高负载下的位置保持能力，避免抓起软球后末端相对命令高度下垂。
                stiffness=3000.0,
                damping=100.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["joint[7-8]"],
                # 夹爪仍由关节驱动和接触反力决定运动，并非直接改写关节位置。
                # 6 kN/m 刚度使实际间隙能在软球反力下收敛到材料模型给出的目标，
                # 而 150 N·s/m 阻尼抑制接触瞬间的回弹。
                effort_limit_sim=80.0,
                velocity_limit_sim=0.5,
                stiffness=6000.0,
                damping=150.0,
            ),
        },
    )
    robot = Articulation(robot_cfg)
    # 导入 USD 使用实例化 prim；解除实例化后才能递归修改碰撞并添加简化碰撞垫。
    sim_utils.make_uninstanceable("/World/env_0/Robot")
    # 关闭视觉网格自动生成的所有高面数凸包碰撞体。
    sim_utils.modify_collision_properties(
        "/World/env_0/Robot",
        sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    )
    # 只在真实左右夹指 link 下添加简单盒形碰撞垫。它们随 prismatic joint 运动，
    # 接触力和摩擦由 Newton 求解，不会直接修改或绑定软球节点。
    pad_cfg = sim_utils.CuboidCfg(
        size=GRIPPER_PAD_SIZE,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15)),
    )
    pad_cfg.func(f"{LEFT_TIP_PATH}/ContactPad", pad_cfg, translation=LEFT_PAD_LOCAL_POS)
    pad_cfg.func(f"{RIGHT_TIP_PATH}/ContactPad", pad_cfg, translation=RIGHT_PAD_LOCAL_POS)

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
    ball = DeformableObject(
        DeformableObjectCfg(
            prim_path="/World/env_0/DeformableBall",
            spawn=sim_utils.MeshSphereCfg(
                radius=BALL_RADIUS,
                deformable_props=NewtonDeformableBodyPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.18, 0.12)),
                physics_material=NewtonDeformableBodyMaterialCfg(
                    # density 单位 kg/m^3；k_mu、k_lambda 是由 E、ν 换算的 Lamé 参数。
                    # particle_radius 影响粒子接触范围，不等于可视球半径。
                    density=BALL_DENSITY,
                    k_mu=BALL_YOUNGS_MODULUS / (2.0 * (1.0 + BALL_POISSONS_RATIO)),
                    k_lambda=(
                        BALL_YOUNGS_MODULUS
                        * BALL_POISSONS_RATIO
                        / ((1.0 + BALL_POISSONS_RATIO) * (1.0 - 2.0 * BALL_POISSONS_RATIO))
                    ),
                    particle_radius=0.006,
                ),
            ),
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.15, -0.16, TABLE_TOP_Z + BALL_RADIUS + 0.01)),
        )
    )
    return robot, block, ball


def sample_object_positions(rng: random.Random) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """在两个互相分离、机械臂可达的桌面区域中采样方块和球的位置。

    方块位于世界 Y 正半区，球位于 Y 负半区。Z 坐标由桌面高度、物体半高/半径
    和安全间隙组成，避免复位时物体与桌面深度穿透。
    """
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
    """复位机器人、方块以及软球的全部动力学状态。

    软体没有可直接代表完整状态的单一 root pose，其状态由所有四面体节点共同决定。
    因此需要整体平移默认节点、清零节点速度，并重设每个节点的运动学标志。

    Args:
        robot: 8 自由度 YAM articulation。
        block: 目标刚体方块。
        ball: Newton/VBD 可变形球。
        block_pos: 方块世界坐标 ``(x, y, z)``，单位 m。
        ball_pos: 期望软球节点平均位置 ``(x, y, z)``，单位 m。
        球的所有节点始终设置为自由节点，抓取只依靠刚体—软体接触。
    """
    # articulation 状态形状为 (num_envs=1, num_joints=8)。
    joint_pos = robot.data.default_joint_pos.torch.clone()
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel)
    robot.set_joint_position_target_index(target=joint_pos)

    # 刚体 pose 顺序为 (x, y, z, qx, qy, qz, qw)。
    block_pose = block.data.default_root_pose.torch.clone()
    block_pose[0, :3] = torch.tensor(block_pos, device=block_pose.device)
    block_pose[0, 3:] = torch.tensor((0.0, 0.0, 0.0, 1.0), device=block_pose.device)
    block.write_root_pose_to_sim_index(root_pose=block_pose)
    block.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=block_pose.device))

    # nodal_state: (1, num_nodes, 6)，最后一维为 [位置 xyz, 速度 xyz]。
    nodal_state = ball.data.default_nodal_state_w.torch.clone()
    current_center = nodal_state[..., :3].mean(dim=1)
    desired_center = torch.tensor(ball_pos, device=nodal_state.device).unsqueeze(0)
    nodal_state[..., :3] += (desired_center - current_center).unsqueeze(1)
    nodal_state[..., 3:] = 0.0
    ball.write_nodal_state_to_sim_index(nodal_state)

    # kinematic_target: (1, num_nodes, 4)，最后一维为 [目标 xyz, free_flag]。
    kinematic_target = ball.data.nodal_kinematic_target.torch.clone()
    kinematic_target[..., :3] = nodal_state[..., :3]
    # Newton 中 flag=1 表示自由动力学节点；任务不使用 flag=0 的运动学抓取节点。
    kinematic_target[..., 3] = 1.0
    ball.write_nodal_kinematic_target_to_sim_index(kinematic_target)

    robot.reset()
    block.reset()
    ball.reset()


def step_scene(
    sim: SimulationContext,
    robot: Articulation,
    block: RigidObject,
    ball: DeformableObject,
    joint_target: torch.Tensor,
) -> None:
    """提交控制量、推进一个物理步，并刷新所有对象的数据缓存。

    调用顺序很重要：先设置机器人与软体目标，再 ``sim.step``，最后调用各对象的
    ``update`` 把求解结果读回张量。``joint_target`` 形状为 ``(1, 8)``。
    """
    robot.set_joint_position_target_index(target=joint_target)
    robot.write_data_to_sim()
    # 目标缓存中的 free_flag 全为 1，此调用不会把任何节点固定到夹爪。
    ball.write_data_to_sim()
    sim.step(render=not args_cli.headless)
    dt = sim.get_physics_dt()
    robot.update(dt)
    block.update(dt)
    ball.update(dt)


def ball_center(ball: DeformableObject) -> torch.Tensor:
    """返回软球所有仿真节点的世界坐标平均值，形状为 ``(1, 3)``。"""
    return ball.data.nodal_pos_w.torch.mean(dim=1)


def gripper_pad_centers(robot: Articulation) -> torch.Tensor:
    """根据左右 tip link 位姿计算两个简化碰撞垫的世界坐标中心。"""
    tip_indices = [robot.body_names.index("tip_left"), robot.body_names.index("tip_right")]
    tip_poses_w = robot.data.body_link_pose_w.torch[0, tip_indices]
    local_centers = torch.tensor(
        (LEFT_PAD_LOCAL_POS, RIGHT_PAD_LOCAL_POS),
        device=robot.device,
        dtype=tip_poses_w.dtype,
    )
    return tip_poses_w[:, :3] + quat_apply(tip_poses_w[:, 3:], local_centers)


def grasp_geometry_metrics(robot: Articulation, ball: DeformableObject) -> tuple[float, float]:
    """返回夹持轴上的实际夹爪净间隙和软球宽度，单位 m。

    夹持轴由两个碰撞垫中心连线定义，所以即使末端存在少量姿态误差，
    也不会错把世界 Y 轴宽度当成球的受压宽度。
    """
    pad_centers = gripper_pad_centers(robot)
    center_delta = pad_centers[1] - pad_centers[0]
    center_distance = torch.linalg.vector_norm(center_delta)
    grasp_axis = center_delta / center_distance.clamp_min(1.0e-8)
    node_projections = ball.data.nodal_pos_w.torch[0] @ grasp_axis
    ball_width = node_projections.max() - node_projections.min()
    # GRIPPER_PAD_SIZE[1] 是两个对向盒形垫沿夹持轴的单个厚度。
    surface_gap = center_distance - GRIPPER_PAD_SIZE[1]
    return surface_gap.item(), ball_width.item()


def grasp_point_kinematics(
    root_pose_w: torch.Tensor,
    arm_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """把 gripper link 的运动学结果转换到实际受控抓取点。

    Args:
        root_pose_w: 基座世界位姿，形状 ``(N, 7)``。
        arm_pos: joint1..joint6 的角度，形状 ``(N, 6)``。

    Returns:
        gripper link 位置 ``(N,3)``、旋转矩阵 ``(N,3,3)``、抓取点位置
        ``(N,3)``、抓取点 6×6 几何雅可比 ``(N,6,6)``。
    """
    gripper_pos_w, gripper_rot_w, geometric_jacobian_w = forward_kinematics_and_jacobian(root_pose_w, arm_pos)
    local_offset = torch.tensor(GRASP_POINT_LOCAL, device=arm_pos.device, dtype=arm_pos.dtype)
    offset_w = (gripper_rot_w @ local_offset.expand(arm_pos.shape[0], 3).unsqueeze(-1)).squeeze(-1)
    grasp_pos_w = gripper_pos_w + offset_w

    # 刚体上偏置点的线速度满足 v_point = v_origin + ω × r，因此需要修正线性雅可比。
    angular_columns = geometric_jacobian_w[:, 3:, :].transpose(1, 2)
    point_velocity_columns = torch.linalg.cross(
        angular_columns,
        offset_w.unsqueeze(1).expand_as(angular_columns),
        dim=-1,
    )
    point_jacobian_w = geometric_jacobian_w[:, :3, :] + point_velocity_columns.transpose(1, 2)
    grasp_geometric_jacobian_w = torch.cat((point_jacobian_w, geometric_jacobian_w[:, 3:, :]), dim=1)
    return gripper_pos_w, gripper_rot_w, grasp_pos_w, grasp_geometric_jacobian_w


def move_grasp_point(
    sim: SimulationContext,
    robot: Articulation,
    block: RigidObject,
    ball: DeformableObject,
    target_pos_w: torch.Tensor,
    gripper_target: float,
    num_steps: int,
    maintain_orientation: bool = True,
    target_rotation_w: torch.Tensor | None = None,
) -> tuple[float, float]:
    """每个物理步用 DLS 逆运动学把夹指中心移动到世界坐标目标。

    ``maintain_orientation=True`` 时同时控制 3 维位置和 3 维姿态；为 False 时仅控制
    位置，用于高位目标释放多余姿态约束。返回最终抓取点误差及项目 FK 与仿真
    gripper link 的初始对齐误差，单位均为 m。
    """
    ee_body_idx = robot.body_names.index("gripper")
    limits = robot.data.joint_pos_limits.torch[0, :6]
    initial_fk_error = 0.0
    if target_rotation_w is None:
        target_rot_w = torch.tensor(
            GRASP_ROTATION_W, device=robot.device, dtype=robot.data.joint_pos.torch.dtype
        ).unsqueeze(0)
    else:
        target_rot_w = target_rotation_w

    for step in range(num_steps):
        root_pose_w = robot.data.root_link_pose_w.torch
        arm_pos = robot.data.joint_pos.torch[:, :6]
        gripper_pos_w, gripper_rot_w, grasp_pos_w, grasp_geometric_jacobian_w = grasp_point_kinematics(
            root_pose_w, arm_pos
        )
        if step == 0:
            simulated_pos_w = robot.data.body_link_pose_w.torch[:, ee_body_idx, :3]
            initial_fk_error = torch.linalg.vector_norm(gripper_pos_w - simulated_pos_w, dim=-1).item()

        # DLS: Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ e。阻尼可避免奇异点附近数值爆炸。
        if maintain_orientation:
            joint_delta = damped_least_squares_pose_step(
                grasp_pos_w,
                target_pos_w,
                gripper_rot_w,
                target_rot_w,
                grasp_geometric_jacobian_w,
                damping=0.05,
            )
        else:
            joint_delta = damped_least_squares_position_step(
                grasp_pos_w,
                target_pos_w,
                grasp_geometric_jacobian_w[:, :3, :],
                damping=0.05,
            )
        # 每个仿真步最多改变 0.02 rad，再限制到 URDF 给出的关节上下界。
        arm_target = arm_pos + torch.clamp(joint_delta, min=-0.02, max=0.02)
        arm_target = torch.clamp(arm_target, min=limits[:, 0], max=limits[:, 1])

        full_target = robot.data.joint_pos.torch.clone()
        full_target[:, :6] = arm_target
        full_target[:, 6:] = gripper_target
        step_scene(sim, robot, block, ball, full_target)

    _, _, final_grasp_pos_w, _ = grasp_point_kinematics(
        robot.data.root_link_pose_w.torch,
        robot.data.joint_pos.torch[:, :6],
    )
    final_error = torch.linalg.vector_norm(final_grasp_pos_w - target_pos_w, dim=-1).item()
    return final_error, initial_fk_error


def solve_approach_joint_target(
    robot: Articulation,
    target_pos_w: torch.Tensor,
    target_rotation_w: torch.Tensor,
    iterations: int = 300,
) -> tuple[torch.Tensor, float, float]:
    """从安全的向下抓取分支求解随机预抓取位姿，不推进物理仿真。"""
    arm_target = torch.tensor(
        (APPROACH_IK_SEED,),
        device=robot.device,
        dtype=robot.data.joint_pos.torch.dtype,
    )
    limits = robot.data.joint_pos_limits.torch[0, :6]
    root_pose_w = robot.data.root_link_pose_w.torch

    for _ in range(iterations):
        _, current_rotation_w, _ = forward_kinematics_and_jacobian(
            root_pose_w,
            arm_target,
        )
        # 抓取点相对 gripper link 有固定偏移，因此沿用与在线控制一致的点雅可比。
        _, _, grasp_pos_w, grasp_jacobian_w = grasp_point_kinematics(root_pose_w, arm_target)
        joint_delta = damped_least_squares_pose_step(
            grasp_pos_w,
            target_pos_w,
            current_rotation_w,
            target_rotation_w,
            grasp_jacobian_w,
            damping=0.03,
        )
        arm_target = torch.clamp(
            arm_target + torch.clamp(joint_delta, min=-0.05, max=0.05),
            min=limits[:, 0],
            max=limits[:, 1],
        )

    _, solved_rotation_w, solved_pos_w, _ = grasp_point_kinematics(root_pose_w, arm_target)
    position_error = torch.linalg.vector_norm(solved_pos_w - target_pos_w, dim=-1).item()
    rotation_error = torch.linalg.matrix_norm(solved_rotation_w - target_rotation_w, dim=(-2, -1)).item()
    return arm_target, position_error, rotation_error


def execute_arm_trajectory(
    sim: SimulationContext,
    robot: Articulation,
    block: RigidObject,
    ball: DeformableObject,
    target_arm_pos: torch.Tensor,
    gripper_target: float,
    num_steps: int,
) -> None:
    """用零起止速度的三次插值平滑执行关节轨迹。"""
    start_arm_pos = robot.data.joint_pos.torch[:, :6].clone()
    for step in range(num_steps):
        phase = (step + 1) / num_steps
        alpha = phase * phase * (3.0 - 2.0 * phase)
        full_target = robot.data.joint_pos.torch.clone()
        full_target[:, :6] = start_arm_pos + alpha * (target_arm_pos - start_arm_pos)
        full_target[:, 6:] = gripper_target
        step_scene(sim, robot, block, ball, full_target)


def hold_grasp(
    sim: SimulationContext,
    robot: Articulation,
    block: RigidObject,
    ball: DeformableObject,
    target_pos_w: torch.Tensor,
    gripper_start: float,
    gripper_end: float,
    num_steps: int,
    target_rotation_w: torch.Tensor,
) -> None:
    """保持抓取点位姿，同时把两个夹指从 ``gripper_start`` 线性闭合到终值。"""
    for step in range(num_steps):
        alpha = (step + 1) / num_steps
        gripper_target = gripper_start + alpha * (gripper_end - gripper_start)
        move_grasp_point(
            sim,
            robot,
            block,
            ball,
            target_pos_w,
            gripper_target,
            num_steps=1,
            target_rotation_w=target_rotation_w,
        )


def pick_and_lift_ball(
    sim: SimulationContext,
    robot: Articulation,
    block: RigidObject,
    ball: DeformableObject,
    initial_ball_pos: tuple[float, float, float],
    motion_steps: int,
    lift_height: float,
) -> dict[str, float]:
    """执行“预抓取→下降→物理闭合→接触稳定→抬升→保持”的硬编码状态序列。

    Args:
        initial_ball_pos: 稳定阶段结束后测得的球心世界位置。
        motion_steps: 每个主要机械臂移动阶段使用的物理步数。
        lift_height: 期望沿世界 Z 轴抬升的距离，单位 m。

    Returns:
        包含各阶段位置误差、FK 对齐误差、命令抬升量和实际球心抬升量的字典。
    """
    device = robot.device
    dtype = robot.data.joint_pos.torch.dtype
    # 所有目标保留 batch 维度，形状为 (1, 3)。先用稳定阶段测得的球心到达球上方；
    # 到位后会再读取一次球心，补偿软球在桌面上的少量滚动。
    initial_ball_target = torch.tensor((initial_ball_pos,), device=device, dtype=dtype)
    pregrasp_height = 0.10
    pregrasp_offset = torch.tensor(((0.0, 0.0, pregrasp_height),), device=device, dtype=dtype)
    pregrasp_target = initial_ball_target + pregrasp_offset
    approach_rotation_w = torch.tensor(GRASP_ROTATION_W, device=device, dtype=dtype).unsqueeze(0)

    # 阶段 1：先从已知安全分支离线反解，再以夹爪张开的关节轨迹移动到球上方。
    # 该高度即使叠加约 1 cm 的关节跟踪误差，碰撞垫底面与球顶仍有安全余量。
    pregrasp_joint_target, solved_position_error, solved_rotation_error = solve_approach_joint_target(
        robot,
        pregrasp_target,
        approach_rotation_w,
    )
    if solved_position_error > 0.002 or solved_rotation_error > 0.02:
        raise RuntimeError(
            "Pregrasp IK did not converge: "
            f"position_error={solved_position_error:.6f}, rotation_error={solved_rotation_error:.6f}"
        )
    execute_arm_trajectory(
        sim,
        robot,
        block,
        ball,
        pregrasp_joint_target,
        GRIPPER_OPEN_POSITION,
        motion_steps,
    )

    # 运动期间软球可能仍有毫米级滚动，因此按最新球心再求解一次预抓取位置。
    tracked_pregrasp_target = ball_center(ball).clone() + pregrasp_offset
    tracked_joint_target, solved_position_error, solved_rotation_error = solve_approach_joint_target(
        robot,
        tracked_pregrasp_target,
        approach_rotation_w,
    )
    if solved_position_error > 0.002 or solved_rotation_error > 0.02:
        raise RuntimeError(
            "Tracked pregrasp IK did not converge: "
            f"position_error={solved_position_error:.6f}, rotation_error={solved_rotation_error:.6f}"
        )
    execute_arm_trajectory(
        sim,
        robot,
        block,
        ball,
        tracked_joint_target,
        GRIPPER_OPEN_POSITION,
        max(60, motion_steps // 2),
    )
    _, _, actual_pregrasp_pos_w, _ = grasp_point_kinematics(
        robot.data.root_link_pose_w.torch,
        robot.data.joint_pos.torch[:, :6],
    )
    pregrasp_error = torch.linalg.vector_norm(actual_pregrasp_pos_w - tracked_pregrasp_target, dim=-1).item()
    gripper_index = robot.body_names.index("gripper")
    fk_error = torch.linalg.vector_norm(
        forward_kinematics_and_jacobian(
            robot.data.root_link_pose_w.torch,
            robot.data.joint_pos.torch[:, :6],
        )[0]
        - robot.data.body_link_pose_w.torch[:, gripper_index, :3],
        dim=-1,
    ).item()
    print(
        f"PICK_LIFT_STAGE=pregrasp error_m={pregrasp_error:.6f} " f"ball_center={ball_center(ball)[0].cpu().tolist()}",
        flush=True,
    )
    print(f"GRIPPER_PAD_CENTERS_PREGRASP={gripper_pad_centers(robot).cpu().tolist()}", flush=True)

    # 阶段 2：用最新球心坐标对心，保持夹爪张开和当前姿态垂直下降。
    grasp_target = ball_center(ball).clone()
    grasp_error, _ = move_grasp_point(
        sim,
        robot,
        block,
        ball,
        grasp_target,
        GRIPPER_OPEN_POSITION,
        motion_steps,
        target_rotation_w=approach_rotation_w,
    )
    print(
        f"PICK_LIFT_STAGE=grasp error_m={grasp_error:.6f} " f"ball_center={ball_center(ball)[0].cpu().tolist()}",
        flush=True,
    )
    print(f"GRIPPER_PAD_CENTERS_OPEN={gripper_pad_centers(robot).cpu().tolist()}", flush=True)
    # 阶段 3：末端保持不动，joint7/8 平滑闭合。
    hold_grasp(
        sim,
        robot,
        block,
        ball,
        grasp_target,
        GRIPPER_OPEN_POSITION,
        GRIPPER_GRASP_POSITION,
        max(60, motion_steps // 2),
        approach_rotation_w,
    )
    print(
        f"PICK_LIFT_STAGE=closed ball_center={ball_center(ball)[0].cpu().tolist()} "
        f"finger_joints={robot.data.joint_pos.torch[0, 6:].cpu().tolist()}",
        flush=True,
    )
    print(f"GRIPPER_PAD_CENTERS_CLOSED={gripper_pad_centers(robot).cpu().tolist()}", flush=True)

    # 阶段 4：保持闭合，让接触和软体形变充分收敛后再抬升。
    measured_gap, ball_width_after_grasp = grasp_geometry_metrics(robot, ball)
    print(
        f"PHYSICAL_GRASP target_gap_m={GRIPPER_TARGET_GAP:.6f} "
        f"measured_gap_m={measured_gap:.6f} ball_width_on_grasp_axis_m={ball_width_after_grasp:.6f}",
        flush=True,
    )
    closed_target = robot.data.joint_pos.torch.clone()
    closed_target[:, 6:] = GRIPPER_GRASP_POSITION
    for _ in range(60):
        step_scene(sim, robot, block, ball, closed_target)

    # 阶段 5：从当前抓取点沿世界 Z 轴直接抬升，并保持闭合时的可达姿态。
    # 这避免了旧实现切换到另一个 IK 关节分支而产生的横向扫动。
    center_before_lift = ball_center(ball).clone()
    _, _, grasp_before_lift, _ = grasp_point_kinematics(
        robot.data.root_link_pose_w.torch,
        robot.data.joint_pos.torch[:, :6],
    )
    lift_target = grasp_before_lift.clone()
    lift_target[:, 2] += lift_height
    lift_error, _ = move_grasp_point(
        sim,
        robot,
        block,
        ball,
        lift_target,
        GRIPPER_GRASP_POSITION,
        motion_steps * 2,
        target_rotation_w=approach_rotation_w,
    )
    print(
        f"PICK_LIFT_STAGE=lift error_m={lift_error:.6f} "
        f"ball_center={ball_center(ball)[0].cpu().tolist()}",
        flush=True,
    )
    final_target = robot.data.joint_pos.torch.clone()
    final_target[:, 6:] = GRIPPER_GRASP_POSITION
    for _ in range(120):
        step_scene(sim, robot, block, ball, final_target)

    # 使用球体全部节点的平均 Z 位移，而不是夹爪命令位移，作为真实完成指标。
    center_after_lift = ball_center(ball)
    actual_lift = (center_after_lift[:, 2] - center_before_lift[:, 2]).item()
    return {
        "pregrasp_error_m": pregrasp_error,
        "grasp_error_m": grasp_error,
        "lift_target_error_m": lift_error,
        "fk_alignment_error_m": fk_error,
        "commanded_lift_m": lift_height,
        "actual_ball_lift_m": actual_lift,
        "ball_height_before_m": center_before_lift[:, 2].item(),
        "ball_height_after_m": center_after_lift[:, 2].item(),
    }


def main() -> None:
    """构建场景、执行随机 episode，并汇总运行时正确性检查。"""
    # 尽早拒绝无效参数，避免启动昂贵的 Kit/Newton 场景后才报错。
    if args_cli.episodes < 1 or args_cli.steps_per_episode < 1:
        raise ValueError("--episodes and --steps-per-episode must both be positive")
    if args_cli.motion_steps < 1:
        raise ValueError("--motion-steps must be positive")
    if args_cli.lift_height <= 0.0:
        raise ValueError("--lift-height must be positive")
    if args_cli.pick_lift and args_cli.episodes != 1:
        raise ValueError("--pick-lift currently supports exactly one episode")

    # SimulationContext 必须先创建，再生成使用该后端的资产，最后统一 reset 初始化。
    sim = make_sim()
    sim.set_camera_view(eye=(1.25, -1.20, 1.15), target=(0.12, 0.0, TABLE_TOP_Z))
    robot, block, ball = spawn_scene()
    sim.reset()

    # USD/URDF 转换若丢失夹指关节，会让后续 [:6]、[6:] 切片静默出错。
    if robot.num_joints != 8:
        raise RuntimeError(f"Expected 8 YAM joints, got {robot.num_joints}: {robot.joint_names}")

    rng = random.Random(args_cli.seed)
    results: list[dict[str, object]] = []
    pick_lift_result: dict[str, float] | None = None
    for episode in range(args_cli.episodes):
        # 同一个 Random 实例连续采样；相同 seed 可完整复现整个位置序列。
        block_pos, ball_pos = sample_object_positions(rng)
        reset_episode(robot, block, ball, block_pos, ball_pos)

        # 复位后先让重力、桌面碰撞和关节驱动达到稳定状态。
        for _ in range(args_cli.steps_per_episode):
            step_scene(sim, robot, block, ball, robot.data.default_joint_pos.torch)

        if args_cli.pick_lift:
            settled_ball_pos = tuple(ball_center(ball)[0].cpu().tolist())
            pick_lift_result = pick_and_lift_ball(
                sim,
                robot,
                block,
                ball,
                settled_ball_pos,
                args_cli.motion_steps,
                args_cli.lift_height,
            )

        # NaN/Inf 往往意味着求解器发散，必须作为集成测试失败处理。
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

    # 以下带固定前缀的输出既便于人工查看，也便于脚本/CI 解析。
    print(f"NEWTON_SOLVER={type(sim.cfg.physics.solver_cfg).__name__}")
    print(f"YAM_JOINT_NAMES={robot.joint_names}")
    print(f"YAM_BODY_NAMES={robot.body_names}")
    gripper_index = robot.body_names.index("gripper")
    print(f"YAM_GRIPPER_POSE={robot.data.body_link_pose_w.torch[0, gripper_index].cpu().tolist()}")
    print("YAM_KINEMATICS=project URDF-derived Torch FK/Jacobian")
    for result in results:
        print(f"RESET_RESULT={result}")
    if pick_lift_result is not None:
        print(f"PICK_LIFT_RESULT={pick_lift_result}")
        if pick_lift_result["fk_alignment_error_m"] > 0.002:
            raise RuntimeError(f"URDF FK does not align with the simulated gripper: {pick_lift_result}")
        # 成功标准基于实际球心位移，而不是末端目标或控制命令。
        minimum_lift = args_cli.lift_height * MIN_LIFT_RATIO
        if pick_lift_result["actual_ball_lift_m"] < minimum_lift:
            raise RuntimeError(f"Ball was not lifted high enough; required {minimum_lift:.3f} m: {pick_lift_result}")
        print("YAM_DEFORMABLE_BALL_LIFT_OK")
    print("YAM_DEFORMABLE_TASK_INTEGRATION_OK")

    if args_cli.keep_open:
        if args_cli.headless:
            raise ValueError("--keep-open requires the Kit visualizer; use --visualizer kit.")
        print("GUI_READY_CLOSE_WINDOW_TO_EXIT", flush=True)
        # GUI 保持循环继续推进物理，窗口不会在测试结束后立刻消失。
        hold_target = robot.data.joint_pos.torch.clone()
        while simulation_app.is_running():
            step_scene(sim, robot, block, ball, hold_target)


if __name__ == "__main__":
    try:
        main()
    finally:
        # 即使 Python 抛出异常也显式释放 Kit、CUDA 和 USD 资源。
        simulation_app.close()
