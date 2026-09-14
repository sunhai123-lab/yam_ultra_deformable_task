"""由项目内 YAM Ultra 2 官方 URDF 推导的最小运动学实现。

模块只依赖 PyTorch，提供以下能力：

* 根据基座位姿和 joint1..joint6 计算 gripper link 的世界位姿；
* 计算末端 6×6 几何雅可比（前三行线速度，后三行角速度）；
* 使用阻尼最小二乘法（DLS）计算位置或完整位姿的单步关节增量。

之所以不直接请求仿真 articulation Jacobian，是因为当前 Isaac Lab beta 的
MJWarp/VBD 耦合管理器在该路径上可能原生退出。下面三组常量逐项对应
``assets/vendor/i2rt/yam_ultra_2_v2/yam_ultra.urdf`` 中 joint1..joint6。
所有长度单位为 m，角度单位为 rad，四元数顺序为 xyzw。
"""

from __future__ import annotations

import torch

# 每个关节坐标系相对于前一 link 坐标系的固定平移 <origin xyz="...">。
JOINT_ORIGINS_XYZ = (
    (0.0, 0.0, 0.0733),
    (-0.0455, 0.03285, -0.02),
    (4.08431e-7, -0.0678, 0.264),
    (-0.0600003, 0.0678, -0.244999),
    (-0.0403003, -0.0338507, -0.0703851),
    (0.0404996, 2.39858e-7, -0.0419481),
)

# 每个关节坐标系相对于前一 link 的固定欧拉角 <origin rpy="...">。
JOINT_ORIGINS_RPY = (
    (0.0, 1.57079632679, 3.14159265359),
    (-4.21325e-15, -1.5065e-29, 9.98062e-15),
    (-6.20015e-15, 6.98506e-29, -5.07758e-15),
    (1.04275e-14, -2.77556e-17, -4.04405e-15),
    (-1.02186e-14, 2.77556e-17, -6.12323e-17),
    (7.6222e-29, -5.3963e-29, 4.42535e-15),
)

# 各旋转关节在自身关节坐标系中的单位转轴 <axis xyz="...">。
JOINT_AXES = (
    (-1.0, 0.0, 0.0),
    (-3.58252e-15, -1.0, 3.2007e-16),
    (4.54273e-15, 1.0, 1.04275e-14),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.46014e-15, 2.40808e-14, -1.0),
)


def _rpy_matrix(rpy: torch.Tensor) -> torch.Tensor:
    """把固定轴 roll-pitch-yaw 转为旋转矩阵。

    输入形状为 ``(..., 3)``，输出为 ``(..., 3, 3)``，采用 URDF 的
    ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` 约定。
    """
    roll, pitch, yaw = rpy.unbind(-1)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    return torch.stack(
        (
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr,
            -sp,
            cp * sr,
            cp * cr,
        ),
        dim=-1,
    ).reshape(rpy.shape[:-1] + (3, 3))


def _axis_angle_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """使用 Rodrigues 公式计算绕固定轴旋转的批量旋转矩阵。

    ``axis`` 形状为 ``(3,)``，``angle`` 可包含任意 batch 维度。
    """
    axis = axis / torch.linalg.vector_norm(axis)
    x, y, z = axis
    zero = torch.zeros_like(x)
    # [axis]× 是满足 [axis]× v = axis × v 的反对称矩阵。
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)  # ？作用
    eye = torch.eye(3, device=angle.device, dtype=angle.dtype)
    outer = axis[:, None] * axis[None, :]
    c = torch.cos(angle)[..., None, None]
    s = torch.sin(angle)[..., None, None]
    return c * eye + (1.0 - c) * outer + s * skew


def _quat_xyzw_matrix(quat: torch.Tensor) -> torch.Tensor:
    """将 Isaac Lab 的 ``(x, y, z, w)`` 四元数转为旋转矩阵。

    函数内部归一化四元数，以抑制仿真浮点误差造成的非正交旋转矩阵。
    """
    quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
    x, y, z, w = quat.unbind(-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),  # ？作用，为什么这么写
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def forward_kinematics_and_jacobian(
    root_pose_w: torch.Tensor, arm_joint_pos: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算 gripper link 的世界位姿及几何雅可比。

    Args:
        root_pose_w: base link 位姿 ``(x,y,z,qx,qy,qz,qw)``，形状 ``(N,7)``。
        arm_joint_pos: joint1..joint6 角度，形状 ``(N,6)``。

    Returns:
        末端位置 ``(N,3)``、旋转矩阵 ``(N,3,3)`` 和几何雅可比 ``(N,6,6)``。
        雅可比满足 ``[v; ω] = J q_dot``，所有量均表达在世界坐标系。
    """
    if root_pose_w.ndim != 2 or root_pose_w.shape[1] != 7:
        raise ValueError(f"Expected root_pose_w shape (N, 7), got {tuple(root_pose_w.shape)}")
    if arm_joint_pos.shape != (root_pose_w.shape[0], 6):
        raise ValueError(f"Expected arm_joint_pos shape ({root_pose_w.shape[0]}, 6), got {arm_joint_pos.shape}")

    batch_size = root_pose_w.shape[0]
    device, dtype = root_pose_w.device, root_pose_w.dtype
    # transform 始终表示“当前 link 坐标系到世界坐标系”的齐次变换。
    transform = torch.eye(4, device=device, dtype=dtype).repeat(batch_size, 1, 1)
    transform[:, :3, :3] = _quat_xyzw_matrix(root_pose_w[:, 3:])
    transform[:, :3, 3] = root_pose_w[:, :3]

    origins_xyz = torch.tensor(JOINT_ORIGINS_XYZ, device=device, dtype=dtype)
    origins_rpy = torch.tensor(JOINT_ORIGINS_RPY, device=device, dtype=dtype)
    axes = torch.tensor(JOINT_AXES, device=device, dtype=dtype)
    origin_rotations = _rpy_matrix(origins_rpy)
    joint_positions_w: list[torch.Tensor] = []
    joint_axes_w: list[torch.Tensor] = []

    for joint_id in range(6):
        # 先乘 URDF 中不随关节角变化的 origin 变换，到达当前关节坐标系。
        origin_transform = torch.eye(4, device=device, dtype=dtype).repeat(batch_size, 1, 1)
        origin_transform[:, :3, :3] = origin_rotations[joint_id]
        origin_transform[:, :3, 3] = origins_xyz[joint_id]
        joint_transform = transform @ origin_transform

        # 雅可比需要保存每个关节原点 p_i 和转轴 z_i 的世界坐标表达。
        joint_positions_w.append(joint_transform[:, :3, 3])
        axis_w = (joint_transform[:, :3, :3] @ axes[joint_id].expand(batch_size, 3).unsqueeze(-1)).squeeze(-1)
        joint_axes_w.append(axis_w)

        # 再乘由当前关节角产生的轴角旋转，进入下一 link 坐标系。
        motion_transform = torch.eye(4, device=device, dtype=dtype).repeat(batch_size, 1, 1)
        motion_transform[:, :3, :3] = _axis_angle_matrix(axes[joint_id], arm_joint_pos[:, joint_id])
        transform = joint_transform @ motion_transform

    gripper_pos_w = transform[:, :3, 3]
    gripper_rot_w = transform[:, :3, :3]
    # 对旋转关节 i：线速度列 Jv_i = z_i × (p_ee-p_i)，角速度列 Jw_i = z_i。
    linear_columns = [
        torch.linalg.cross(axis_w, gripper_pos_w - joint_pos_w, dim=-1)
        for axis_w, joint_pos_w in zip(joint_axes_w, joint_positions_w)
    ]
    jacobian = torch.cat(
        (torch.stack(linear_columns, dim=-1), torch.stack(joint_axes_w, dim=-1)),
        dim=1,
    )
    return gripper_pos_w, gripper_rot_w, jacobian


def damped_least_squares_position_step(
    current_pos_w: torch.Tensor,
    target_pos_w: torch.Tensor,
    position_jacobian_w: torch.Tensor,
    damping: float = 0.05,
) -> torch.Tensor:
    """计算仅控制末端位置的单步 DLS 关节增量。

    Args:
        current_pos_w: 当前点位置，形状 ``(N,3)``。
        target_pos_w: 目标点位置，形状 ``(N,3)``。
        position_jacobian_w: 线速度雅可比，形状 ``(N,3,6)``。
        damping: 阻尼系数 λ；越大越稳定，但单步收敛更慢。

    Returns:
        关节角增量 ``Δq``，形状 ``(N,6)``，单位 rad。

    使用公式 ``Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ e``。相比直接求逆 ``J``，该形式
    在机械臂接近奇异位形时仍保持有限解。
    """
    error = target_pos_w - current_pos_w
    jacobian_t = position_jacobian_w.transpose(1, 2)
    regularizer = (damping**2) * torch.eye(3, device=current_pos_w.device, dtype=current_pos_w.dtype)
    return (
        jacobian_t @ torch.linalg.solve(position_jacobian_w @ jacobian_t + regularizer, error.unsqueeze(-1))
    ).squeeze(-1)


def damped_least_squares_pose_step(
    current_pos_w: torch.Tensor,
    target_pos_w: torch.Tensor,
    current_rot_w: torch.Tensor,
    target_rot_w: torch.Tensor,
    geometric_jacobian_w: torch.Tensor,
    damping: float = 0.05,
) -> torch.Tensor:
    """计算同时控制 3 维位置和 3 维姿态的单步 DLS 关节增量。

    Args:
        current_pos_w/target_pos_w: 当前/目标位置，形状 ``(N,3)``。
        current_rot_w/target_rot_w: 当前/目标旋转矩阵，形状 ``(N,3,3)``。
        geometric_jacobian_w: 几何雅可比，形状 ``(N,6,6)``。
        damping: 6 维 DLS 阻尼系数 λ。

    Returns:
        joint1..joint6 的角度增量，形状 ``(N,6)``。

    姿态误差使用两旋转矩阵对应列向量叉积之和，是适合小步迭代的世界系角误差。
    """
    position_error = target_pos_w - current_pos_w
    orientation_error = 0.5 * (
        torch.linalg.cross(current_rot_w[:, :, 0], target_rot_w[:, :, 0], dim=-1)
        + torch.linalg.cross(current_rot_w[:, :, 1], target_rot_w[:, :, 1], dim=-1)
        + torch.linalg.cross(current_rot_w[:, :, 2], target_rot_w[:, :, 2], dim=-1)
    )
    pose_error = torch.cat((position_error, orientation_error), dim=-1)
    jacobian_t = geometric_jacobian_w.transpose(1, 2)
    regularizer = (damping**2) * torch.eye(6, device=current_pos_w.device, dtype=current_pos_w.dtype)
    return (
        jacobian_t @ torch.linalg.solve(geometric_jacobian_w @ jacobian_t + regularizer, pose_error.unsqueeze(-1))
    ).squeeze(-1)
