"""Minimal YAM Ultra 2 kinematics derived from the vendored official URDF.

Only PyTorch is used.  This module exists because the Isaac Lab beta coupled
MJWarp/VBD manager currently exits while evaluating articulation Jacobians.
The constants below mirror ``assets/vendor/i2rt/yam_ultra_2_v2/yam_ultra.urdf``.
"""

from __future__ import annotations

import torch


JOINT_ORIGINS_XYZ = (
    (0.0, 0.0, 0.0733),
    (-0.0455, 0.03285, -0.02),
    (4.08431e-7, -0.0678, 0.264),
    (-0.0600003, 0.0678, -0.244999),
    (-0.0403003, -0.0338507, -0.0703851),
    (0.0404996, 2.39858e-7, -0.0419481),
)

JOINT_ORIGINS_RPY = (
    (0.0, 1.57079632679, 3.14159265359),
    (-4.21325e-15, -1.5065e-29, 9.98062e-15),
    (-6.20015e-15, 6.98506e-29, -5.07758e-15),
    (1.04275e-14, -2.77556e-17, -4.04405e-15),
    (-1.02186e-14, 2.77556e-17, -6.12323e-17),
    (7.6222e-29, -5.3963e-29, 4.42535e-15),
)

JOINT_AXES = (
    (-1.0, 0.0, 0.0),
    (-3.58252e-15, -1.0, 3.2007e-16),
    (4.54273e-15, 1.0, 1.04275e-14),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.46014e-15, 2.40808e-14, -1.0),
)


def _rpy_matrix(rpy: torch.Tensor) -> torch.Tensor:
    """Convert fixed-axis roll-pitch-yaw angles to rotation matrices."""
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
    """Apply Rodrigues' formula to one normalized axis and batched angles."""
    axis = axis / torch.linalg.vector_norm(axis)
    x, y, z = axis
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    eye = torch.eye(3, device=angle.device, dtype=angle.dtype)
    outer = axis[:, None] * axis[None, :]
    c = torch.cos(angle)[..., None, None]
    s = torch.sin(angle)[..., None, None]
    return c * eye + (1.0 - c) * outer + s * skew


def _quat_xyzw_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert normalized (x, y, z, w) quaternions to rotation matrices."""
    quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
    x, y, z, w = quat.unbind(-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
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
    """Return gripper position, rotation, and geometric Jacobian in world frame.

    Args:
        root_pose_w: Base-link pose ``(x, y, z, qx, qy, qz, qw)``, shape ``(N, 7)``.
        arm_joint_pos: Revolute joint positions ``joint1..joint6``, shape ``(N, 6)``.
    """
    if root_pose_w.ndim != 2 or root_pose_w.shape[1] != 7:
        raise ValueError(f"Expected root_pose_w shape (N, 7), got {tuple(root_pose_w.shape)}")
    if arm_joint_pos.shape != (root_pose_w.shape[0], 6):
        raise ValueError(f"Expected arm_joint_pos shape ({root_pose_w.shape[0]}, 6), got {arm_joint_pos.shape}")

    batch_size = root_pose_w.shape[0]
    device, dtype = root_pose_w.device, root_pose_w.dtype
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
        origin_transform = torch.eye(4, device=device, dtype=dtype).repeat(batch_size, 1, 1)
        origin_transform[:, :3, :3] = origin_rotations[joint_id]
        origin_transform[:, :3, 3] = origins_xyz[joint_id]
        joint_transform = transform @ origin_transform

        joint_positions_w.append(joint_transform[:, :3, 3])
        axis_w = (joint_transform[:, :3, :3] @ axes[joint_id].expand(batch_size, 3).unsqueeze(-1)).squeeze(-1)
        joint_axes_w.append(axis_w)

        motion_transform = torch.eye(4, device=device, dtype=dtype).repeat(batch_size, 1, 1)
        motion_transform[:, :3, :3] = _axis_angle_matrix(axes[joint_id], arm_joint_pos[:, joint_id])
        transform = joint_transform @ motion_transform

    gripper_pos_w = transform[:, :3, 3]
    gripper_rot_w = transform[:, :3, :3]
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
    """Compute one position-only damped least-squares joint increment."""
    error = target_pos_w - current_pos_w
    jacobian_t = position_jacobian_w.transpose(1, 2)
    regularizer = (damping**2) * torch.eye(3, device=current_pos_w.device, dtype=current_pos_w.dtype)
    return (jacobian_t @ torch.linalg.solve(position_jacobian_w @ jacobian_t + regularizer, error.unsqueeze(-1))).squeeze(-1)


def damped_least_squares_pose_step(
    current_pos_w: torch.Tensor,
    target_pos_w: torch.Tensor,
    current_rot_w: torch.Tensor,
    target_rot_w: torch.Tensor,
    geometric_jacobian_w: torch.Tensor,
    damping: float = 0.05,
) -> torch.Tensor:
    """Compute one 6D pose DLS step with a world-frame rotation error."""
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
        jacobian_t
        @ torch.linalg.solve(geometric_jacobian_w @ jacobian_t + regularizer, pose_error.unsqueeze(-1))
    ).squeeze(-1)
