# YAM Ultra 基于 Isaac Lab Newton 的可变形球抓取与放置

[English](README.md) | 简体中文

本项目使用 Isaac Lab + Newton 构建 YAM Ultra 2 桌面操作场景。机器人通过原始夹指几何、碰撞、摩擦和软体变形真实抓取 VBD 可变形球，并可选择把球放回桌面或搬运到动态刚体方块上。

## 当前仓库结构

任务运行只保留一个主入口：

```text
scripts/yam_pick_ball.py
```

后续新增任务模式优先通过命令行参数或内部函数加入 `yam_pick_ball.py`，不再复制新的 `*_test.py`、`*_integrated.py` 等并行版本。

其他核心文件：

- `source/yam_ultra_deformable_place/yam_ultra_deformable_place/yam_kinematics.py`：项目自写 PyTorch FK、几何 Jacobian、DLS IK；
- `scripts/convert_yam_ultra_2.py`：URDF→USD 资产转换工具，不是第二个任务主程序；
- `docs/yam_pick_ball_reference.md`：参数、函数、变量与核心数学；
- `docs/yam_pick_ball_workflow.md`：运行方式、状态机、验收、调试与维护约定；
- `VERSIONS.md`：验证环境与当前限制。

## 任务流程

```text
随机采样 / reset
  -> 等待方块和软球稳定
  -> 预抓取 IK
  -> 接近
  -> 接触敏感下降
  -> 闭合夹爪
  -> 物理抓取检查
  -> 抬升
  -> 可选极坐标搬运
  -> 下放 / 松爪
  -> 撤离
  -> 最终稳定检查
```

这不是强化学习任务，而是确定性硬编码控制器。抓取过程中不会把软球节点绑定到夹爪，球能否被抬起取决于真实碰撞、法向接触、摩擦和软体变形。

## 当前桌面与随机工作区

当前学习版采用“机器人位于桌心 + 环形随机采样”：

```python
TABLE_SIZE = (1.50, 1.50, 0.08)
TABLE_CENTER = (0.0, 0.0, 0.0)
ROBOT_BASE_XY = TABLE_CENTER[:2]
OBJECT_RADIUS_RANGE = (0.30, 0.45)
OBJECT_BEARING_RANGE = (-2.40, 3.00)
MIN_OBJECT_PLANAR_DISTANCE = 0.18
MAX_TRANSFER_BEARING_DELTA = math.pi
```

物体位置按极坐标生成：

```text
x = robot_x + r cos(theta)
y = robot_y + r sin(theta)
```

随后还会检查物体是否完整落在桌面内、球和方块是否至少相距 18 cm、两者搬运方位差是否不超过 π。随机点落入这个区域并不等于完整末端 pose 一定可达，最终仍由 IK 和真实运动误差检查决定。

## 一次性安装项目包

```bash
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/env_isaaclab/bin/python \
-m pip install -e source/yam_ultra_deformable_place
```

## 常用运行方式

### 只做 reset 和沉降检查

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/yam_pick_ball.py --visualizer none \
--episodes 3 --steps-per-episode 60 --seed 7
```

### 只测试夹爪开合

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/yam_pick_ball.py --visualizer kit --keep-open \
--gripper-check --episodes 1 --steps-per-episode 60 --seed 7
```

### 抓取并抬升

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/yam_pick_ball.py --visualizer none \
--pick-lift --episodes 1 --steps-per-episode 60 --lift-height 0.03 --seed 7
```

### 抓起后放回原位置

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/yam_pick_ball.py --visualizer none \
--pick-lift --put-back --episodes 1 --steps-per-episode 60 --lift-height 0.03 --seed 7
```

### 抓起后放到方块上

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/yam_pick_ball.py --visualizer kit --keep-open \
--pick-lift --place-on-block --episodes 3 --steps-per-episode 60 --lift-height 0.03 --seed 7
```

`--motion-steps` 现在是整体运动时间倍率基准，不是每个阶段固定运行多少步。当前主程序要求 `>=240`；如果要主动调快/调慢某一阶段，优先修改源码中的具名速度常量。

## VS Code

仓库中的 `.vscode/launch.json` 和 `.vscode/tasks.json` 已统一指向 `scripts/yam_pick_ball.py` 的不同 CLI 模式。默认 build task 为 GUI 放球到方块模式。资产转换单独保留一个工具任务。

## 重新生成机器人 USD

只有 vendored URDF 或 mesh 改动时才需要执行：

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/convert_yam_ultra_2.py
```

## 重要成功标记

常见终端标记：

```text
GRIPPER_CHECK_OK
QUICK_GRASP_OK
YAM_DEFORMABLE_BALL_LIFT_OK
YAM_DEFORMABLE_BALL_PUT_BACK_OK
YAM_DEFORMABLE_BALL_ON_BLOCK_OK
EPISODE_COMPLETE
EPISODE_SUMMARY
YAM_DEFORMABLE_TASK_INTEGRATION_OK
```

这些标记表示当前请求模式通过了配置好的几何、速度、支撑、沉降和抬升/放置判据，不代表机器人已经具备任意物体、任意姿态、完整工作空间上的通用抓取能力。

## 文档

`docs/` 只保留两份：

1. [`docs/yam_pick_ball_reference.md`](docs/yam_pick_ball_reference.md)：参数与函数讲解；
2. [`docs/yam_pick_ball_workflow.md`](docs/yam_pick_ball_workflow.md)：运行、流程、验收和调试。

如果文档中的数值与源码不一致，始终以 `scripts/yam_pick_ball.py` 为准。
