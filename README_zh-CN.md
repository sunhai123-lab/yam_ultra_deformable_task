# YAM Ultra 基于 Isaac Lab Newton 的可变形球抓取与放置

[English](README.md) | 简体中文

本项目使用 Isaac Lab + Newton 构建 YAM Ultra 2 桌面操作场景。机器人通过原始夹指几何、碰撞、摩擦和软体变形真实抓取 VBD 可变形球，并可选择把球放回桌面或搬运到动态刚体方块上。

当前已验证的软件环境为 Isaac Lab `v3.0.0-beta2.patch1`、Isaac Sim `6.0.1.0`、Python `3.12.14`、Newton `1.2.1` 和 Warp `1.13.0`。完整版本记录见 [`VERSIONS.md`](VERSIONS.md)。

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
OBJECT_RADIUS_RANGE = (0.30, 0.43)
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

下面的命令都应在工程根目录执行，并使用匹配版本 Isaac Lab 的 launcher 和 `env_isaaclab` Python 环境。

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

如果要进行大量 episode，可以增加 `--episodes`，例如：

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/yam_pick_ball.py --visualizer none \
--pick-lift --place-on-block --episodes 50 --steps-per-episode 60 \
--motion-steps 240 --lift-height 0.03 --seed 7
```

`--motion-steps` 是整体运动时间倍率基准，不是每个阶段固定运行多少步。当前主程序要求 `>=240`；如果要主动调快/调慢某一阶段，优先修改源码中的具名速度常量。

每个 episode 都会重新随机采样方块和球的位置，复位机器人、方块和软球，等待场景稳定后执行所选任务。可恢复的任务失败会输出 `EPISODE_FAILED`，随后自动进入下一轮。批量结束时会统计 `complete_ok`、`failed`、`settle_ok`、`pick_ok` 和 `placement_ok`。如果出现 NaN/Inf 或 CUDA illegal memory error，程序会停止，因为此时仿真状态已经不可信。

### 最近一次稳健性验证

最近一次自动日志使用当前环形随机工作区，连续运行了 200 个随机 episode：

```text
mode=PICK_LIFT_ON_BLOCK  episodes=200  seed=7
radius=(0.30, 0.43)      bearing=(-2.40, 3.00)
```

| 检查项 | 结果 |
|---|---:|
| 场景沉降稳定 | 200/200 |
| 物理抓取并抬升 | 199/200 |
| 松爪沉降后小球仍留在方块上 | 199/200 |
| 完成的 episode | 199/200 |
| 总体完成率 | 99.5% |

唯一失败的 episode 在 `pick_and_lift` 阶段触发了 `QUICK_GRASP_TIMEOUT`，稳定计数只达到 `1/12`。失败快照显示：小球仍在夹爪包络内，球心与夹爪中心偏差约 `0.84 mm`，没有过度变形，也没有关节限位饱和。因此，这次应归类为抓取稳定性检查超时，而不是已经确认的小球脱落。

释放后的机械臂撤退阶段共出现 42 次 `RETREAT_INCOMPLETE` / `MOTION_TIMEOUT` 警告。这些都是可恢复警告，没有造成放置失败；其余完成的 episode 全部通过最终支撑检查。当前放置成功的判据是：松爪并等待沉降、观察后，小球仍由方块支撑，而不是要求球心必须精确对齐方块中心。

这组结果说明：在当前软球材质、机器人固定姿态、环形工作区、控制器参数和随机种子下，方法表现出较好的稳健性。但单个种子的一次 200 轮运行还不能证明具有通用操作能力；后续应使用更多随机种子，并扩大工作区、材质和参数范围后再做统计结论。完整原始记录见[200 轮运行日志](logs/yam_pick_ball_20260916_135206.log)。

## VS Code

仓库中的 `.vscode/launch.json` 和 `.vscode/tasks.json` 已统一指向 `scripts/yam_pick_ball.py` 的不同 CLI 模式。

- F5：`Task: Place ball on block GUI`，默认 3 轮，完成后通过 `--keep-open` 保持 GUI；
- Ctrl+Shift+B：默认 build task，运行 50 轮 GUI 放球到方块；
- 其他配置覆盖 reset/settle、抓取抬升、夹爪诊断和资产转换。

GUI 批量运行中如果遇到可恢复失败，终端会记录失败并继续下一轮；最终汇总输出后手动关闭 GUI 窗口即可。

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
RUN_CONFIG
EPISODE_START
EPISODE_COMPLETE
EPISODE_FAILED
EPISODE_SUMMARY
RUN_SUCCESS
RUN_COMPLETE
GRASP_FAILURE
```

`EPISODE_COMPLETE` 表示该轮通过了配置好的检查；`RUN_SUCCESS` 表示所有请求的 episode 都完成；`RUN_COMPLETE` 表示批量运行结束，但其中有可恢复失败，需要结合 `EPISODE_FAILED` 和汇总计数查看；`GRASP_FAILURE` 表示顶层致命异常。通过这些脚本化检查不代表机器人已经具备任意物体、任意姿态、完整工作空间上的通用抓取能力。

## 文档

`docs/` 只保留两份：

1. [`docs/yam_pick_ball_reference.md`](docs/yam_pick_ball_reference.md)：参数与函数讲解；
2. [`docs/yam_pick_ball_workflow.md`](docs/yam_pick_ball_workflow.md)：运行、流程、验收和调试。

如果文档中的数值与源码不一致，始终以 `scripts/yam_pick_ball.py` 为准。
