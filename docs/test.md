测试初始化场景
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/smoke_test_integrated_task.py \
--visualizer kit \
--keep-open \
--episodes 3 \
--steps-per-episode 120 \
--seed 7

夹爪测试
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/smoke_test_integrated_task.py \
--visualizer kit \
--episodes 1 \
--steps-per-episode 120 \
--seed 7 \
--gripper-check \
--keep-open

## 2026-09-13：提速与物理抓起

在工程根目录执行以下 GUI 命令（抓起 3 cm，完成后继续保持窗口）：

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
../isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/smoke_test_integrated_task.py \
--visualizer kit --keep-open --episodes 1 --steps-per-episode 60 \
--pick-lift --lift-height 0.03 --motion-steps 240 --seed 7
```

速度在主脚本常量区调节，单位是**仿真时间**下的速度，不保证 GUI 实时播放：

- `GRIPPER_COMMAND_SPEED = 0.04`：空载开合单指速度，完整单程命令约 1.18 秒。
- `GRIPPER_CONTACT_SPEED = 0.01`：抓球时接触闭合速度，避免空载速度直接挤压软球。
- `ARM_CARTESIAN_SPEED = 0.10`：下降和抬升的直线轨迹速度。
- 空载关节轨迹默认峰值 1 rad/s；原有夹爪驱动速度上限 0.05 m/s 保留。
- `--motion-steps` 现在是所有运动段的时间倍率基准，**不是固定阶段步数**：240 为默认、480 为半速。稳定等待和验收保持步数不受它影响。过度减小会导致驱动跟不上或接触不稳。

抓取修正：原来整根夹指的单一凸包会填平凹部；现在用 Newton 自带凸分解处理原始夹指网格，只覆盖当前场景碰撞近似，不修改资产文件或视觉模型，不添加板子、不绑定软体节点。保持阶段持续使用最后的驱动目标，不将有跟踪误差的实际角度重新设为目标。

本次必要的无 GUI 验证：seed=7、1 episode、目标抬升 0.03 m。球心实际抬升 **0.026534 m**，球底离桌 **0.024840 m**，通过抬升后的 120 步（1 秒仿真时间）保持检查，输出 `YAM_DEFORMABLE_BALL_LIFT_OK`。仍保持原验收标准：球心升高至少为目标的 85%，球底离桌至少 2 mm，并检查夹持区域和过度变形。

这只证明该种子的短距离抓起；0.4 m 抬升、长时间保持、多随机位置成功率、放置到方块尚未验收。本轮未另跑空载开合专项或 GUI 测试。参数接口清理后做了语法检查，未重复启动仿真。
