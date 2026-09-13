# 抓球并放到刚体方块顶面

当前分阶段速度和等待逻辑已优化，见 [流程提速说明](flow_speed.md)；连续随机运行见 [多轮 episode](multi_episode.md)。下文单轮验证数据为历史记录。

## 运行

- Ctrl+Shift+B 默认执行 `Task: Place ball on block GUI`。
- F5 在启动配置下拉框中选择同名配置。
- 旧的原地放回配置仍保留，可单独选择。

工程根目录终端命令：

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
../isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/smoke_test_integrated_task.py \
--visualizer kit --keep-open --episodes 3 --steps-per-episode 60 \
--pick-lift --place-on-block --lift-height 0.03 --seed 7
```

`--place-on-block` 需要 `--pick-lift`，与 `--put-back`、`--stop-after-hold` 互斥。
`--lift-height 0.03` 是初始抓起验收的高度，不是整个搬运过程的最大高度。

## 流程和边界

1. 随机初始化球和方块，等待稳定，按原流程抓起球。
2. 读取方块实际位置，目标取其顶面中心。
3. 先垂直抬高，球底接触包络高于方块顶面约 6 cm 后再横向搬运。
4. 对准顶面中心，较远段下降速度 0.06 m/s，最后接近速度 0.015 m/s。
5. 根据球底减去粒子接触半径与方块顶面的距离决定释放时机；缓慢张爪，等待稳定后撤离 10 cm。
6. 撤离后复查球和方块的稳定性，不能把掉回桌面视作成功。

复用主脚本 `put_ball_back(..., on_block=True)`，不改变原地放回分支。方块保持动态刚体，球保持自由可变形体；没有节点绑定、传送、清零释放速度或修改材质。
当前仅接受近水平顶面：方块倾角约不超过 1.15°、相对搬运前的位移不超过 3 mm，否则中止。它不是任意倾斜物体的通用放置算法。

验收包括：目标水平误差 ≤ 5 mm、松爪后最大漂移 ≤ 3 mm、球心投影在实际方块顶面内侧、球底相对方块顶面的距离和球/方块速度满足稳定判据。几何近接判据不是接触力传感器测量。

## 2026-09-13 验证结果

完成一次无 GUI 全流程验证，seed=7：

- 初始实际抬升 26.68 mm（目标 30 mm），原抓取验收通过。
- 放置相对目标中心的水平误差 **2.253 mm**。
- 松爪至撤离复查的最大水平漂移 **0.735 mm**。
- 球底接触包络相对方块顶面为 **−0.154 mm**，处于允许的接触误差范围。
- 完全张爪后 **0.25 秒仿真时间**满足稳定条件；撤离后继续检查 **0.5 秒**。
- 最终球心速度约 **0.289 mm/s**，最大节点速度约 **0.538 mm/s**。
- 输出 `YAM_DEFORMABLE_BALL_ON_BLOCK_OK` 及集成成功标记。

通用 `BALL_SUPPORT` 日志里的 `contact_clearance_m` 仍以桌面为参考，因此球在 5 cm 高的方块上时约为 0.05 m，并非悬空错误。`PLACE_ON_BLOCK_RESULT.contact_clearance_m` 才是相对方块顶面的距离；通用初始化位移日志包含整段搬运距离，不是释放漂移。

本次运行后修正了目标球心日志的名义 Z 值及阶段标签（不影响运动），并做语法/配置检查，未重复仿真。尚未验证多随机位置、长期保持或 GUI 画面；相同位置 seed 也不保证软体网格完全一致。
