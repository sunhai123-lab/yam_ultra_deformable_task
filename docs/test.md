测试初始化场景
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/smoke_test_integrated_task.py \
--visualizer kit \
--keep-open \
--episodes 3 \
--steps-per-episode 120 \
--seed 7