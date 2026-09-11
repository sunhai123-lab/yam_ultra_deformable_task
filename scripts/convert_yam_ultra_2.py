"""Convert the vendored I2RT YAM Ultra 2 URDF to a fixed-base USD asset."""

from pathlib import Path

from isaaclab.app import AppLauncher


app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = PROJECT_ROOT / "assets/vendor/i2rt/yam_ultra_2_v2/yam_ultra.urdf"
USD_DIR = PROJECT_ROOT / "assets/generated/yam_ultra_2"


def main() -> None:
    """Run a deterministic conversion suitable for the first articulation tests."""
    cfg = UrdfConverterCfg(
        asset_path=str(URDF_PATH),
        usd_dir=str(USD_DIR),
        fix_base=True,
        merge_fixed_joints=False,
        collision_from_visuals=True,
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=10.0),
            # stiffness=100.0：刚度，关节偏离目标位置时产生的恢复力
            # damping=10.0：阻尼，用于抑制振荡
            target_type="position",
        ),
    )
    converter = UrdfConverter(cfg)
    print(f"YAM_ULTRA_2_USD_OK={converter.usd_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
