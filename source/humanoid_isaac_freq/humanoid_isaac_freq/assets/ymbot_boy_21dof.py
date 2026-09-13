"""21-DOF YMBOY: existing 12-DOF legs plus waist and AMP upper body.

The derived URDF retains the supplied upper body and uses the 12-DOF
URDF's actuated leg joints and child links. Upper-body tuning is provisional.
"""
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from . import HUMANOID_ISAAC_FREQ_DATA_DIR
from .ymbot_boy_12dof import YMBOT_BOY_12DOF_CFG

YMBOT_BOY_21DOF_CFG = YMBOT_BOY_12DOF_CFG.copy()
YMBOT_BOY_21DOF_CFG.spawn.asset_path = str(
    HUMANOID_ISAAC_FREQ_DATA_DIR / "Robots/ymboy-12dof/ymboy_21dof_training.urdf"
)
YMBOT_BOY_21DOF_CFG.init_state.joint_pos.update({
    "waist_yaw_joint": 0.0, ".*_shoulder_.*_joint": 0.0, ".*_elbow_joint": 0.0,
})
YMBOT_BOY_21DOF_CFG.actuators.update({
    "waist": ImplicitActuatorCfg(
        joint_names_expr=["waist_yaw_joint"], effort_limit_sim=94.0,
        velocity_limit_sim=14.0, stiffness=20.0, damping=5.0,
        armature=150e-6 * 18**2, friction=0.002, dynamic_friction=0.001,
    ),
    # Match AMPMINIMAL's explicit motor model (not hardware-safe limits).
    "arms": IdealPDActuatorCfg(
        joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint"],
        effort_limit=94.0, velocity_limit=14.0, stiffness=40.0,
        damping={".*_shoulder_pitch_joint": 5.0, ".*_shoulder_roll_joint": 5.0,
                 ".*_shoulder_yaw_joint": 5.0, ".*_elbow_joint": 5.0},
        armature=0.01,
    ),
})
