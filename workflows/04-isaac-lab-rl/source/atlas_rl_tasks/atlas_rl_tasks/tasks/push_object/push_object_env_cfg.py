"""Manager-based env cfg: a Unitree G1 humanoid pushes a puck across a table with its left arm.

Rebuilds the "push donut" task from tutorial #14 (Custom Humanoid Task Training in Isaac Lab). The transcript lists
what the env file holds (08:26-09:16): "the robot configuration, the object USD file, the table position, the initial
object position, and the target push distance"; a scene with "the table, the object, and the robot"; actions on "the
seven joints of the robot's left arm"; observations "arm joint positions, joint velocities, wrist position, and object
position"; and the three-part reward. Events (domain randomization) and commands come from the ``isaaclab.sh --new``
template the tutorial generates (05:15-06:30) and follow the NVIDIA Developer Blog loop in the evidence image:
randomize mass, friction and gains, and add observation noise.

Differences from the tutorial, stated plainly:
- The kitchen assets in the video come from a sponsor's Hugging Face repo whose name the captions garble. This cfg uses
  a primitive cylinder ("puck") and a cuboid table so it runs without that download. Swap ``OBJECT_SPAWN`` for a
  ``sim_utils.UsdFileCfg(usd_path=...)`` to use a SimReady asset.
- The tutorial's robot USDs are the author's own (captioned "Go1 humanoid", likely G1). This cfg uses Isaac Lab's
  ``G1_MINIMAL_CFG``. Its arm has 5 joints per side; Unitree's 29-DoF G1 has the 7 the tutorial mentions. The joint
  regex below matches both.
- Numbers (table height, reach envelope, weights) are starting values that were never trained. Tune on the GPU box.

Pinned: Isaac Sim 5.0.0, Isaac Lab v2.2.x (see ../../../../../README.md).
not executed: needs GPU (Isaac Lab, Isaac Sim 5.0, RTX GPU).
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip

from . import mdp

# ---------------------------------------------------------------------------------------------------------------------
# Task constants (env-local frame: robot root at the origin, +x forward, z up; Isaac Lab quaternions are w, x, y, z)
# ---------------------------------------------------------------------------------------------------------------------
LEFT_ARM_JOINTS = ["left_shoulder_.*", "left_elbow_.*", "left_wrist_.*"]  # 5 joints on G1_MINIMAL_CFG, 7 on 29-DoF G1
WRIST_BODY = "left_palm_link"  # verify: body name in G1_MINIMAL_CFG's USD (29-DoF G1: "left_wrist_yaw_link")
ROOT_Z = 0.74  # G1_MINIMAL_CFG init_state.pos z  # verify against isaaclab_assets.robots.unitree

TABLE_TOP_Z = 0.70  # verify: inside the left hand's reach with the pelvis fixed at ROOT_Z
TABLE_CENTER_X = 0.47
TABLE_SIZE = (0.50, 0.90, TABLE_TOP_Z)

PUCK_RADIUS = 0.045
PUCK_HEIGHT = 0.035
OBJECT_INIT = (0.32, 0.20, TABLE_TOP_Z + PUCK_HEIGHT / 2)  # in front of the left shoulder

# "target push distance": the command samples a target 0.10-0.18 m beyond the start, on the table.
TARGET_X = (0.42, 0.50)
TARGET_Y = (0.15, 0.25)
TARGET_Z_B = OBJECT_INIT[2] - ROOT_Z  # target z in the root frame: puck centre height

SUCCESS_RADIUS = 0.03  # m, planar
EPISODE_S = 8.0

OBJECT_SPAWN = sim_utils.CylinderCfg(
    radius=PUCK_RADIUS,
    height=PUCK_HEIGHT,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.55, 0.20)),
)

ARM = SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINTS)
WRIST = SceneEntityCfg("robot", body_names=WRIST_BODY)
OBJ = SceneEntityCfg("object")


# ---------------------------------------------------------------------------------------------------------------------
# Scene: ground, light, robot, table, object
# ---------------------------------------------------------------------------------------------------------------------
@configclass
class PushObjectSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))

    # Articulation root on the robot's top-level prim; the fixed root keeps the humanoid standing while the arm learns.
    robot: ArticulationCfg = G1_MINIMAL_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=G1_MINIMAL_CFG.spawn.replace(
            articulation_props=G1_MINIMAL_CFG.spawn.articulation_props.replace(fix_root_link=True)  # verify: field
        ),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.35, 0.25)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(TABLE_CENTER_X, 0.0, TABLE_TOP_Z / 2)),
    )

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=OBJECT_SPAWN,
        init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_INIT),
    )


# ---------------------------------------------------------------------------------------------------------------------
# MDP
# ---------------------------------------------------------------------------------------------------------------------
@configclass
class CommandsCfg:
    """Target position for the object, in the robot root frame; resampled once per episode."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=WRIST_BODY,
        resampling_time_range=(EPISODE_S + 2.0, EPISODE_S + 2.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=TARGET_X,
            pos_y=TARGET_Y,
            pos_z=(TARGET_Z_B, TARGET_Z_B),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Joint position targets for the left arm (tutorial #14: "the seven joints of the robot's left arm")."""

    left_arm = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=LEFT_ARM_JOINTS, scale=0.5, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        arm_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": ARM}, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        arm_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": ARM}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        wrist_pos = ObsTerm(func=mdp.wrist_pos_in_root_frame, params={"asset_cfg": WRIST})
        object_pos = ObsTerm(func=mdp.object_pos_in_root_frame, noise=Unoise(n_min=-0.005, n_max=0.005))
        target = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True  # "Add Noise" in the evidence diagram
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Resets and domain randomization ("Add Randomization": mass, rigid-body properties, gains)."""

    # startup: once per environment when the sim starts
    object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": OBJ, "mass_distribution_params": (0.5, 2.0), "operation": "scale"},
    )
    object_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": OBJ,
            "static_friction_range": (0.3, 0.9),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    # reset: every episode
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.04, 0.04), "y": (-0.04, 0.04), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
            "asset_cfg": OBJ,
        },
    )
    reset_arm = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"asset_cfg": ARM, "position_range": (-0.1, 0.1), "velocity_range": (0.0, 0.0)},
    )
    arm_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": ARM,
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )


@configclass
class RewardsCfg:
    """Tutorial #14's three terms plus two small smoothness penalties. Weights are per second (scaled by step_dt)."""

    # (1) dense: wrist closer to the object, in (0, 1]
    reach = RewTerm(func=mdp.reach_object, weight=1.0, params={"std": 0.1, "robot_cfg": WRIST})
    # (2) dense: per-step decrease of the object-to-target distance (metres); 0.15 m of progress pays 1.5 over the run
    progress = RewTerm(func=mdp.ObjectGoalProgress, weight=500.0, params={"command_name": "object_pose"})
    # (3) sparse: paid every step the object sits at the target, so success outpays hovering at the object (5 vs 1 /s)
    success = RewTerm(
        func=mdp.object_at_goal, weight=5.0, params={"command_name": "object_pose", "threshold": SUCCESS_RADIUS}
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-3)
    arm_joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4, params={"asset_cfg": ARM})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropped = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": TABLE_TOP_Z - 0.10, "asset_cfg": OBJ}
    )
    object_out_of_reach = DoneTerm(func=mdp.object_out_of_reach, params={"max_distance": 0.75})
    # No success termination while training: ending the episode on success would forfeit the remaining per-step
    # reward and teach the policy to avoid the goal. mdp.object_reached_goal exists for evaluation.


# ---------------------------------------------------------------------------------------------------------------------
# Env cfgs
# ---------------------------------------------------------------------------------------------------------------------
@configclass
class PushObjectEnvCfg(ManagerBasedRLEnvCfg):
    scene: PushObjectSceneCfg = PushObjectSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2  # 100 Hz physics, 50 Hz policy
        self.episode_length_s = EPISODE_S
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.viewer.eye = (1.8, 1.8, 1.6)
        self.viewer.lookat = (0.3, 0.2, 0.7)


@configclass
class PushObjectEnvCfg_PLAY(PushObjectEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.object_mass = None
        self.events.object_friction = None
        self.events.arm_gains = None
