"""Train a PPO residual Cartesian policy with an IK low-level controller.

PPO owns the decisions: dx/dy/dz and gripper closure.  IK only translates the
Cartesian command into seven Franka joint targets.  The latch is a passive
prismatic mechanism, so success means moving its joint rather than moving the
robot hand to a scripted waypoint.
"""
import math
import os
import numpy as np

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

from isaacgym import gymapi, gymtorch, gymutil
import torch
from actor_critic import PPOAgent


def quat_conjugate(q):
    result = q.clone()
    result[:, :3] *= -1
    return result


def quat_multiply(a, b):
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack((
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ), dim=-1)


def orientation_error(desired, current):
    relative = quat_multiply(desired, quat_conjugate(current))
    return relative[:, :3] * torch.sign(relative[:, 3:4])


def damped_least_squares_ik(jacobian, dpose, damping=0.05):
    jt = jacobian.transpose(1, 2)
    regularizer = torch.eye(6, device=jacobian.device).unsqueeze(0) * damping ** 2
    matrix = jacobian @ jt + regularizer
    # torch.solve is retained for the Torch version shipped with Isaac Gym.
    solution = torch.solve(dpose.detach().cpu(), matrix.detach().cpu()).solution
    return (jt @ solution.to(jacobian.device)).squeeze(-1)


def compute_gae(rewards, values, dones, bootstrap, gamma=0.99, lam=0.95):
    """GAE over [time, environment], including a nonterminal bootstrap value."""
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(bootstrap)
    next_value = bootstrap
    for t in reversed(range(rewards.shape[0])):
        alive = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * alive - values[t]
        gae = delta + gamma * lam * alive * gae
        advantages[t] = gae
        next_value = values[t]
    return advantages, advantages + values


np.random.seed(42)
torch.manual_seed(42)
gym = gymapi.acquire_gym()

parameters = [
    {"name": "--num_envs", "type": int, "default": 16},
    {"name": "--episode_length", "type": int, "default": 300},
    {"name": "--rollout_length", "type": int, "default": 256},
    {"name": "--lift_curriculum_fraction", "type": float, "default": 0.25},
    {"name": "--approach_assist", "type": float, "default": 0.8},
    {"name": "--lift_assist", "type": float, "default": 1.0},
    {"name": "--eval", "action": "store_true"},
    {"name": "--checkpoint", "type": str, "default": "ppo_latch_desk_contact_v8.pt"},
    {"name": "--debug_print", "action": "store_true"},
    {"name": "--headless", "action": "store_true"},
]
args = gymutil.parse_arguments(
    description="Franka PPO Cartesian residual + IK latch slider", custom_parameters=parameters
)
if not torch.cuda.is_available():
    if args.use_gpu_pipeline or args.use_gpu or not args.headless:
        print("No CUDA GPU is available; using CPU physics.")
    args.use_gpu_pipeline = False
    args.use_gpu = False
    args.sim_device = "cpu"
    args.compute_device_id = -1
    args.graphics_device_id = -1
    if not args.headless:
        raise SystemExit(
            "Cannot open the Isaac Gym viewer because no CUDA-capable GPU is "
            "visible. Run with --headless, or fix NVIDIA/CUDA visibility so "
            "torch.cuda.device_count() is greater than 0."
        )
device = torch.device(args.sim_device if args.use_gpu_pipeline else "cpu")

# Observation includes contact-unlocked slide and post-release lift state.
obs_dim, action_dim = 28, 4
agent = PPOAgent(obs_dim, action_dim, device)
if os.path.isfile(args.checkpoint):
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])
        print(f"Loaded checkpoint: {args.checkpoint}")
    except (KeyError, RuntimeError) as error:
        print(f"Ignoring incompatible checkpoint {args.checkpoint}: {error}")

sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0, 0, -9.81)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
if args.physics_engine != gymapi.SIM_PHYSX:
    raise RuntimeError("This task requires PhysX")
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 16
sim_params.physx.num_velocity_iterations = 4
sim_params.physx.contact_offset = 0.005
sim_params.physx.rest_offset = 0.0
sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

sim = gym.create_sim(
    args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params
)
if sim is None:
    raise RuntimeError("Could not create simulation")

viewer = None if args.headless else gym.create_viewer(sim, gymapi.CameraProperties())
# A soft vertical light keeps the latch's shadow directly underneath it.  The
# default diagonal light displaced the shadow enough to make a grounded latch
# look as if it were floating above the table.
gym.set_light_parameters(
    sim,
    0,
    gymapi.Vec3(0.35, 0.35, 0.35),
    gymapi.Vec3(0.75, 0.75, 0.75),
    gymapi.Vec3(0.0, 0.0, -1.0),
)
asset_root = os.environ.get(
    "ISAACGYM_ASSET_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(gymapi.__file__), "../../assets")),
)

table_options = gymapi.AssetOptions()
table_options.fix_base_link = True
table_asset = gym.create_box(sim, 0.6, 1.0, 0.4, table_options)

franka_options = gymapi.AssetOptions()
franka_options.fix_base_link = True
franka_options.disable_gravity = True
franka_options.flip_visual_attachments = True
franka_options.armature = 0.01
franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
franka_asset_path = os.path.join(asset_root, franka_asset_file)
if not os.path.isfile(franka_asset_path):
    raise FileNotFoundError(f"Missing Franka asset: {franka_asset_path}")
franka_asset = gym.load_asset(
    sim, asset_root, franka_asset_file, franka_options
)

latch_options = gymapi.AssetOptions()
latch_options.fix_base_link = False  # Entire lightweight latch assembly rests freely on desk.
latch_options.disable_gravity = False
latch_options.use_mesh_materials = True
latch_options.vhacd_enabled = False
latch_asset_file = "urdf/think_shell_slider.urdf"
latch_asset_path = os.path.join(asset_root, latch_asset_file)
if not os.path.isfile(latch_asset_path):
    raise FileNotFoundError(f"Missing latch asset: {latch_asset_path}")
latch_asset = gym.load_asset(sim, asset_root, latch_asset_file, latch_options)

# Pinching a light plastic part needs enough tangential friction to carry its
# weight. Explicit values avoid backend/default-material differences.
table_shape_props = gym.get_asset_rigid_shape_properties(table_asset)
for prop in table_shape_props:
    prop.friction = 0.9
    prop.restitution = 0.0
gym.set_asset_rigid_shape_properties(table_asset, table_shape_props)
franka_shape_props = gym.get_asset_rigid_shape_properties(franka_asset)
for prop in franka_shape_props:
    prop.friction = 1.5
    prop.restitution = 0.0
gym.set_asset_rigid_shape_properties(franka_asset, franka_shape_props)
latch_shape_props = gym.get_asset_rigid_shape_properties(latch_asset)
for prop in latch_shape_props:
    prop.friction = 1.5
    prop.restitution = 0.0
gym.set_asset_rigid_shape_properties(latch_asset, latch_shape_props)

franka_props = gym.get_asset_dof_properties(franka_asset)
franka_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
franka_props["stiffness"][:7].fill(500.0)
franka_props["damping"][:7].fill(60.0)
franka_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
franka_props["stiffness"][7:].fill(80.0)
franka_props["damping"][7:].fill(15.0)

latch_props = gym.get_asset_dof_properties(latch_asset)
latch_props["driveMode"].fill(gymapi.DOF_MODE_POS)
latch_props["stiffness"].fill(600.0)
latch_props["damping"].fill(60.0)
latch_props["friction"].fill(0.10)
# Initially both axes are locked. Contact unlocks slide; release unlocks lift.

default_franka_pos = np.array(
    [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
    dtype=np.float32,
)
default_franka_state = np.zeros(9, dtype=gymapi.DofState.dtype)
default_franka_state["pos"] = default_franka_pos
default_latch_state = np.zeros(2, dtype=gymapi.DofState.dtype)

table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.5, 0.0, 0.2)
table_top = 0.4

# Restore the bright checkerboard floor used by the original example.  Without
# a ground plane the viewer clears uncovered pixels to black between env tables.
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

latch_pose = gymapi.Transform()
# The latch is now the dynamic root body. Spawn it slightly above the tabletop
# and let gravity/contact determine its resting pose during the settle frames.
latch_spawn_clearance = 0.060
latch_pose.p = gymapi.Vec3(0.55, 0.0, table_top + latch_spawn_clearance)
franka_pose = gymapi.Transform()

num_envs = args.num_envs
num_per_row = max(1, int(math.sqrt(num_envs)))
env_lower = gymapi.Vec3(-0.9, -0.9, 0)
env_upper = gymapi.Vec3(0.9, 0.9, 0.9)
envs, latch_handles, franka_handles = [], [], []
hand_indices, latch_body_indices = [], []
left_finger_indices, right_finger_indices = [], []
slide_dof_indices, lift_dof_indices, franka_dof_indices = [], [], []
latch_actor_indices, franka_actor_indices = [], []

for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)
    gym.set_rigid_body_color(
        env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
        gymapi.Vec3(0.72, 0.76, 0.82),
    )
    latch_handle = gym.create_actor(env, latch_asset, latch_pose, "latch_mechanism", i, 0)
    gym.set_rigid_body_color(
        env, latch_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
        gymapi.Vec3(0.05, 0.95, 0.12),
    )
    gym.set_actor_dof_properties(env, latch_handle, latch_props)
    gym.set_actor_dof_states(env, latch_handle, default_latch_state, gymapi.STATE_ALL)
    latch_handles.append(latch_handle)
    latch_actor_indices.append(gym.get_actor_index(env, latch_handle, gymapi.DOMAIN_SIM))

    franka_handle = gym.create_actor(env, franka_asset, franka_pose, "franka", i, 2)
    gym.set_actor_dof_properties(env, franka_handle, franka_props)
    gym.set_actor_dof_states(env, franka_handle, default_franka_state, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, franka_handle, default_franka_pos)
    franka_handles.append(franka_handle)
    franka_actor_indices.append(gym.get_actor_index(env, franka_handle, gymapi.DOMAIN_SIM))

    hand_indices.append(gym.find_actor_rigid_body_index(
        env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM
    ))
    left_finger_indices.append(gym.find_actor_rigid_body_index(
        env, franka_handle, "panda_leftfinger", gymapi.DOMAIN_SIM
    ))
    right_finger_indices.append(gym.find_actor_rigid_body_index(
        env, franka_handle, "panda_rightfinger", gymapi.DOMAIN_SIM
    ))
    latch_body_indices.append(gym.find_actor_rigid_body_index(
        env, latch_handle, "latch", gymapi.DOMAIN_SIM
    ))
    slide_dof_indices.append(gym.get_actor_dof_index(
        env, latch_handle, 0, gymapi.DOMAIN_SIM
    ))
    lift_dof_indices.append(gym.get_actor_dof_index(
        env, latch_handle, 1, gymapi.DOMAIN_SIM
    ))
    franka_dof_indices.append([
        gym.get_actor_dof_index(env, franka_handle, j, gymapi.DOMAIN_SIM) for j in range(9)
    ])

if viewer is not None:
    gym.viewer_camera_look_at(
        viewer, envs[0], gymapi.Vec3(1.35, 0.9, 0.9), gymapi.Vec3(0.5, 0, 0.4)
    )

gym.prepare_sim(sim)
rb_states = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
root_states = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
dof_states = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
contact_forces = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim))
jacobian = gymtorch.wrap_tensor(gym.acquire_jacobian_tensor(sim, "franka"))
hand_link = gym.get_asset_rigid_body_dict(franka_asset)["panda_hand"]
j_eef = jacobian[:, hand_link - 1, :, :7]

hand_indices = torch.tensor(hand_indices, dtype=torch.long, device=device)
latch_body_indices = torch.tensor(latch_body_indices, dtype=torch.long, device=device)
left_finger_indices = torch.tensor(left_finger_indices, dtype=torch.long, device=device)
right_finger_indices = torch.tensor(right_finger_indices, dtype=torch.long, device=device)
slide_dof_indices = torch.tensor(slide_dof_indices, dtype=torch.long, device=device)
lift_dof_indices = torch.tensor(lift_dof_indices, dtype=torch.long, device=device)
franka_dof_indices = torch.tensor(franka_dof_indices, dtype=torch.long, device=device)
latch_actor_indices = torch.tensor(latch_actor_indices, dtype=torch.int32, device=device)
franka_actor_indices = torch.tensor(franka_actor_indices, dtype=torch.int32, device=device)

num_sim_dofs = gym.get_sim_dof_count(sim)
dof_targets = torch.zeros(num_sim_dofs, dtype=torch.float32, device=device)
default_franka_tensor = torch.tensor(default_franka_pos, device=device).repeat(num_envs, 1)
dof_targets[franka_dof_indices] = default_franka_tensor

down_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(num_envs, 1)
previous_action = torch.zeros((num_envs, action_dim), device=device)
episode_step = torch.zeros(num_envs, dtype=torch.long, device=device)
episode_return = torch.zeros(num_envs, device=device)
contact_streak = torch.zeros(num_envs, dtype=torch.long, device=device)
grasp_latched = torch.zeros(num_envs, dtype=torch.bool, device=device)
grasp_hold_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
slide_unlocked = torch.zeros(num_envs, dtype=torch.bool, device=device)
released = torch.zeros(num_envs, dtype=torch.bool, device=device)
previous_hand_pos = torch.zeros((num_envs, 3), device=device)
eef_target = torch.zeros((num_envs, 3), device=device)
gripper_target = torch.tensor(franka_props["upper"][7:9], device=device).repeat(num_envs, 1)
completed_episodes = 0
ppo_updates = 0
control_decimation = 4
action_smoothing = 0.65
release_threshold = 0.010
lift_success_height = 0.030
free_object_mode = True


def refresh():
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_net_contact_force_tensor(sim)


def enforce_latch_constraints():
    """Hard-clamp every axis that is locked in the current task stage."""
    slide_locked = (~slide_unlocked) | released
    # A released latch may be in the air only while a stable bilateral grasp
    # is maintained. Losing the grasp returns it to its desk-height target.
    lift_locked = (~released) | (released & (contact_streak < 2))

    dof_states[slide_dof_indices[slide_locked], 0] = dof_targets[
        slide_dof_indices[slide_locked]
    ]
    dof_states[slide_dof_indices[slide_locked], 1] = 0
    dof_states[lift_dof_indices[lift_locked], 0] = dof_targets[
        lift_dof_indices[lift_locked]
    ]
    dof_states[lift_dof_indices[lift_locked], 1] = 0

    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_states))


def observe():
    hand = rb_states[hand_indices]
    latch = rb_states[latch_body_indices]
    left_finger = rb_states[left_finger_indices]
    right_finger = rb_states[right_finger_indices]
    finger_midpoint = 0.5 * (left_finger[:, :3] + right_finger[:, :3])
    hand_velocity = hand[:, 7:10].clamp(-1, 1)
    slide = dof_states[slide_dof_indices]
    lift = dof_states[lift_dof_indices]
    fingers = dof_states[franka_dof_indices[:, 7], 0] + dof_states[franka_dof_indices[:, 8], 0]
    left_distance = torch.norm(latch[:, :3] - left_finger[:, :3], dim=-1)
    right_distance = torch.norm(latch[:, :3] - right_finger[:, :3], dim=-1)
    latch_horizontal_force = torch.norm(contact_forces[latch_body_indices, :2], dim=-1)
    latch_is_pushed = latch_horizontal_force > 0.05
    left_contact = (
        (torch.norm(contact_forces[left_finger_indices], dim=-1) > 0.05)
        & (left_distance < 0.115) & latch_is_pushed
    ).float()
    right_contact = (
        (torch.norm(contact_forces[right_finger_indices], dim=-1) > 0.05)
        & (right_distance < 0.115) & latch_is_pushed
    ).float()
    episode_fraction = episode_step.float() / float(args.episode_length)
    return torch.cat((
        hand[:, :3] / 0.8,
        hand[:, 3:7],
        hand_velocity,
        (latch[:, :3] - finger_midpoint) / 0.30,
        slide[:, 0:1] / 0.055,
        slide[:, 1:2] / 0.20,
        (latch[:, 2:3] - initial_latch_body_z.unsqueeze(-1)) / 0.035,
        latch[:, 9:10] / 0.10,
        fingers.unsqueeze(-1) / 0.08,
        left_contact.unsqueeze(-1),
        right_contact.unsqueeze(-1),
        gripper_target.mean(-1, keepdim=True) / 0.04,
        previous_action,
        episode_fraction.unsqueeze(-1),
        slide_unlocked.float().unsqueeze(-1),
        released.float().unsqueeze(-1),
    ), dim=-1)


def configure_latch_stage(env_id, stage):
    """Set stage 0=locked, 1=contact-unlocked slide, 2=released lift."""
    props = gym.get_actor_dof_properties(envs[env_id], latch_handles[env_id])
    if stage == 2:
        # Hold the completed slide and make vertical motion passive.
        props["driveMode"][0] = gymapi.DOF_MODE_POS
        props["stiffness"][0] = 600.0
        props["damping"][0] = 60.0
        props["driveMode"][1] = gymapi.DOF_MODE_NONE
        props["stiffness"][1] = 0.0
        props["damping"][1] = 0.08
    elif stage == 1:
        # A real two-finger contact makes horizontal sliding passive.
        props["driveMode"][0] = gymapi.DOF_MODE_NONE
        props["stiffness"][0] = 0.0
        props["damping"][0] = 0.15
        props["driveMode"][1] = gymapi.DOF_MODE_POS
        props["stiffness"][1] = 600.0
        props["damping"][1] = 60.0
    else:
        # No contact: neither solver impulses nor gravity may move the latch.
        props["driveMode"][:2].fill(gymapi.DOF_MODE_POS)
        props["stiffness"][:2].fill(600.0)
        props["damping"][:2].fill(60.0)
    gym.set_actor_dof_properties(envs[env_id], latch_handles[env_id], props)


def reset(indices):
    global completed_episodes
    if indices.numel() == 0:
        return
    reset_positions = default_franka_tensor[indices].clone()
    reset_positions[:, :7] += torch.empty_like(reset_positions[:, :7]).uniform_(-0.025, 0.025)
    dof_states[franka_dof_indices[indices], 0] = reset_positions
    dof_states[franka_dof_indices[indices], 1] = 0
    start_released = torch.zeros(indices.numel(), dtype=torch.bool, device=device)
    if (not free_object_mode and args.headless and not args.eval
            and args.lift_curriculum_fraction > 0):
        start_released = (
            torch.rand(indices.numel(), device=device) < args.lift_curriculum_fraction
        )

    dof_states[slide_dof_indices[indices], :] = 0
    dof_states[lift_dof_indices[indices], :] = 0
    dof_states[slide_dof_indices[indices], 0] = start_released.float() * 0.055
    dof_targets[franka_dof_indices[indices]] = reset_positions
    dof_targets[slide_dof_indices[indices]] = start_released.float() * 0.055
    dof_targets[lift_dof_indices[indices]] = 0
    env_ids = indices.detach().cpu().tolist()
    stages = start_released.detach().cpu().tolist()
    for idx, start_in_lift_stage in zip(env_ids, stages):
        configure_latch_stage(idx, 2 if start_in_lift_stage else 0)

    latch_roots = latch_actor_indices[indices].long()
    root_states[latch_roots] = initial_latch_root_states[indices]
    # Full tensor setters are less brittle than the indexed setters on CPU
    # PhysX when CUDA is unavailable or has fallen back after initialization.
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_states))
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states))
    previous_action[indices] = 0
    episode_step[indices] = 0
    episode_return[indices] = 0
    contact_streak[indices] = 0
    grasp_latched[indices] = False
    grasp_hold_steps[indices] = 0
    slide_unlocked[indices] = start_released
    released[indices] = start_released
    gripper_target[indices] = torch.tensor(franka_props["upper"][7:9], device=device)
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(dof_targets))
    eef_target[indices] = initial_hand_pos[indices]
    previous_hand_pos[indices] = eef_target[indices]
    completed_episodes += indices.numel()


# Settle once, then initialize Cartesian targets from the true simulated hand pose.
gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(dof_targets))
for _ in range(30 if free_object_mode else 4):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
refresh()
eef_target.copy_(rb_states[hand_indices, :3])
previous_hand_pos.copy_(eef_target)
initial_hand_pos = eef_target.clone()
initial_latch_root_states = root_states[latch_actor_indices.long()].clone()
initial_latch_body_z = rb_states[latch_body_indices, 2].clone()

buffer = {key: [] for key in ("obs", "actions", "rewards", "values", "log_probs", "dones")}
policy_steps = 0

while viewer is None or not gym.query_viewer_has_closed(viewer):
    obs = observe()
    raw_action, log_prob, value = agent.get_action(obs, deterministic=args.eval)
    # Low-pass the sampled command.  Exploration remains stochastic, but a
    # single noisy PPO sample can no longer jerk the Cartesian/gripper target.
    action = action_smoothing * previous_action + (1.0 - action_smoothing) * torch.tanh(raw_action)

    # PPO supplies residual Cartesian motion. During training, a small IK
    # approach prior brings the fingertips into the trial region; PPO still
    # decides residual XYZ corrections and when/how far to close the gripper.
    ppo_delta = action[:, :3] * torch.tensor([0.012, 0.012, 0.008], device=device)
    current_grip_width = (
        dof_states[franka_dof_indices[:, 7], 0]
        + dof_states[franka_dof_indices[:, 8], 0]
    )
    # The collision geometry physically stops the measured finger DOFs near
    # 0.049 m total even though their commanded target is tighter (0.024 m).
    # That separation is evidence of a squeezed object, not a loose gripper.
    lift_ready = grasp_latched & (grasp_hold_steps >= 8) & (current_grip_width < 0.055)
    # Hold the hand still briefly while the fingers squeeze around the part.
    squeezing = grasp_latched & (~lift_ready)
    ppo_delta[squeezing] = 0.0
    # Do not let one exploratory downward action cancel a confirmed lift.
    ppo_delta[:, 2] = torch.where(
        lift_ready, torch.relu(ppo_delta[:, 2]), ppo_delta[:, 2]
    )
    approach_delta = torch.zeros_like(ppo_delta)
    if args.approach_assist > 0:
        finger_midpoint_now = 0.5 * (
            rb_states[left_finger_indices, :3] + rb_states[right_finger_indices, :3]
        )
        precontact_target = rb_states[latch_body_indices, :3].clone()
        precontact_target[:, 2] += 0.025
        approach_error = precontact_target - finger_midpoint_now
        approach_norm = torch.norm(approach_error, dim=-1, keepdim=True).clamp(min=1e-6)
        approach_delta = approach_error * torch.clamp(0.008 / approach_norm, max=1.0)
        approach_delta *= args.approach_assist * (contact_streak < 2).float().unsqueeze(-1)
    # Once PPO has produced a real, closed bilateral grasp, bias the Cartesian
    # target upward by 2.5 mm per policy step. This moves only the robot; the
    # free latch rises solely if finger friction/contact is actually holding it.
    lift_delta = torch.zeros_like(ppo_delta)
    below_lift_goal = (
        rb_states[latch_body_indices, 2] - initial_latch_body_z < lift_success_height + 0.015
    )
    lift_delta[:, 2] = (
        0.0030 * args.lift_assist * (lift_ready & below_lift_goal).float()
    )
    eef_target += ppo_delta + approach_delta + lift_delta
    eef_target[:, 0].clamp_(0.20, 0.75)
    eef_target[:, 1].clamp_(-0.40, 0.40)
    eef_target[:, 2].clamp_(table_top + 0.018, 0.82)

    # Stateful PPO gripper: +1 closes a little, -1 opens a little.  Incremental
    # control lets the policy try, release, reposition, and try again.
    finger_upper = torch.tensor(franka_props["upper"][7:9], device=device)
    gripper_target -= action[:, 3:4] * 0.003
    gripper_target.copy_(torch.maximum(torch.minimum(gripper_target, finger_upper), torch.zeros_like(gripper_target)))
    # Preserve the grasp during the short assisted lift instead of allowing the
    # next exploratory sample to fully reopen the fingers.
    gripper_target.copy_(torch.where(
        grasp_latched.unsqueeze(-1),
        torch.minimum(gripper_target, torch.full_like(gripper_target, 0.012)),
        gripper_target,
    ))
    dof_targets[franka_dof_indices[:, 7:9]] = gripper_target

    stage_was_released = released.clone()
    old_slide = dof_states[slide_dof_indices, 0].clone()
    old_lift = dof_states[lift_dof_indices, 0].clone()
    old_latch_z = rb_states[latch_body_indices, 2].clone()
    old_finger_midpoint = 0.5 * (
        rb_states[left_finger_indices, :3] + rb_states[right_finger_indices, :3]
    )
    old_reach_distance = torch.norm(
        old_finger_midpoint - rb_states[latch_body_indices, :3], dim=-1
    )
    lower = torch.tensor(franka_props["lower"][:7], device=device)
    upper = torch.tensor(franka_props["upper"][:7], device=device)
    for _ in range(control_decimation):
        # Run the IK servo at every 60 Hz physics frame, like the smooth Isaac
        # Gym IK example, while PPO changes its command only every four frames.
        hand = rb_states[hand_indices]
        dpose = torch.cat(
            (eef_target - hand[:, :3], orientation_error(down_q, hand[:, 3:7])), -1
        )
        dpose[:, :3].clamp_(-0.04, 0.04)
        arm_delta = damped_least_squares_ik(j_eef, dpose.unsqueeze(-1))
        arm_now = dof_states[franka_dof_indices[:, :7], 0]
        arm_target = arm_now + arm_delta
        dof_targets[franka_dof_indices[:, :7]] = torch.maximum(
            torch.minimum(arm_target, upper), lower
        )
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(dof_targets))
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        refresh()
        enforce_latch_constraints()
        if viewer is not None:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False)
            gym.sync_frame_time(sim)
    new_slide = dof_states[slide_dof_indices, 0]
    new_lift = dof_states[lift_dof_indices, 0]
    latch_pos = rb_states[latch_body_indices, :3]
    hand_pos = rb_states[hand_indices, :3]
    left_finger_pos = rb_states[left_finger_indices, :3]
    right_finger_pos = rb_states[right_finger_indices, :3]
    finger_midpoint = 0.5 * (left_finger_pos + right_finger_pos)
    reach_distance = torch.norm(finger_midpoint - latch_pos, dim=-1)
    left_finger_distance = torch.norm(left_finger_pos - latch_pos, dim=-1)
    right_finger_distance = torch.norm(right_finger_pos - latch_pos, dim=-1)
    left_force = torch.norm(contact_forces[left_finger_indices], dim=-1)
    right_force = torch.norm(contact_forces[right_finger_indices], dim=-1)
    # GPU-safe contact discrimination: table contact can create finger force,
    # but it cannot count unless the latch itself receives horizontal force.
    latch_horizontal_force = torch.norm(contact_forces[latch_body_indices, :2], dim=-1)
    latch_is_pushed = latch_horizontal_force > 0.05
    left_contact = (
        (left_force > 0.05) & (left_finger_distance < 0.115) & latch_is_pushed
    )
    right_contact = (
        (right_force > 0.05) & (right_finger_distance < 0.115) & latch_is_pushed
    )
    bilateral_contact = left_contact & right_contact
    contact_streak = torch.where(
        bilateral_contact,
        contact_streak + 1,
        torch.clamp(contact_streak - 1, min=0),
    )
    # Contact can flicker after the latch leaves the table. Remember a verified
    # grasp while the fingers remain closed and geometrically around the latch.
    finger_width = (
        dof_states[franka_dof_indices[:, 7], 0]
        + dof_states[franka_dof_indices[:, 8], 0]
    )
    closed_around_latch = (
        (finger_width < 0.080)
        & (left_finger_distance < 0.125)
        & (right_finger_distance < 0.125)
    )
    squeeze_grasp = closed_around_latch & latch_is_pushed
    newly_grasped = ((bilateral_contact & (contact_streak >= 2)) | squeeze_grasp) & closed_around_latch
    grasp_latched = newly_grasped | (grasp_latched & closed_around_latch)
    grasp_hold_steps = torch.where(
        grasp_latched,
        grasp_hold_steps + 1,
        torch.zeros_like(grasp_hold_steps),
    )

    # The slider cannot move at all until bilateral contact is sustained.
    newly_slide_unlocked = (
        torch.zeros_like(released) if free_object_mode else
        ((~slide_unlocked) & (~released) & (contact_streak >= 2))
    )
    for idx in torch.nonzero(newly_slide_unlocked, as_tuple=False).squeeze(-1).detach().cpu().tolist():
        configure_latch_stage(idx, 1)
    slide_unlocked |= newly_slide_unlocked

    # If the fingers let go before release, freeze the exact current position.
    lost_slide_contact = slide_unlocked & (~released) & (~bilateral_contact)
    for idx in torch.nonzero(lost_slide_contact, as_tuple=False).squeeze(-1).detach().cpu().tolist():
        dof_targets[slide_dof_indices[idx]] = new_slide[idx]
        configure_latch_stage(idx, 0)
    slide_unlocked &= ~lost_slide_contact

    # Release only after a real grasp has moved the horizontal slider far
    # enough. Each environment transitions independently.
    newly_released = (
        slide_unlocked & (~released) & bilateral_contact
        & (new_slide >= release_threshold)
    )
    for idx in torch.nonzero(newly_released, as_tuple=False).squeeze(-1).detach().cpu().tolist():
        # Lock exactly where contact placed it; never motor it forward.
        dof_targets[slide_dof_indices[idx]] = new_slide[idx]
        configure_latch_stage(idx, 2)
    released |= newly_released

    slide_progress = new_slide - old_slide
    lift_progress = new_lift - old_lift
    world_lift_progress = latch_pos[:, 2] - old_latch_z
    contact_quality = 0.5 * (left_contact.float() + right_contact.float())
    positive_progress = torch.relu(slide_progress)
    # Sliding while held is valuable; moving it without finger contact is a
    # push/hack and is penalized instead of being mistaken for a grasp.
    slide_stage = (~stage_was_released).float()
    reward = 30.0 * slide_progress * slide_stage
    reward += 180.0 * positive_progress * contact_quality * slide_stage
    reward -= 100.0 * positive_progress * (1.0 - contact_quality) * slide_stage
    reward += 3.0 * newly_released.float()

    # After release, upward latch travel is the task. Reward lifting only while
    # the fingers maintain contact; dropping it loses progress reward.
    lift_stage = released.float()
    reward += 160.0 * torch.relu(lift_progress) * contact_quality * lift_stage
    reward -= 80.0 * torch.relu(-lift_progress) * lift_stage
    reward += 0.08 * bilateral_contact.float() * lift_stage
    # In free-object mode the whole latch, not an internal joint, is lifted.
    reward += 160.0 * torch.relu(world_lift_progress) * contact_quality
    reward -= 40.0 * torch.relu(-world_lift_progress) * contact_quality
    # Potential-based reach shaping tells PPO whether its last Cartesian trial
    # moved toward or away from the latch.  The former exp(-20*d) signal was
    # nearly zero at the initial 20-40 cm distance and allowed random wandering.
    reach_progress = old_reach_distance - reach_distance
    reward += 12.0 * reach_progress
    # Contact curriculum: first reach the grasp region, then close one finger,
    # then establish a stable two-finger grasp.
    grasp_region = reach_distance < 0.075
    reward += 0.015 * grasp_region.float()
    reward += 0.05 * (left_contact.float() + right_contact.float())
    reward += 0.20 * bilateral_contact.float()
    reward += 0.15 * grasp_latched.float()
    closed_fraction = 1.0 - (
        dof_states[franka_dof_indices[:, 7], 0]
        + dof_states[franka_dof_indices[:, 8], 0]
    ) / 0.08
    reward += 0.025 * closed_fraction.clamp(0, 1) * grasp_region.float()
    reward -= 0.03 * closed_fraction.clamp(0, 1) * (reach_distance > 0.12).float()
    excessive_finger_force = torch.relu(left_force - 30.0) + torch.relu(right_force - 30.0)
    reward -= 0.001 * excessive_finger_force
    reward -= 0.002 * action[:, :3].square().sum(-1)
    reward -= 0.02 * torch.relu(-slide_progress) * 120.0 * slide_stage

    episode_step += 1
    object_height = latch_pos[:, 2] - initial_latch_body_z
    holding_for_success = grasp_latched if free_object_mode else (contact_streak >= 2)
    success = (object_height >= lift_success_height) & holding_for_success
    timeout = episode_step >= args.episode_length
    done = success | timeout
    reward += 10.0 * success.float()
    reward -= 1.0 * (timeout & ~success).float()
    episode_return += reward

    buffer["obs"].append(obs.detach())
    buffer["actions"].append(raw_action.detach())
    buffer["rewards"].append(reward.detach())
    buffer["values"].append(value.detach())
    buffer["log_probs"].append(log_prob.detach())
    buffer["dones"].append(done.float().detach())
    previous_action.copy_(action)
    previous_hand_pos.copy_(hand_pos)
    policy_steps += 1

    if args.debug_print and policy_steps % 30 == 0:
        active_grasps = grasp_latched if free_object_mode else slide_unlocked
        locked_slide = (~slide_unlocked) | released
        max_locked_error = torch.abs(
            dof_states[slide_dof_indices[locked_slide], 0]
            - dof_targets[slide_dof_indices[locked_slide]]
        ).max().item() if bool(locked_slide.any()) else 0.0
        print(
            f"step={policy_steps} slide_unlocked={bool(slide_unlocked[0])} "
            f"released={bool(released[0])} "
            f"slide={new_slide[0].item():.4f} lift={new_lift[0].item():.4f} "
            f"latch_z={latch_pos[0, 2].item():.4f} table_z={table_top:.4f} "
            f"reach={reach_distance[0].item():.3f} "
            f"finger_d=({left_finger_distance[0].item():.3f},{right_finger_distance[0].item():.3f}) "
            f"grip_width={finger_width[0].item():.3f} "
            f"force=({left_force[0].item():.2f},{right_force[0].item():.2f}) "
            f"latch_xy_force={latch_horizontal_force[0].item():.2f} "
            f"contact=({bool(left_contact[0])},{bool(right_contact[0])}) "
            f"grasped={bool(grasp_latched[0])} "
            f"hold={int(grasp_hold_steps[0].item())} ready={bool(lift_ready[0])} "
            f"active={int(active_grasps.sum().item())}/{num_envs} "
            f"released_n={int(released.sum().item())} "
            f"max_lift={new_lift.max().item():.4f} "
            f"obj_h={object_height[0].item():.4f} "
            f"locked_err={max_locked_error:.6f} "
            f"reward={reward[0].item():.3f} return={episode_return[0].item():.2f}"
        )

    reset(torch.nonzero(done, as_tuple=False).squeeze(-1))

    if not args.eval and len(buffer["obs"]) >= args.rollout_length:
        with torch.no_grad():
            bootstrap = agent.get_value(observe())
        rewards = torch.stack(buffer["rewards"])
        values = torch.stack(buffer["values"])
        dones = torch.stack(buffer["dones"])
        advantages, returns = compute_gae(rewards, values, dones, bootstrap)

        agent.update(
            torch.stack(buffer["obs"]).flatten(0, 1),
            torch.stack(buffer["actions"]).flatten(0, 1),
            returns.flatten(),
            advantages.flatten(),
            torch.stack(buffer["log_probs"]).flatten(),
        )
        ppo_updates += 1
        torch.save(
            {"actor": agent.actor.state_dict(), "critic": agent.critic.state_dict()},
            args.checkpoint,
        )
        buffer = {key: [] for key in buffer}
        if args.debug_print:
            print(f">>> PPO update at policy step {policy_steps}; episodes={completed_episodes}")

if viewer is not None:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
