"""
Franka latch pick debug script for Isaac Gym.
Soft Actor and PPO will Learn Grasping Policy for Robust Grasp
IK-only first:
1) target is think_shell, not cube
2) latch_z_offset = 0.24 for visible latch
3) lift enabled after grasp
4) tune approach_offset only
"""

import os
import math
import numpy as np

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch

try:
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_nvfuser_enabled(False)
except Exception:
    pass


def quat_conjugate_local(q):
    out = q.clone()
    out[:, 0:3] *= -1.0
    return out


def quat_mul_local(a, b):
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    w = aw * bw - ax * bx - ay * by - az * bz
    return torch.stack((x, y, z, w), dim=-1)


def orientation_error(desired, current):
    cc = quat_conjugate_local(current)
    q_r = quat_mul_local(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def control_ik(dpose):
    global damping, j_eef, num_envs, device
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device).unsqueeze(0) * (damping ** 2)
    A = j_eef @ j_eef_T + lmbda

    # CPU solve avoids RTX 40xx + torch 1.8 CUDA solve issue
    x_cpu = torch.solve(dpose.detach().cpu(), A.detach().cpu()).solution
    x = x_cpu.to(device)

    u = (j_eef_T @ x).view(num_envs, 7)
    return u


np.random.seed(42)
torch.set_printoptions(precision=4, sci_mode=False)

gym = gymapi.acquire_gym()

custom_parameters = [
    {"name": "--controller", "type": str, "default": "ik"},
    {"name": "--num_envs", "type": int, "default": 1},
    {"name": "--debug_print", "action": "store_true"},
    {"name": "--headless", "action": "store_true"},
]

args = gymutil.parse_arguments(
    description="Franka latch IK grasp tuning",
    custom_parameters=custom_parameters,
)

device = args.sim_device if args.use_gpu_pipeline else "cpu"

# ---------------- Sim ----------------
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = args.use_gpu_pipeline

if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 2
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.002
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise RuntimeError("This example requires PhysX.")

damping = 0.01

sim = gym.create_sim(
    args.compute_device_id,
    args.graphics_device_id,
    args.physics_engine,
    sim_params,
)

if sim is None:
    raise RuntimeError("Failed to create sim")

viewer = None
if not args.headless:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise RuntimeError("Failed to create viewer")

asset_root = "../../assets"

# ---------------- Table ----------------
table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

# ---------------- Red cube marker only ----------------
box_size = 0.045
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
box_asset = gym.create_box(sim, box_size, box_size, box_size, asset_options)

# ---------------- Franka ----------------
franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.flip_visual_attachments = True
franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)

# ---------------- Latch asset ----------------
think_shell_asset_file = "urdf/think_shell.urdf"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True         # dynamic latch: can be lifted
asset_options.disable_gravity = False
asset_options.use_mesh_materials = False    # allow visible color
asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
asset_options.override_inertia = True
asset_options.vhacd_enabled = True
think_shell_asset = gym.load_asset(sim, asset_root, think_shell_asset_file, asset_options)

# ---------------- Franka DOFs ----------------
franka_dof_props = gym.get_asset_dof_properties(franka_asset)
franka_lower_limits = franka_dof_props["lower"]
franka_upper_limits = franka_dof_props["upper"]
franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)

franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
franka_dof_props["stiffness"][:7].fill(800.0)
franka_dof_props["damping"][:7].fill(80.0)

franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
franka_dof_props["stiffness"][7:].fill(800.0)
franka_dof_props["damping"][7:].fill(40.0)

franka_num_dofs = gym.get_asset_dof_count(franka_asset)

default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
default_dof_pos[:7] = franka_mids[:7]
default_dof_pos[7:] = franka_upper_limits[7:]

default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = default_dof_pos

franka_link_dict = gym.get_asset_rigid_body_dict(franka_asset)
franka_hand_index = franka_link_dict["panda_hand"]

# ---------------- Environments ----------------
num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs)) if num_envs > 1 else 1

spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

print(f"Creating {num_envs} environments")

franka_pose = gymapi.Transform()
franka_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)

table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)
table_top_z = table_pose.p.z + 0.5 * table_dims.z

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

envs = []
hand_idxs = []
think_shell_idxs = []
init_pos_list = []
init_rot_list = []

for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)

    # Red cube marker, not target
    marker_pose = gymapi.Transform()
    marker_pose.p = gymapi.Vec3(
        table_pose.p.x - 0.10,
        table_pose.p.y + 0.12,
        table_top_z + 0.5 * box_size,
    )
    marker_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0.0)

    box_handle = gym.create_actor(env, box_asset, marker_pose, "box_marker", i, 0)
    gym.set_rigid_body_color(
        env,
        box_handle,
        0,
        gymapi.MESH_VISUAL_AND_COLLISION,
        gymapi.Vec3(0.8, 0.05, 0.05),
    )

    # Latch target
    latch_z_offset = 0.24

    think_shell_pose = gymapi.Transform()
    think_shell_pose.p = gymapi.Vec3(
        table_pose.p.x + 0.05,
        table_pose.p.y,
        table_top_z + latch_z_offset,
    )
    think_shell_pose.r = gymapi.Quat.from_axis_angle(
        gymapi.Vec3(0, 1, 0),
        math.radians(-90),
    )

    think_shell_handle = gym.create_actor(
        env,
        think_shell_asset,
        think_shell_pose,
        "think_shell",
        i,
        0,
    )

    # Bright green latch
    gym.set_rigid_body_color(
        env,
        think_shell_handle,
        0,
        gymapi.MESH_VISUAL_AND_COLLISION,
        gymapi.Vec3(0.0, 1.0, 0.0),
    )

    if i == 0:
        rb_names = gym.get_actor_rigid_body_names(env, think_shell_handle)
        print("think_shell rigid bodies:", rb_names)

    think_shell_idx = gym.get_actor_rigid_body_index(
        env,
        think_shell_handle,
        0,
        gymapi.DOMAIN_SIM,
    )
    think_shell_idxs.append(think_shell_idx)

    franka_handle = gym.create_actor(env, franka_asset, franka_pose, "franka", i, 2)
    gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)
    gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, franka_handle, default_dof_pos)

    hand_handle = gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
    hand_pose = gym.get_rigid_transform(env, hand_handle)

    init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
    init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

    hand_idx = gym.find_actor_rigid_body_index(
        env,
        franka_handle,
        "panda_hand",
        gymapi.DOMAIN_SIM,
    )
    hand_idxs.append(hand_idx)

if viewer is not None:
    cam_pos = gymapi.Vec3(1.4, 1.0, 0.9)
    cam_target = gymapi.Vec3(0.45, 0.0, 0.35)
    gym.viewer_camera_look_at(viewer, envs[0], cam_pos, cam_target)

# ---------------- Tensors ----------------
gym.prepare_sim(sim)

init_pos = torch.tensor(init_pos_list, dtype=torch.float32, device=device).view(num_envs, 3)
init_rot = torch.tensor(init_rot_list, dtype=torch.float32, device=device).view(num_envs, 4)

down_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(num_envs, 1)

_jacobian = gym.acquire_jacobian_tensor(sim, "franka")
jacobian = gymtorch.wrap_tensor(_jacobian)
j_eef = jacobian[:, franka_hand_index - 1, :, :7]

_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)

dof_pos = dof_states[:, 0].view(num_envs, 9, 1)
dof_vel = dof_states[:, 1].view(num_envs, 9, 1)

pos_action = torch.zeros_like(dof_pos).squeeze(-1)
effort_action = torch.zeros_like(pos_action)

step = 0

while True:
    if viewer is not None and gym.query_viewer_has_closed(viewer):
        break

    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)

    obj_pos = rb_states[think_shell_idxs, :3]
    obj_rot = rb_states[think_shell_idxs, 3:7]

    hand_pos = rb_states[hand_idxs, :3]
    hand_rot = rb_states[hand_idxs, 3:7]

    obj_dist = torch.norm(obj_pos - hand_pos, dim=-1).unsqueeze(-1).clamp(min=1e-6)

    gripper_sep = (dof_pos[:, 7] + dof_pos[:, 8]).squeeze(-1)

    # ---------------- Grasp tuning section ----------------
    # phase 0: hover above grasp point
    # phase 1: descend to grasp point
    # phase 2: close gripper and wait
    # phase 3: lift latch straight upward

    hover_height = 0.20

    # ORIGINAL NOTATION:
    # approach_offset is the manual grasp point relative to latch base_link.
    # Tune only these three numbers.
    approach_offset = torch.tensor([
        -0.02,    # x offset
        -0.18,    # y offset
       -0.12,    # z offset, because latch_z_offset = 0.24
    ], device=device).view(1, 3)

    hover_pos = obj_pos + approach_offset
    hover_pos[:, 2] += hover_height

    grasp_pos = obj_pos + approach_offset

    hover_dist = torch.norm(hand_pos - hover_pos, dim=-1)
    grasp_dist = torch.norm(hand_pos - grasp_pos, dim=-1)

    if step == 0:
        phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        close_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        lift_start_pos = torch.zeros((num_envs, 3), device=device)
    phase = torch.where(
        (phase == 0) & (hover_dist < 0.05),
        torch.ones_like(phase),
        phase,
    )

    phase = torch.where(
        (phase == 1) & (grasp_dist < 0.04),
        torch.full_like(phase, 2),
        phase,
    )

    close_count = torch.where(phase == 2, close_count + 1, close_count)

    # Enter lift only after the gripper has had time to close.
    enter_lift = (phase == 2) & (close_count > 60)

    # Save the grasp point once, so phase 3 moves straight up instead of chasing the latch.
    lift_start_pos = torch.where(
        enter_lift.unsqueeze(-1),
        grasp_pos.detach(),
        lift_start_pos,
    )

    phase = torch.where(
        enter_lift,
        torch.full_like(phase, 3),
        phase,
    )

    # Lift straight upward from fixed XY.
    lift_pos = lift_start_pos.clone()
    lift_pos[:, 2] += 0.25

    goal_pos = torch.where(
        (phase == 0).unsqueeze(-1),
        hover_pos,
        torch.where(
            (phase == 3).unsqueeze(-1),
            lift_pos,
            grasp_pos,
        ),
    )
    goal_rot = down_q

    pos_err = goal_pos - hand_pos
    orn_err = orientation_error(goal_rot, hand_rot)
    dpose = torch.cat([pos_err, orn_err], dim=-1).unsqueeze(-1)

    max_pos_err = 0.06
    pos_norm = torch.norm(dpose[:, :3, 0], dim=-1, keepdim=True).clamp(min=1e-6)
    scale = torch.clamp(max_pos_err / pos_norm, max=1.0)
    dpose[:, :3, :] *= scale.view(num_envs, 1, 1)

    pos_action[:, :7] = dof_pos.squeeze(-1)[:, :7] + control_ik(dpose)

    #close_gripper = phase >= 2
    close_gripper = phase >= 2

    phase = torch.where(
        (phase == 2) & (gripper_sep < 0.01),
        torch.full_like(phase, 3),
        phase,
    )

    lift_pos = lift_start_pos.clone()
    lift_pos[:, 2] += 0.25
    grip_open = torch.ones((num_envs, 2), device=device) * 0.04
    grip_close = torch.zeros((num_envs, 2), device=device)

    pos_action[:, 7:9] = torch.where(
        close_gripper.unsqueeze(-1),
        grip_close,
        grip_open,
    )

    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))
    gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(effort_action))

    if args.debug_print and step % 60 == 0:
        target_dist = torch.norm(hand_pos[0] - goal_pos[0]).item()
        print(
            f"step={step:05d} "
            f"phase={int(phase[0].item())} "
            f"obj={obj_pos[0].detach().cpu().numpy()} "
            f"hand={hand_pos[0].detach().cpu().numpy()} "
            f"goal={goal_pos[0].detach().cpu().numpy()} "
            f"obj_dist={float(obj_dist[0].item()):.4f} "
            f"target_dist={target_dist:.4f} "
            f"grip_sep={float(gripper_sep[0].item()):.4f} "
            f"offset={approach_offset[0].detach().cpu().numpy()}"
        )

    if viewer is not None:
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)

    step += 1

if viewer is not None:
    gym.destroy_viewer(viewer)

gym.destroy_sim(sim)