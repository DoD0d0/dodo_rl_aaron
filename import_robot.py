##bolt robot##
# import numpy as np
# import genesis as gs

# gs.init(backend=gs.gpu)

# scene = gs.Scene(
#     viewer_options=gs.options.ViewerOptions(
#         camera_pos    = (0.0, -2.0, 1.0),
#         camera_lookat = (0.0,  0.0, 0.5),
#         camera_fov    = 40,
#         max_FPS       = 60,
#     ),
#     sim_options=gs.options.SimOptions(
#         dt       = 0.01,   # 100 Hz
#         substeps = 2,
#     ),
#     show_viewer=True,
# )

# # 2. 添加地面
# scene.add_entity(gs.morphs.Plane())


# # bolt = scene.add_entity(
# #     gs.morphs.MJCF(
# #         file  = "/home/nvidiapc/dodo/Genesis/genesis/assets/urdf/bolt/bolt_mv.xml",
# #         pos   = (0, 0, 0.5),
# #         euler = (0, 0, 0),
# #         # MJCF 默认 is_free=True → 浮动基座
# #     )
# # )

# scene.build(n_envs=1)

# # 5. 关节名称 & dof 索引
# jnt_names = [
#     "FR_HAA", "FR_HFE", "FR_KFE",
#     "FL_HAA", "FL_HFE", "FL_KFE"
# ]
# dofs_idx = [bolt.get_joint(name).dof_idx_local for name in jnt_names]

# torque_command = 1 * np.ones(len(dofs_idx), dtype=np.float32)

# for step in range(500):
#     # 直接施加扭矩
#     bolt.control_dofs_force(torque_command, dofs_idx)
#     input("按回车继续下一步仿真…")
    
#     # 可选：打印当前步骤的“控制扭矩”与“实际扭矩”
#     if step % 5 == 0:
#         print(f"Step {step}")
#         print(" Control force:", bolt.get_dofs_control_force(dofs_idx))
#         print(" Internal force:", bolt.get_dofs_force(dofs_idx))
    
#     scene.step()

########################## dodo robot ##########################
import numpy as np
import genesis as gs

from dodo_env_new_2 import Dodoenvironment
gs.init(backend=gs.gpu)


scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos    = (1.5, -1.5, 1.5),
        camera_lookat = (0.0,  0.0, 0.5),
        camera_fov    = 40,
        max_FPS       = 60,
    ),
    sim_options=gs.options.SimOptions(
        dt       = 0.01,   # 100 Hz
        substeps = 2,
    ),
    show_viewer=True,
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)

dodo = scene.add_entity(
    gs.morphs.URDF(
        file  = "/Users/aaronalexander/DoDodo/dodobot_v3/urdf/dodobot_v3_simple.urdf",
        pos = (0, 0, 0.55),
        euler = (0, 0, 0),
    )
)

scene.build(n_envs=1)

jnt_names = [ "left_joint_1","right_joint_1","left_joint_2","right_joint_2","left_joint_3","right_joint_3","left_joint_4","right_joint_4"]
dofs_idx  = [dodo.get_joint(name).dof_idx_local for name in jnt_names]
n_dofs    = len(dofs_idx)

q_amp  = 0.5
freq   = 0.5
omega  = 2 * np.pi * freq

kp     = 200.0  * np.ones(n_dofs, dtype=np.float32)
kv     = 2.0*np.sqrt(kp) 
dodo.set_dofs_kp(kp, dofs_idx)
dodo.set_dofs_kv(kv, dofs_idx)

dodo.set_dofs_force_range(
    lower = -100*np.ones(n_dofs, dtype=np.float32),
    upper =  100*np.ones(n_dofs, dtype=np.float32),
    dofs_idx_local = dofs_idx,
)
total_steps = 2000
dt = scene.sim_options.dt

for step in range(total_steps):
    t = step * dt
    q_des = q_amp * np.sin(omega * t) * np.ones(n_dofs, dtype=np.float32)
    dodo.control_dofs_position(q_des, dofs_idx)
    input("enter to continue…")
    base_pos = dodo.get_pos()
    print(f"[torque ctrl] step {step:4d} → base height = {base_pos[0,2]:.4f} m")
    scene.step()

torque_amp = 5.0 

for step in range(total_steps):
    t = step * dt
    torque = torque_amp * np.sin(omega * t) * np.ones(n_dofs, dtype=np.float32)
    dodo.control_dofs_force(torque, dofs_idx)
    scene.step()
