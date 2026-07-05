# Franka_PPO
Learn Graping Latch using PPO and IK so PPO can learn smooth behavior like:  
open → move around latch → close slowly → squeeze → lift

Prerequisites
    
    Download IsaacGym 
    
    run ./create_conda_env_rlgpu.sh
    
    Ubuntu 18.04 or 20.04.

    Python 3.6, 3.7 or 3.8.

    Minimum NVIDIA driver version:

        Linux: 470
        
Recommend in VSCode:

1. conda activate rlgpu 
2. python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())" -> it should print "True 1"
3. put this folder under isaacgym/python/Franka_PPO_v2
4. export LD_LIBRARY_PATH=/home/jimmy/anaconda3/envs/rlgpu/lib
5. export LD_LIBRARY_PATH=/home/jimmy/anaconda3/envs/rlgpu/lib:${LD_LIBRARY_PATH}
6. cd PPO_franka
7. python franka_latch_ik_lift_cpu_solve.py \
  --num_envs 16 \
  --eval \
  --headless \
  --debug_print \
  --checkpoint ppo_latch_free_v9.pt

Training:

PPO update at policy step 499200; episodes=29406
step=499230 slide_unlocked=False released=False slide=0.0000 lift=0.0000 latch_z=0.4586 table_z=0.4000 reach=0.059 finger_d=(0.069,0.072) grip_width=0.077 force=(0.88,1.21) latch_xy_force=0.57 contact=(True,True) grasped=False hold=0 ready=False active=1/16 released_n=0 max_lift=0.0000 obj_h=0.0586 locked_err=0.000000 reward=1.584 return=13.03

when the console reaches approximately
1. PPO update at policy step 9984:
cp ppo_latch_free_v9.pt ppo_latch_free_v9_10k.pt
2. PPO update at policy step 49920:
cp ppo_latch_free_v9.pt ppo_latch_free_v9_50k.pt
