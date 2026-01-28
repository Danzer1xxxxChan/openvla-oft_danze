export MUJOCO_GL="osmesa"    # glfw, glx, osmesa

#MUJOCO_GL="osmesa" CUDA_VISIBLE_DEVICES=2 python experiments/robot/libero/run_libero_eval.py \
#  --pretrained_checkpoint ZechenBai/OpenVLA-OFT \
#  --use_l1_regression False \
#  --use_diffusion False \
#  --use_proprio False \
#  --num_images_in_input 1 \
#  --task_suite_name libero_10 \
#  --custom_unnorm_key robot_rlds_kevin

#MUJOCO_GL="osmesa" CUDA_VISIBLE_DEVICES=2 python experiments/robot/libero/run_libero_eval.py \
#  --pretrained_checkpoint /mnt/bn/vgfm2/test_dit/zechen/openvla-oft/ckpt_log/openvla-7b+robot_rlds_kevin+b8+lr-0.0005+lora-r32+dropout-0.0--parallel_dec--8_acts_chunk--l1_reg--3rd_person_img--no_crop--small_lr \
#  --use_l1_regression True \
#  --use_diffusion False \
#  --use_proprio False \
#  --num_images_in_input 1 \
#  --task_suite_name libero_10 \
#  --custom_unnorm_key robot_rlds_kevin \
#  --num_trials_per_task 1

#MUJOCO_GL="osmesa" CUDA_VISIBLE_DEVICES=2 python experiments/robot/libero/run_libero_eval.py \
#  --pretrained_checkpoint ckpt_log/openvla-7b-1e4+robot_rlds_kevin+b8+lr-1e-05+lora-r16+dropout-0.0--parallel_dec--8_acts_chunk--l1_reg--3rd_person_img--no_crop--small_lr--10000_chkpt \
#  --use_l1_regression True \
#  --use_diffusion False \
#  --use_proprio False \
#  --num_images_in_input 1 \
#  --task_suite_name libero_10 \
#  --custom_unnorm_key robot_rlds_kevin \
#  --num_trials_per_task 1

MUJOCO_GL="osmesa" CUDA_VISIBLE_DEVICES=2 python experiments/robot/libero/run_libero_eval_zechen.py \
  --pretrained_checkpoint ckpt_log/openvla-7b+pick_n_place_zechen_version+b8+lr-5e-05+lora-r32+dropout-0.0--parallel_dec--8_acts_chunk--l1_reg--3rd_person_img--no_aug--50000_chkpt \
  --use_l1_regression True \
  --use_diffusion False \
  --use_proprio False \
  --num_images_in_input 1 \
  --task_suite_name libero_10 \
  --custom_unnorm_key pick_n_place_zechen_version \
  --num_trials_per_task 1 \

