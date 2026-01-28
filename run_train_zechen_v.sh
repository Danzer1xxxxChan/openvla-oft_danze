export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=2
export NCCL_IB_TIMEOUT=22
export TORCH_NCCL_BLOCKING_WAIT=0
# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_HANDLING=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Force Python to use local prismatic code
# export PYTHONPATH=/users/xiaokangliu/robotics/openvla-oft-xk:$PYTHONPATH
export PYTHONPATH=/users/danze/VLA/openvla-oft-xk:$PYTHONPATH

# --data_root_dir /mnt/bn/vgfm2/test_dit/zechen/DATA/Robo/real_world \
  # --data_root_dir /storage/xiaokangliu/data/VLAST-Data/empty_rlds/ \
  # --data_root_dir /storage/xiaokangliu/data/VLAST-Data/empty_rlds_nonorm/ \
  # --data_root_dir /storage/xiaokangliu/data/VLAST-Data/empty_rlds_skip_frames \
torchrun --standalone --nnodes 1 --nproc-per-node 7 vla-scripts/finetune_real_world.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /storage/danze/VLA/openvla/pick_and_place_orange_block_01_27/ \
  --dataset_name pick_n_place_ee \
  --run_root_dir /storage/danze/VLA/openvla/ckpt_log_danze_orange_block_01_27 \
  --use_l1_regression False \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio False \
  --batch_size 2 \
  --learning_rate 5e-5 \
  --num_steps_before_decay 10000 \
  --max_steps 10005 \
  --save_freq 1000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_project "OpenVLA-OFT-Orange-Block" \
  --merge_lora_during_training True \
  --run_id_note orange_block_noq99_nonorm_nol1_parallel_dec--24_acts_chunk--l1_reg--3rd_person_img--no_aug \
  --shuffle_buffer_size 10000

# remember to turn off center_crop in evaluation

