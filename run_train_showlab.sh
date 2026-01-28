#torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
#  --vla_path openvla/openvla-7b \
#  --data_root_dir /mnt/bn/vgfm2/test_dit/zechen/DATA/Robo/real_world \
#  --dataset_name robot_rlds_kevin \
#  --run_root_dir ./ckpt_log \
#  --use_l1_regression False \
#  --use_diffusion False \
#  --use_film False \
#  --num_images_in_input 1 \
#  --use_proprio False \
#  --batch_size 8 \
#  --learning_rate 5e-4 \
#  --num_steps_before_decay 100000 \
#  --max_steps 150005 \
#  --save_freq 10000 \
#  --save_latest_checkpoint_only False \
#  --image_aug True \
#  --lora_rank 32 \
#  --wandb_project "OpenVLA-OFT" \
#  --run_id_note parallel_dec--8_acts_chunk--discrete_tok--3rd_person_img

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

#torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
#  --vla_path openvla/openvla-7b \
#  --data_root_dir /mnt/bn/vgfm2/test_dit/zechen/DATA/Robo/real_world \
#  --dataset_name robot_rlds_kevin \
#  --run_root_dir ./ckpt_log \
#  --use_l1_regression True \
#  --use_diffusion False \
#  --use_film False \
#  --num_images_in_input 1 \
#  --use_proprio False \
#  --batch_size 8 \
#  --learning_rate 5e-4 \
#  --num_steps_before_decay 100000 \
#  --max_steps 150005 \
#  --save_freq 10000 \
#  --save_latest_checkpoint_only True \
#  --image_aug False \
#  --lora_rank 32 \
#  --wandb_project "OpenVLA-OFT" \
#  --merge_lora_during_training True \
#  --run_id_note parallel_dec--8_acts_chunk--l1_reg--3rd_person_img--no_crop

torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /mnt/bn/vgfm2/test_dit/zechen/DATA/Robo/real_world \
  --dataset_name robot_rlds_kevin \
  --run_root_dir ./ckpt_log \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio False \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --num_steps_before_decay 7000 \
  --max_steps 50005 \
  --save_freq 7000 \
  --save_latest_checkpoint_only True \
  --image_aug False \
  --lora_rank 16 \
  --wandb_project "OpenVLA-OFT" \
  --merge_lora_during_training True \
  --run_id_note parallel_dec--8_acts_chunk--l1_reg--3rd_person_img--no_crop--1e4
