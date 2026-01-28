#!/usr/bin/env bash

# GPU 设置（可选）
export CUDA_VISIBLE_DEVICES=1

# 基础配置（按需修改）
PRETRAINED_CKPT="ckpt_log/openvla-7b+pick_n_place_zechen_version+b8+lr-5e-05+lora-r32+dropout-0.0--parallel_dec--8_acts_chunk--l1_reg--3rd_person_img--no_aug--50000_chkpt"
DATA_ROOT="/mnt/bn/vgfm2/test_dit/zechen/DATA/Robo/real_world"
DATASET="pick_n_place_zechen_version"

# 评测开关（与训练一致）
USE_L1="True"
USE_DIFF="False"
USE_PROPRIO="False"
NUM_IMAGES="1"
CENTER_CROP="False"   # 若训练用了图像增强，需 True；否则 False

# 运行与日志
BATCH_SIZE=1
SPLIT="train"           # "val" 或 "train"
MAX_BATCHES=64        # -1 跑完整 split
RUN_NOTE="openloop_eval"
LOG_DIR="./ckpt_log/logs"

# 1) 运行开环评测
python experiments/robot/run_open_loop_eval_real.py \
  --pretrained_checkpoint "${PRETRAINED_CKPT}" \
  --data_root_dir "${DATA_ROOT}" \
  --dataset_name "${DATASET}" \
  --custom_unnorm_key "${DATASET}" \
  --use_l1_regression ${USE_L1} \
  --use_diffusion ${USE_DIFF} \
  --use_proprio ${USE_PROPRIO} \
  --num_images_in_input ${NUM_IMAGES} \
  --center_crop ${CENTER_CROP} \
  --batch_size ${BATCH_SIZE} \
  --split ${SPLIT} \
  --max_batches ${MAX_BATCHES} \
  --run_id_note "${RUN_NOTE}" \
  --local_log_dir "${LOG_DIR}"

# 2) 找到最新结果 JSON
RESULT_JSON=$(ls -t ${LOG_DIR}/OPENLOOP-${DATASET}-*.json | head -n 1)
if [[ -z "${RESULT_JSON}" ]]; then
  echo "No result JSON found in ${LOG_DIR}"; exit 1;
fi
echo "Latest result: ${RESULT_JSON}"

# 3) 画图（单次结果）
python experiments/robot/plot_open_loop_eval.py \
  --results "${RESULT_JSON}" \
  --labels "${DATASET}" \
  --out_dir "${LOG_DIR}" \
  --per_item_dim_plots 3

# 4) 多结果对比（可选，取消注释使用）
# python experiments/robot/plot_open_loop_eval.py \
#   --results "${LOG_DIR}/OPENLOOP-${DATASET}-<timeA>.json" "${LOG_DIR}/OPENLOOP-${DATASET}-<timeB>.json" \
#   --labels "runA" "runB" \
#   --out_dir "${LOG_DIR}"
