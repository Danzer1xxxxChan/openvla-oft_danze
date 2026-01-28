import os
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Union

import draccus
import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm

# 数据与模型依赖
from prismatic.vla.constants import (
    ACTION_DIM,
    NUM_ACTIONS_CHUNK,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NormalizationType,
)
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.models.backbones.llm.prompting import PurePromptBuilder

# 评测侧工具：加载 VLA/Processor/ActionHead/Projectors，并解析 unnorm_key
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_model,
    set_seed_everywhere,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

@dataclass
class OpenLoopEvalConfig:
    # 模型
    pretrained_checkpoint: Union[str, Path] = ""
    model_family: str = "openvla"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    lora_rank: int = 32

    # 预测与输入
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 1
    use_proprio: bool = False
    center_crop: bool = False

    # 数据
    data_root_dir: Union[str, Path] = ""
    dataset_name: str = ""               # e.g., "pick_n_place_zechen_version"
    image_aug: bool = False              # 评测时应为 False
    batch_size: int = 1                  # 建议开环评测用 1，避免批内不同长序列
    split: str = "val"                   # "train" 或 "val"
    shuffle_buffer_size: int = 10000
    max_batches: int = -1                # 负数代表跑完整 split

    # 反标准化 key
    custom_unnorm_key: str = ""          # 默认用 dataset_name；如训练时另存了 key，可显式传入

    # 日志
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    seed: int = 7

def _resolve_unnorm_key(model, dataset_name: str, custom_unnorm_key: str) -> str:
    # 尝试按 dataset_name 匹配，若不在则尝试 *_no_noops，最后使用 custom_unnorm_key
    # 若都不匹配，回退到首个可用 key（与训练保存一致）
    norm_stats_keys = list(model.norm_stats.keys()) if hasattr(model, "norm_stats") else []
    candidates = []
    if dataset_name:
        candidates.extend([dataset_name, f"{dataset_name}_no_noops"])
    if custom_unnorm_key:
        candidates.extend([custom_unnorm_key, f"{custom_unnorm_key}_no_noops"])

    for k in candidates:
        if k in norm_stats_keys:
            return k
    if norm_stats_keys:
        logger.warning(f"Cannot find matching unnorm key; fallback to {norm_stats_keys[0]}")
        return norm_stats_keys[0]
    raise RuntimeError("No dataset_statistics found in checkpoint; cannot unnormalize actions!")

def _unnormalize_actions(gt_norm: np.ndarray, action_stats: Dict[str, Any]) -> np.ndarray:
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        high, low = np.array(action_stats["max"]), np.array(action_stats["min"])
        mask = action_stats.get("mask", np.ones_like(low, dtype=bool))
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        high, low = np.array(action_stats["q99"]), np.array(action_stats["q01"])
        mask = action_stats.get("mask", np.ones_like(low, dtype=bool))
    else:
        raise ValueError("Unsupported normalization type")

    gt = np.where(mask, 0.5 * (gt_norm + 1.0) * (high - low + 1e-8) + low, gt_norm)
    return gt

def _metrics_for_pair(pred: np.ndarray, gt: np.ndarray) -> Dict[str, Any]:
    # pred/gt: (T=NUM_ACTIONS_CHUNK, D=ACTION_DIM)
    abs_err = np.abs(pred - gt)
    mae_all = abs_err.mean()

    pos_mae = abs_err[:, :3].mean()
    rot_mae = abs_err[:, 3:6].mean()
    grip_mae = abs_err[:, 6].mean()

    # 夹爪准确率：阈值 0.5（去标准化后）
    gt_grip_open = (gt[:, 6] > 0.5).astype(np.float32)
    pd_grip_open = (pred[:, 6] > 0.5).astype(np.float32)
    grip_acc = (gt_grip_open == pd_grip_open).mean()

    # 位置 ADE/FDE（欧氏距离）
    pos_err = pred[:, :3] - gt[:, :3]
    pos_l2 = np.linalg.norm(pos_err, axis=1)
    ade = pos_l2.mean()
    fde = pos_l2[-1]

    # 步长曲线（每步 MAE，跨维度求平均）
    step_mae = abs_err.mean(axis=1).tolist()

    return dict(
        mae_all=float(mae_all),
        mae_pos=float(pos_mae),
        mae_rot=float(rot_mae),
        mae_grip=float(grip_mae),
        grip_acc=float(grip_acc),
        ade=float(ade),
        fde=float(fde),
        step_mae=step_mae,
    )

@draccus.wrap()
def main(cfg: OpenLoopEvalConfig) -> None:
    set_seed_everywhere(cfg.seed)

    assert cfg.pretrained_checkpoint, "pretrained_checkpoint is required"
    assert cfg.dataset_name, "dataset_name is required"
    assert cfg.batch_size == 1, "For open-loop evaluation, please set batch_size=1"

    # 1) 模型与头
    model = get_model(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, model.llm_dim) if (cfg.use_l1_regression or cfg.use_diffusion) else None
    noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim) if cfg.use_diffusion else None

    # 2) 反标准化 key
    unnorm_key = _resolve_unnorm_key(model, cfg.dataset_name, cfg.custom_unnorm_key)
    action_stats = model.norm_stats[unnorm_key]["action"]

    # 3) 数据集与 DataLoader（严格复用训练时的变换）
    use_wrist_image = cfg.num_images_in_input > 1
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
        showlab_franka=True,
    )
    dataset = RLDSDataset(
        data_root_dir=cfg.data_root_dir,
        data_mix=cfg.dataset_name,
        batch_transform=batch_transform,
        resize_resolution=tuple(model.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        train=(cfg.split == "train"),
        image_aug=False,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=collator, num_workers=0)

    # 4) 评测循环
    per_item_results = []
    agg_buffers = {
        "mae_all": [],
        "mae_pos": [],
        "mae_rot": [],
        "mae_grip": [],
        "grip_acc": [],
        "ade": [],
        "fde": [],
        "step_mae": [],  # list of lists
    }

    n = 0
    pbar = tqdm.tqdm(loader, desc="Open-loop evaluating")
    model.eval()
    with torch.inference_mode():
        for batch in pbar:
            # 输入（与训练一致）：注意 predict_action 需要 input_ids/attention_mask/pixel_values
            inputs = {
                "input_ids": batch["input_ids"].to(model.device),
                "attention_mask": batch["attention_mask"].to(model.device),
                "pixel_values": batch["pixel_values"].to(model.device, dtype=torch.bfloat16),
            }

            proprio = batch.get("proprio", None)
            if proprio is not None:
                proprio = proprio.numpy()

            # 模型预测（开环 chunk）
            pred_actions, _ = model.predict_action(
                **inputs,
                unnorm_key=unnorm_key,
                proprio=proprio,
                proprio_projector=None,             # 若 cfg.use_proprio=True 且你有 projector，可在此传入
                noisy_action_projector=noisy_action_projector,
                action_head=action_head,
                use_film=cfg.use_film,
                do_sample=False,
            )
            # pred_actions: (T=NUM_ACTIONS_CHUNK, D=ACTION_DIM) 已是去标准化后的

            # GT 动作（来自 RLDS，是标准化后的，需要去标准化）
            gt_norm = batch["actions"][0].numpy()  # (T, D)
            gt_actions = _unnormalize_actions(gt_norm, action_stats)

            # 指标
            metrics = _metrics_for_pair(pred_actions, gt_actions)
            per_item_results.append(
                {
                    **metrics,
                    "gt_seq": gt_actions.tolist(),
                    "pred_seq": pred_actions.tolist(),
                }
            )
            for k in agg_buffers:
                if k == "step_mae":
                    agg_buffers[k].append(metrics[k])
                else:
                    agg_buffers[k].append(metrics[k])

            # 更新进度
            n += 1
            pbar.set_postfix(mae_all=f"{metrics['mae_all']:.4f}", grip_acc=f"{metrics['grip_acc']:.3f}")

            if cfg.max_batches > 0 and n >= cfg.max_batches:
                break

    # 5) 聚合
    def mean(xs): return float(np.mean(xs)) if len(xs) else float("nan")
    overall = {
        "count": n,
        "mae_all": mean(agg_buffers["mae_all"]),
        "mae_pos": mean(agg_buffers["mae_pos"]),
        "mae_rot": mean(agg_buffers["mae_rot"]),
        "mae_grip": mean(agg_buffers["mae_grip"]),
        "grip_acc": mean(agg_buffers["grip_acc"]),
        "ade": mean(agg_buffers["ade"]),
        "fde": mean(agg_buffers["fde"]),
        "step_mae": np.mean(np.array(agg_buffers["step_mae"]), axis=0).tolist() if len(agg_buffers["step_mae"]) else [],
    }

    # 6) 保存
    run_id = f"OPENLOOP-{cfg.dataset_name}-{DATE_TIME}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    out_path = os.path.join(cfg.local_log_dir, run_id + ".json")
    with open(out_path, "w") as f:
        json.dump({"overall": overall, "per_item": per_item_results}, f, indent=2)
    logger.info(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()
