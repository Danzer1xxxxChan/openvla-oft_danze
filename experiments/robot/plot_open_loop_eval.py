import argparse
import json
import os
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

METRIC_KEYS = ["mae_all", "mae_pos", "mae_rot", "mae_grip", "ade", "fde"]
DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

def load_result(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    overall = data.get("overall", {})
    per_item = data.get("per_item", [])
    step_mae = overall.get("step_mae", [])
    return overall, per_item, step_mae

def extract_gt_pred_arrays(per_item) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays shaped (N, T, 7) for GT / prediction sequences.

    Gracefully handle older result files where sequences are missing or have
    unexpected shapes by skipping those entries.
    """

    gt_list, pd_list = [], []
    for it in per_item:
        if "gt_seq" not in it or "pred_seq" not in it:
            continue

        gt_arr = np.array(it["gt_seq"], dtype=np.float32)
        pd_arr = np.array(it["pred_seq"], dtype=np.float32)

        # Expect shape (T, 7); skip if shape invalid
        if gt_arr.ndim != 2 or pd_arr.ndim != 2:
            continue
        if gt_arr.shape != pd_arr.shape:
            continue
        if gt_arr.shape[1] != len(DIM_NAMES):
            continue

        gt_list.append(gt_arr)
        pd_list.append(pd_arr)

    if not gt_list:
        empty = np.empty((0, 0, 0), dtype=np.float32)
        return empty, empty

    gt = np.stack(gt_list, axis=0)  # (N, T, 7)
    pd = np.stack(pd_list, axis=0)  # (N, T, 7)
    return gt, pd

def plot_step_mae(results: List[str], labels: List[str], out_dir: str):
    from pathlib import Path
    plt.figure(figsize=(7, 4))
    for path, label in zip(results, labels):
        _, _, step_mae = load_result(path)
        if not step_mae:
            continue
        xs = np.arange(1, len(step_mae) + 1)
        plt.plot(xs, step_mae, marker="o", linewidth=2, label=label)
    plt.xlabel("Prediction step (k)")
    plt.ylabel("MAE (avg over dims)")
    plt.title("Open-loop step-wise MAE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "step_mae_trend.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")
    plt.close()

def plot_overall_bars(results: List[str], labels: List[str], out_dir: str):
    vals = []
    for path in results:
        overall, _, _ = load_result(path)
        vals.append([overall.get(k, np.nan) for k in METRIC_KEYS])

    vals = np.array(vals)  # (N_runs, N_metrics)
    N, M = vals.shape
    width = 0.85 / max(N, 1)
    xs = np.arange(M)

    plt.figure(figsize=(max(7, M * 1.2), 4.2))
    for i in range(N):
        plt.bar(xs + i * width, vals[i], width=width, label=labels[i])
    plt.xticks(xs + width * (N - 1) / 2, METRIC_KEYS, rotation=20)
    plt.ylabel("Value")
    plt.title("Overall metrics comparison")
    plt.grid(axis="y", alpha=0.3)
    if N > 1:
        plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "overall_metrics.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")
    plt.close()

def plot_per_item_box(results: List[str], labels: List[str], out_dir: str):
    boxes = []
    for path in results:
        _, per_item, _ = load_result(path)
        mae_all = [x.get("mae_all", np.nan) for x in per_item]
        boxes.append(mae_all if len(mae_all) > 0 else [np.nan])

    plt.figure(figsize=(max(7, len(results) * 1.2), 4))
    plt.boxplot(boxes, labels=labels, vert=True, showfliers=False)
    plt.ylabel("Per-item MAE (all dims)")
    plt.title("Distribution of per-item MAE")
    plt.grid(axis="y", alpha=0.3)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "per_item_mae_box.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")
    plt.close()

def plot_dim_trend_mean(results: List[str], labels: List[str], out_dir: str):
    # 多 run 对比：每个维度画一张图，曲线为各 run 的全数据集均值（可直观看到趋势）
    os.makedirs(out_dir, exist_ok=True)
    for d in range(7):
        plt.figure(figsize=(7, 4))
        for path, label in zip(results, labels):
            _, per_item, _ = load_result(path)
            gt, pd = extract_gt_pred_arrays(per_item)
            if gt.size == 0 or gt.ndim != 3:
                continue
            # (N, T)
            gt_dim = gt[:, :, d]
            pd_dim = pd[:, :, d]
            xs = np.arange(1, gt_dim.shape[1] + 1)
            plt.plot(xs, gt_dim.mean(axis=0), label=f"{label}-GT", linestyle="--", linewidth=2)
            plt.plot(xs, pd_dim.mean(axis=0), label=f"{label}-Pred", linewidth=2)
        plt.xlabel("Prediction step (k)")
        plt.ylabel(f"{DIM_NAMES[d]} value")
        plt.title(f"Mean trend over dataset - {DIM_NAMES[d]}")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2)
        out_path = os.path.join(out_dir, f"dim_trend_mean_{DIM_NAMES[d]}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        print(f"Saved: {out_path}")
        plt.close()

def plot_dim_trend_items(result: str, out_dir: str, item_indices: List[int]):
    # 单个结果文件里，挑选若干样本，画“7 子图”的 gt/pred 对比
    _, per_item, _ = load_result(result)
    os.makedirs(out_dir, exist_ok=True)
    for idx in item_indices:
        if idx < 0 or idx >= len(per_item) or "gt_seq" not in per_item[idx]:
            continue
        gt = np.array(per_item[idx]["gt_seq"], dtype=np.float32)
        pd = np.array(per_item[idx]["pred_seq"], dtype=np.float32)

        if gt.ndim != 2 or pd.ndim != 2 or gt.shape != pd.shape or gt.shape[1] != len(DIM_NAMES):
            continue

        T = gt.shape[0]
        xs = np.arange(1, T + 1)

        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()
        for d in range(7):
            ax = axes[d]
            ax.plot(xs, gt[:, d], label="GT", linestyle="--", linewidth=2)
            ax.plot(xs, pd[:, d], label="Pred", linewidth=2)
            ax.set_title(DIM_NAMES[d])
            ax.set_xlabel("k"); ax.set_ylabel("value")
            ax.grid(True, alpha=0.3)
        axes[-1].axis("off")  # 空白第8格
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle(f"Per-dimension trend (item {idx})")
        out_path = os.path.join(out_dir, f"dim_trend_item_{idx}.png")
        plt.tight_layout(rect=[0, 0, 0.98, 0.95])
        plt.savefig(out_path, dpi=200)
        print(f"Saved: {out_path}")
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True, help="One or more OPENLOOP-*.json files")
    parser.add_argument("--labels", nargs="*", default=None, help="Legend labels; default=filenames")
    parser.add_argument("--out_dir", type=str, default=None, help="Output dir; default=dirname(results[0])")
    parser.add_argument("--per_item_dim_plots", type=int, default=0, help="Draw N per-item dim-trend figures from the first result")
    parser.add_argument("--item_indices", nargs="*", type=int, default=None, help="Explicit per-item indices to plot")
    args = parser.parse_args()

    results = args.results
    labels = args.labels or [os.path.splitext(os.path.basename(p))[0] for p in results]
    assert len(labels) == len(results), "labels length must match results count"

    out_dir = args.out_dir or os.path.dirname(results[0]) or "."
    os.makedirs(out_dir, exist_ok=True)

    # 既有图
    plot_step_mae(results, labels, out_dir)
    plot_overall_bars(results, labels, out_dir)
    plot_per_item_box(results, labels, out_dir)

    # 新增维度趋势图
    plot_dim_trend_mean(results, labels, out_dir)

    # 单样本 7 维子图
    if args.item_indices is not None:
        plot_dim_trend_items(results[0], out_dir, args.item_indices)
    elif args.per_item_dim_plots > 0:
        plot_dim_trend_items(results[0], out_dir, list(range(args.per_item_dim_plots)))

if __name__ == "__main__":
    main()