"""Plotting utilities for individual experiments and experiment comparisons.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    Các bước: fig, axes = plt.subplots(1, 3, figsize=...); plot; set_title/xlabel/legend;
              fig.savefig(path, dpi=..., bbox_inches="tight"); plt.close(fig)
    Gợi ý: đánh dấu best_epoch bằng đường thẳng đứng.
    """
    history = result.get("history", {})
    epochs = history.get("epoch", [])
    cfg = result.get("cfg", {})
    summary = result.get("summary", {})
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    if epochs:
        axes[0].plot(epochs, history.get("train_loss", []), label="train loss")
        axes[0].plot(epochs, history.get("val_loss", []), label="validation loss")
        axes[1].plot(epochs, history.get("val_acc", []), label="validation accuracy")
        axes[1].plot(epochs, history.get("val_macro_f1", []), label="validation macro-F1")
        axes[2].plot(epochs, history.get("grad_norm", []), label="gradient norm")
        best_epoch = summary.get("best_epoch")
        if best_epoch in epochs:
            for axis in axes:
                axis.axvline(best_epoch, color="gray", linestyle="--", alpha=0.6)

    axes[0].set_title("Loss")
    axes[1].set_title("Validation metrics")
    axes[2].set_title("Gradient norm (before clipping)")
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Score")
    axes[2].set_ylabel("L2 norm")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.25)
        if axis.lines:
            axis.legend()

    exp_id = cfg.get("exp_id", "experiment")
    config_label = (
        f"{cfg.get('optimizer', '?')} | lr={cfg.get('lr', '?')} | "
        f"batch={cfg.get('batch', '?')} | hidden={cfg.get('hidden', '?')} | "
        f"dropout={cfg.get('dropout', '?')}"
    )
    figure.suptitle(f"{exp_id} — {config_label}")
    figure.tight_layout()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    if not results:
        raise ValueError("results must contain at least one experiment")
    figure, axis = plt.subplots(figsize=(8, 5))
    plotted = False
    for result in results:
        cfg = result.get("cfg", {})
        history = result.get("history", {})
        values = history.get(metric)
        if values is None:
            raise KeyError(f"Metric {metric!r} is not present in experiment history")
        epochs = history.get("epoch", list(range(1, len(values) + 1)))
        if len(epochs) != len(values):
            raise ValueError(f"Metric {metric!r} has a different number of epochs")
        axis.plot(epochs, values, label=cfg.get("exp_id", "experiment"))
        plotted = plotted or bool(values)
    axis.set_title(title or metric)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(metric)
    axis.grid(True, alpha=0.25)
    if plotted:
        axis.legend()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
