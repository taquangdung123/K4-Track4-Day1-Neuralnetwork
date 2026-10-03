"""Training, evaluation, prediction, and seeding utilities for the lab.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import copy
import csv
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). Chọn `lr` bằng val trước khi chạy.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # bắt buộc chọn bằng validation trước khi huấn luyện
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    cm = np.asarray(cm, dtype=np.float64)
    if cm.shape != (7, 7):
        raise ValueError(f"Confusion matrix must have shape (7, 7), got {cm.shape}")
    true_positive = np.diag(cm)
    false_positive = cm.sum(axis=0) - true_positive
    false_negative = cm.sum(axis=1) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    per_class_f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros(7, dtype=np.float64),
        where=denominator > 0,
    )
    return float(per_class_f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model.eval()
    predictions = [
        model(X[start : start + batch_size]).argmax(dim=1)
        for start in range(0, len(X), batch_size)
    ]
    if not predictions:
        return torch.empty(0, dtype=torch.int64, device=X.device)
    return torch.cat(predictions).to(dtype=torch.int64)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    if loss_name not in ("ce", "mse"):
        raise ValueError("loss_name must be 'ce' or 'mse'")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(X) == 0 or len(X) != len(y):
        raise ValueError("X and y must have the same non-zero number of samples")

    model.eval()
    total_loss = 0.0
    confusion = torch.zeros((7, 7), dtype=torch.int64, device=X.device)
    for start in range(0, len(X), batch_size):
        xb = X[start : start + batch_size]
        yb = y[start : start + batch_size]
        logits = model(xb)
        if loss_name == "ce":
            total_loss += float(F.cross_entropy(logits, yb, reduction="sum").item())
        else:
            total_loss += float(
                F.mse_loss(
                    logits,
                    F.one_hot(yb, num_classes=logits.shape[1]).to(dtype=logits.dtype),
                    reduction="sum",
                ).item()
            )
        predictions = logits.argmax(dim=1)
        flat = yb.to(torch.int64) * 7 + predictions.to(torch.int64)
        confusion += torch.bincount(flat, minlength=49).reshape(7, 7)

    accuracy = float(confusion.diag().sum().item() / len(X))
    confusion_np = confusion.cpu().numpy()
    divisor = len(X) if loss_name == "ce" else len(X) * 7
    return {
        "loss": total_loss / divisor,
        "acc": accuracy,
        "macro_f1": macro_f1_from_confusion(confusion_np),
    }


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ bạn lấy trung bình thế nào).
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    if loss_name == "mse":
        targets = F.one_hot(y, num_classes=logits.shape[1]).to(dtype=logits.dtype)
        return F.mse_loss(logits, targets, reduction="mean")
    raise ValueError(f"Unsupported loss {loss_name!r}; choose 'ce' or 'mse'")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "clip_fraction": [...],
                     "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}
    (tên khoá của summary trùng tên cột trong experiments.xlsx)

    Các bước:
      0. set_seed(cfg["seed"]); tạo model = MLP(...), assert count_params(model) == EXPECTED_PARAMS[hidden]
         chuyển model lên device; tạo optimizer = build_optimizer(...)
         nếu precision == "fp16": scaler = torch.amp.GradScaler(...)
      1. step0_loss = evaluate(model, X_val, y_val)["loss"]   # TRƯỚC bước cập nhật đầu tiên; kỳ vọng ≈ ln 7
      2. for epoch in 1..epochs:
           model.train()
           for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator):
               with torch.autocast(...)  nếu precision != "fp32":   # chỉ bọc forward + loss
                   logits = model(xb); loss = compute_loss(logits, yb, cfg["loss"])
               optimizer.zero_grad(set_to_none=True)
               backward (qua scaler nếu fp16)
               nếu fp16 và có clip: scaler.unscale_(optimizer)  TRƯỚC khi clip
               gn = clip_gradients(model.parameters(), cfg["clip_norm"])   # chuẩn TRƯỚC khi cắt; ghi lại
               bước cập nhật (scaler.step(optimizer); scaler.update() nếu fp16, ngược lại optimizer.step())
               nếu loss là NaN/inf: đặt diverged=True và dừng sớm, ĐỪNG để notebook treo
           cuối epoch (dùng evaluate, chế độ eval):
               train_loss trên toàn bộ train (hoặc 1 tập con CỐ ĐỊNH ~50 000 mẫu), val_loss/val_acc/val_macro_f1
               grad_norm trung bình của epoch; thời gian epoch (torch.cuda.synchronize() nếu dùng GPU)
               nếu val_loss tốt nhất từ trước tới giờ: lưu best_state (bản sao state_dict) và best_epoch
      3. tổng hợp summary tại best_epoch (val_acc, val_macro_f1 lấy ở best_epoch); peak_mem_MB nếu có GPU
    TUYỆT ĐỐI không đưa X_eval vào hàm này để chọn epoch/cấu hình. Chỉ dùng val.
    """
    config = {**DEFAULT_CFG, **cfg}
    if config["lr"] is None or not math.isfinite(float(config["lr"])) or config["lr"] <= 0:
        raise ValueError("cfg['lr'] must be positive; select it using validation data")
    if config["batch"] <= 0 or config["epochs"] <= 0:
        raise ValueError("cfg['batch'] and cfg['epochs'] must be positive")
    precision = config["precision"]
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be 'fp32', 'fp16', or 'bf16'")

    X_tr, y_tr = data["X_tr"], data["y_tr"]
    X_val, y_val = data["X_val"], data["y_val"]
    device = X_tr.device
    if any(data[key].device != device for key in ("y_tr", "X_val", "y_val")):
        raise ValueError("Training and validation tensors must be on the same device")
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("FP16 autocast training requires a CUDA device")
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA device does not support bfloat16")
    if precision == "bf16" and device.type not in ("cuda", "cpu"):
        raise ValueError(f"BF16 autocast is not configured for {device.type}")

    set_seed(int(config["seed"]))
    hidden = tuple(config["hidden"])
    model = MLP(
        hidden=hidden,
        dropout=float(config["dropout"]),
        init=config["init"],
    ).to(device)
    expected_params = EXPECTED_PARAMS.get(hidden)
    if expected_params is not None and count_params(model) != expected_params:
        raise AssertionError(
            f"Architecture {hidden} has {count_params(model)} parameters; "
            f"expected {expected_params}"
        )
    optimizer = build_optimizer(
        config["optimizer"],
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
        momentum=float(config["momentum"]),
    )

    step0 = evaluate(model, X_val, y_val, loss_name=config["loss"])
    step0_loss = step0["loss"]
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "clip_fraction": [],
        "epoch_time_s": [],
    }
    best_val_loss = step0_loss
    best_epoch = 0
    best_metrics = step0
    best_state = copy.deepcopy(model.state_dict())
    diverged = not math.isfinite(step0_loss)
    scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None
    autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(config["seed"]))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, int(config["epochs"]) + 1):
        if diverged:
            break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()
        model.train()
        step_norms = []
        clipped_steps = 0
        for xb, yb in iterate_batches(
            X_tr, y_tr, int(config["batch"]), generator=generator, shuffle=True
        ):
            optimizer.zero_grad(set_to_none=True)
            if autocast_dtype is None:
                logits = model(xb)
                loss = compute_loss(logits, yb, config["loss"])
            else:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, config["loss"])
            if not bool(torch.isfinite(loss).item()):
                diverged = True
                break
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

            grad_norm = clip_gradients(model.parameters(), config["clip_norm"])
            if not math.isfinite(grad_norm):
                diverged = True
                optimizer.zero_grad(set_to_none=True)
                break
            step_norms.append(grad_norm)
            if config["clip_norm"] is not None and grad_norm > config["clip_norm"]:
                clipped_steps += 1
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            parameters_are_finite = torch.stack(
                [torch.isfinite(parameter).all() for parameter in model.parameters()]
            ).all()
            if not bool(parameters_are_finite.item()):
                diverged = True
                model.load_state_dict(best_state)
                break

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_time = time.perf_counter() - epoch_start
        train_metrics = evaluate(model, X_tr, y_tr, loss_name=config["loss"])
        val_metrics = evaluate(model, X_val, y_val, loss_name=config["loss"])
        metrics_are_finite = all(
            math.isfinite(metrics[key])
            for metrics in (train_metrics, val_metrics)
            for key in ("loss", "acc", "macro_f1")
        )
        if not metrics_are_finite:
            diverged = True
            model.load_state_dict(best_state)
            train_metrics = evaluate(model, X_tr, y_tr, loss_name=config["loss"])
            val_metrics = best_metrics
        grad_norm_mean = float(np.mean(step_norms)) if step_norms else 0.0
        clip_fraction = clipped_steps / len(step_norms) if step_norms else 0.0
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["grad_norm"].append(grad_norm_mean)
        history["clip_fraction"].append(clip_fraction)
        history["epoch_time_s"].append(epoch_time)

        if math.isfinite(val_metrics["loss"]) and val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_metrics = val_metrics
            best_state = copy.deepcopy(model.state_dict())
        if diverged:
            break

    if history["epoch"]:
        final_train_loss = history["train_loss"][-1]
        final_val_loss = history["val_loss"][-1]
    else:
        final_train_loss = evaluate(model, X_tr, y_tr, loss_name=config["loss"])["loss"]
        final_val_loss = step0_loss
    peak_mem_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2))
        if device.type == "cuda"
        else 0.0
    )
    mean_epoch_time = (
        float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0
    )
    summary = {
        "step0_loss": step0_loss,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "val_acc": best_metrics["acc"],
        "val_macro_f1": best_metrics["macro_f1"],
        "time_per_epoch_s": mean_epoch_time,
        "peak_mem_MB": peak_mem_mb,
        "diverged": diverged,
    }
    return {
        "cfg": config,
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    row_ids = np.asarray(row_id)
    predictions = np.asarray(preds)
    if row_ids.ndim != 1 or predictions.ndim != 1 or len(row_ids) != len(predictions):
        raise ValueError("row_id and preds must be one-dimensional arrays of equal length")
    if not np.issubdtype(row_ids.dtype, np.integer):
        if not np.isfinite(row_ids).all() or not np.equal(row_ids, row_ids.astype(np.int64)).all():
            raise ValueError("row_id values must be integers")
    if not np.issubdtype(predictions.dtype, np.integer):
        if not np.isfinite(predictions).all() or not np.equal(predictions, predictions.astype(np.int64)).all():
            raise ValueError("predictions must be integer labels")
    row_ids = row_ids.astype(np.int64, copy=False)
    predictions = predictions.astype(np.int64, copy=False)
    if len(np.unique(row_ids)) != len(row_ids):
        raise ValueError("row_id values must be unique")
    if len(predictions) and (predictions.min() < 0 or predictions.max() > 6):
        raise ValueError("predictions must be labels in 0..6")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("row_id", "pred"))
        writer.writerows(zip(row_ids.tolist(), predictions.tolist()))


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    config = {**DEFAULT_CFG, **result.get("cfg", {}), **cfg}
    model = MLP(
        hidden=tuple(config["hidden"]),
        dropout=float(config["dropout"]),
        init=config["init"],
    ).to(data["X_eval"].device)
    model.load_state_dict(result["best_state"])
    predictions = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], predictions.cpu().numpy(), pred_path)

    repo_root = Path(__file__).resolve().parents[2]
    evaluation_script = repo_root / "scripts" / "evaluate.py"
    if not evaluation_script.is_file():
        raise FileNotFoundError(f"Evaluation script not found: {evaluation_script}")
    prediction_file = Path(pred_path).resolve()
    score_file = prediction_file.parent / "eval_result.json"
    command = [
        sys.executable,
        str(evaluation_script),
        "--pred",
        str(prediction_file),
        "--out",
        str(score_file),
    ]
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        command,
        cwd=repo_root,
        check=False,
        capture_output=True,
        encoding="utf-8",
        env=environment,
    )
    if completed.returncode:
        stdout = completed.stdout.encode("ascii", "backslashreplace").decode("ascii")
        stderr = completed.stderr.encode("ascii", "backslashreplace").decode("ascii")
        raise RuntimeError(
            f"Evaluation command failed with exit code {completed.returncode}: {command}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    print(f"Evaluation scores written to {score_file}")
