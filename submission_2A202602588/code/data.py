"""Utilities for loading, splitting, normalizing, and batching the lab data.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    from pathlib import Path

    directory = Path(processed_dir)
    train_path = directory / "train.npz"
    eval_path = directory / "eval.npz"
    if not train_path.is_file() or not eval_path.is_file():
        raise FileNotFoundError(
            f"Expected train.npz and eval.npz in {directory.resolve()}. "
            "Run scripts/split_data.py from the repository root first."
        )

    with np.load(train_path) as train_data, np.load(eval_path) as eval_data:
        X_train = np.asarray(train_data["X"], dtype=np.float32)
        y_train = np.asarray(train_data["y"], dtype=np.int64)
        X_eval = np.asarray(eval_data["X"], dtype=np.float32)
        y_eval = np.asarray(eval_data["y"], dtype=np.int64)
        eval_row_id = np.asarray(eval_data["row_id"], dtype=np.int64)

    for name, X, y in (("train", X_train, y_train), ("eval", X_eval, y_eval)):
        if X.ndim != 2 or X.shape[1] != 54:
            raise ValueError(f"{name} features must have shape (N, 54), got {X.shape}")
        if y.shape != (len(X),):
            raise ValueError(f"{name} labels must have shape ({len(X)},), got {y.shape}")
        if not np.isfinite(X).all():
            raise ValueError(f"{name} features contain non-finite values")
        if len(y) and (y.min() < 0 or y.max() > 6):
            raise ValueError(f"{name} labels must be in 0..6")
    if eval_row_id.shape != y_eval.shape:
        raise ValueError("eval row_id and labels must have the same length")
    if len(np.unique(eval_row_id)) != len(eval_row_id):
        raise ValueError("eval row_id values must be unique")

    return X_train, y_train, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    from sklearn.model_selection import train_test_split

    X_tr, X_val, y_tr, y_val = train_test_split(
        X,
        y,
        test_size=val_fraction,
        random_state=seed,
        stratify=y,
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    """
    X_tr = np.asarray(X_tr, dtype=np.float32)
    if X_tr.ndim != 2 or X_tr.shape[1] < N_NUMERIC:
        raise ValueError(f"X_tr must be a 2-D array with at least {N_NUMERIC} columns")
    mean = X_tr[:, :N_NUMERIC].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = X_tr[:, :N_NUMERIC].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std == 0] = 1.0
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    X = np.asarray(X, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] < N_NUMERIC:
        raise ValueError(f"X must be a 2-D array with at least {N_NUMERIC} columns")
    if mean.shape != (N_NUMERIC,) or std.shape != (N_NUMERIC,):
        raise ValueError(f"mean and std must both have shape ({N_NUMERIC},)")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("mean and std must be finite, and std must be positive")
    standardized = X.copy()
    standardized[:, :N_NUMERIC] = (standardized[:, :N_NUMERIC] - mean) / std
    return standardized


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(
        X_train_full, y_train_full, val_fraction=val_fraction, seed=seed
    )
    mean, std = fit_standardizer(X_tr)
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)

    target_device = torch.device(device)
    data = {
        "X_tr": torch.as_tensor(X_tr, dtype=torch.float32, device=target_device),
        "y_tr": torch.as_tensor(y_tr, dtype=torch.int64, device=target_device),
        "X_val": torch.as_tensor(X_val, dtype=torch.float32, device=target_device),
        "y_val": torch.as_tensor(y_val, dtype=torch.int64, device=target_device),
        "X_eval": torch.as_tensor(X_eval, dtype=torch.float32, device=target_device),
        "y_eval": torch.as_tensor(y_eval, dtype=torch.int64, device=target_device),
        "eval_row_id": eval_row_id,
        "standardizer_mean": mean,
        "standardizer_std": std,
    }
    majority_label = int(np.bincount(y_tr, minlength=7).argmax())
    majority_accuracy = float(np.mean(y_val == majority_label))
    print(
        f"train: {tuple(data['X_tr'].shape)}, "
        f"val: {tuple(data['X_val'].shape)}, "
        f"eval: {tuple(data['X_eval'].shape)}"
    )
    print(f"Majority-class validation accuracy: {majority_accuracy:.4f}")
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Lô cuối được giữ nguyên dù nhỏ hơn batch_size để không bỏ sót mẫu.
    """
    if X.ndim == 0 or y.ndim == 0 or len(X) == 0 or len(X) != len(y):
        raise ValueError("X and y must have the same non-zero number of samples")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    n_samples = len(X)
    if shuffle:
        permutation = torch.randperm(
            n_samples, generator=generator, device=X.device
        )
    else:
        permutation = torch.arange(n_samples, device=X.device)
    for start in range(0, n_samples, batch_size):
        indices = permutation[start : start + batch_size]
        yield X[indices], y[indices]
