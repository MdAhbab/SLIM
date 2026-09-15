"""
Comprehensive metrics utilities.

Computes the standard evaluation metrics used across all stages:
  BCE, MSE, Frobenius, F1, Accuracy, Balanced Accuracy,
  Precision, Recall, AUROC, AUPR, MCC

Usage:
    from src.metrics import compute_all_metrics, format_metrics_report
"""

import numpy as np

NOT_AVAILABLE_MSG = "Score not available, as the model is not designed to produce this metric."


def _safe(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), return None on any error."""
    try:
        val = fn(*args, **kwargs)
        if isinstance(val, float) and np.isnan(val):
            return None
        return val
    except Exception:
        return None


def compute_all_metrics(
    labels,
    probs,
    preds=None,
    bce_loss_val=None,
    mse_loss_val=None,
    frob_loss_val=None,
    is_multilabel=False,
):
    """
    Compute all 11 metrics from ground-truth labels and predicted probabilities.

    Parameters
    ----------
    labels : array-like, shape (N,)
        Ground-truth binary labels (0 or 1).
    probs : array-like, shape (N,)
        Predicted probabilities (after sigmoid).
    preds : array-like, shape (N,), optional
        Binary predictions (>= 0.5). Computed from probs if not given.
    bce_loss_val : float, optional
        Pre-computed BCE loss value.
    mse_loss_val : float, optional
        Pre-computed MSE loss value.
    frob_loss_val : float, optional
        Pre-computed Frobenius penalty value.
    is_multilabel : bool
        If True, binary classification metrics (F1, Acc, BalAcc, Precision,
        Recall, MCC) are marked as not available.

    Returns
    -------
    dict with keys:
        bce, mse, frobenius, f1, accuracy, balanced_accuracy,
        precision, recall, auroc, aupr, mcc
        Values are float or None (if not applicable).
    """
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        accuracy_score,
        balanced_accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        matthews_corrcoef,
    )

    labels = np.asarray(labels).ravel()
    probs = np.asarray(probs).ravel()
    if preds is None:
        preds = (probs >= 0.5).astype(int)
    else:
        preds = np.asarray(preds).ravel().astype(int)
    labels_int = labels.astype(int)

    result = {
        "bce": bce_loss_val,
        "mse": mse_loss_val,
        "frobenius": frob_loss_val,
    }

    # Ranking metrics (always applicable for binary or flattened multi-label)
    result["auroc"] = _safe(roc_auc_score, labels_int, probs)
    result["aupr"] = _safe(average_precision_score, labels_int, probs)

    if is_multilabel:
        # Binary classification metrics don't apply to multi-label
        for key in ("f1", "accuracy", "balanced_accuracy", "precision", "recall", "mcc"):
            result[key] = None
    else:
        result["f1"] = _safe(f1_score, labels_int, preds, zero_division=0)
        result["accuracy"] = _safe(accuracy_score, labels_int, preds)
        result["balanced_accuracy"] = _safe(balanced_accuracy_score, labels_int, preds)
        result["precision"] = _safe(precision_score, labels_int, preds, zero_division=0)
        result["recall"] = _safe(recall_score, labels_int, preds, zero_division=0)
        result["mcc"] = _safe(matthews_corrcoef, labels_int, preds)

    return result


def format_metrics_report(m, prefix="", indent=2):
    """
    Format a metrics dict as a readable multi-line string.

    Parameters
    ----------
    m : dict
        Output of compute_all_metrics().
    prefix : str
        Optional prefix like "[Train]" or "[Val]".
    indent : int
        Number of leading spaces.

    Returns
    -------
    str
    """
    pad = " " * indent
    pfx = f"{prefix} " if prefix else ""
    lines = []

    def _fmt(key, label):
        val = m.get(key)
        if val is None:
            lines.append(f"{pad}{pfx}{label:20s}: {NOT_AVAILABLE_MSG}")
        else:
            lines.append(f"{pad}{pfx}{label:20s}: {val:.4f}")

    _fmt("bce", "BCE Loss")
    _fmt("mse", "MSE Loss")
    _fmt("frobenius", "Frobenius Loss")
    _fmt("auroc", "AUROC")
    _fmt("aupr", "AUPR")
    _fmt("accuracy", "Accuracy")
    _fmt("balanced_accuracy", "Balanced Accuracy")
    _fmt("precision", "Precision")
    _fmt("recall", "Recall")
    _fmt("f1", "F1 Score")
    _fmt("mcc", "MCC")

    return "\n".join(lines)


def format_epoch_line(epoch, max_epochs, train_m, val_m, elapsed):
    """
    Format a compact single-line epoch summary with key metrics.

    Parameters
    ----------
    epoch : int
    max_epochs : int
    train_m : dict from compute_all_metrics
    val_m : dict from compute_all_metrics
    elapsed : float, seconds

    Returns
    -------
    str
    """

    def _v(m, key):
        v = m.get(key)
        return f"{v:.4f}" if v is not None else "N/A"

    return (
        f"Epoch {epoch:3d}/{max_epochs} | "
        f"AUROC {_v(train_m,'auroc')}/{_v(val_m,'auroc')} | "
        f"AUPR {_v(train_m,'aupr')}/{_v(val_m,'aupr')} | "
        f"Acc {_v(train_m,'accuracy')}/{_v(val_m,'accuracy')} | "
        f"F1 {_v(train_m,'f1')}/{_v(val_m,'f1')} | "
        f"MCC {_v(train_m,'mcc')}/{_v(val_m,'mcc')} | "
        f"BalAcc {_v(train_m,'balanced_accuracy')}/{_v(val_m,'balanced_accuracy')} | "
        f"{elapsed:.0f}s"
    )


def metrics_to_history_row(epoch, train_m, val_m, elapsed, test_m=None):
    """
    Build a history row dict containing all metrics for JSON serialization.
    """
    row = {"epoch": epoch, "time": round(elapsed, 1)}
    for prefix, m in [("train", train_m), ("val", val_m)]:
        for key in ("bce", "mse", "frobenius", "auroc", "aupr", "accuracy",
                     "balanced_accuracy", "precision", "recall", "f1", "mcc"):
            row[f"{prefix}_{key}"] = m.get(key)

    if test_m is not None:
        for key in ("bce", "mse", "frobenius", "auroc", "aupr", "accuracy",
                     "balanced_accuracy", "precision", "recall", "f1", "mcc"):
            row[f"test_{key}"] = test_m.get(key)

    return row


# ---------------------------------------------------------------------------
# Threshold selection, calibration and uncertainty
#
# Everything below works from saved per-example predictions, so none of it
# needs a GPU or a second training run.
# ---------------------------------------------------------------------------


def metrics_at_threshold(labels, probs, threshold):
    """All threshold-dependent metrics at one decision threshold."""
    labels = np.asarray(labels).ravel().astype(int)
    probs = np.asarray(probs).ravel()
    preds = (probs >= threshold).astype(int)
    m = compute_all_metrics(labels, probs, preds=preds)
    m["threshold"] = float(threshold)
    return m


def select_threshold(labels, probs, metric="mcc", n_steps=999):
    """Find the threshold maximising `metric` on the data given.

    Call this on VALIDATION predictions only. Choosing a threshold on the test
    set and then reporting test metrics at that threshold reports a number no
    one could obtain in practice.

    Returns (threshold, best_value).
    """
    from sklearn.metrics import f1_score, matthews_corrcoef, balanced_accuracy_score

    labels = np.asarray(labels).ravel().astype(int)
    probs = np.asarray(probs).ravel()
    scorers = {
        "mcc": matthews_corrcoef,
        "f1": lambda y, p: f1_score(y, p, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score,
    }
    if metric not in scorers:
        raise ValueError(f"metric must be one of {sorted(scorers)}")
    score_fn = scorers[metric]

    candidates = np.linspace(1.0 / (n_steps + 1), n_steps / (n_steps + 1), n_steps)
    best_threshold, best_value = 0.5, -np.inf
    for threshold in candidates:
        value = score_fn(labels, (probs >= threshold).astype(int))
        if value > best_value:
            best_threshold, best_value = float(threshold), float(value)
    return best_threshold, best_value


def expected_calibration_error(labels, probs, n_bins=15):
    """Expected calibration error with equal-width probability bins.

    The error is the average gap between predicted probability and observed
    frequency, weighted by how many examples fall in each bin. A perfectly
    calibrated model scores zero. Also returns the bins themselves, which are
    what a reliability diagram plots.
    """
    labels = np.asarray(labels).ravel().astype(float)
    probs = np.asarray(probs).ravel().astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(probs)

    ece, max_gap, rows = 0.0, 0.0, []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        count = int(in_bin.sum())
        if count == 0:
            rows.append({"bin_lower": float(lo), "bin_upper": float(hi),
                         "count": 0, "confidence": None, "accuracy": None})
            continue
        confidence = float(probs[in_bin].mean())
        observed = float(labels[in_bin].mean())
        gap = abs(confidence - observed)
        ece += (count / total) * gap
        max_gap = max(max_gap, gap)
        rows.append({"bin_lower": float(lo), "bin_upper": float(hi),
                     "count": count, "confidence": confidence,
                     "accuracy": observed})

    brier = float(np.mean((probs - labels) ** 2))
    return {"ece": float(ece), "max_calibration_error": float(max_gap),
            "brier": brier, "n_bins": n_bins, "bins": rows}


def bootstrap_ci(labels, probs, metric="aupr", n_resamples=1000, alpha=0.05,
                 seed=0, threshold=0.5):
    """Percentile bootstrap confidence interval for one metric.

    Resamples examples with replacement, so the interval reflects uncertainty
    from the finite test set. It says nothing about variability between
    training runs, which is what repeated seeds measure. Report both.
    """
    from sklearn.metrics import (
        average_precision_score, roc_auc_score, f1_score, matthews_corrcoef,
    )

    labels = np.asarray(labels).ravel().astype(int)
    probs = np.asarray(probs).ravel()
    scorers = {
        "aupr": lambda y, p: average_precision_score(y, p),
        "auroc": lambda y, p: roc_auc_score(y, p),
        "mcc": lambda y, p: matthews_corrcoef(y, (p >= threshold).astype(int)),
        "f1": lambda y, p: f1_score(y, (p >= threshold).astype(int),
                                    zero_division=0),
    }
    if metric not in scorers:
        raise ValueError(f"metric must be one of {sorted(scorers)}")
    score_fn = scorers[metric]

    rng = np.random.default_rng(seed)
    n = len(labels)
    values = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y = labels[idx]
        if y.min() == y.max():
            continue  # a resample with one class only cannot be scored
        values.append(score_fn(y, probs[idx]))

    values = np.asarray(values, dtype=float)
    return {
        "metric": metric,
        "point": float(score_fn(labels, probs)),
        "mean": float(values.mean()) if values.size else float("nan"),
        "lower": float(np.quantile(values, alpha / 2)) if values.size else float("nan"),
        "upper": float(np.quantile(values, 1 - alpha / 2)) if values.size else float("nan"),
        "n_resamples": int(values.size),
        "alpha": alpha,
    }
