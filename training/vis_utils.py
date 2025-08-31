from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

# Use non-interactive backend for headless servers
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def save_metrics_json(metrics: Dict, out_dir: Path, name: str = "metrics.json") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    with p.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return p


def plot_curves(curves: Dict[str, Sequence[float]], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, v in curves.items():
        if not v:
            continue
        plt.figure()
        xs = list(range(1, len(v) + 1))
        plt.plot(xs, list(v), marker="o")
        plt.xlabel("epoch")
        plt.ylabel(k)
        plt.title(f"{prefix} - {k}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_{k}.png", dpi=150)
        plt.close()


def plot_hist(pos: Sequence[float] | None, neg: Sequence[float] | None, out_dir: Path, name: str = "score_hist.png") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    if pos:
        plt.hist(list(pos), bins=50, alpha=0.6, label="pos")
    if neg:
        plt.hist(list(neg), bins=50, alpha=0.6, label="neg")
    if pos or neg:
        plt.legend()
    plt.title("Score distribution")
    plt.tight_layout()
    plt.savefig(out_dir / name, dpi=150)
    plt.close()


def plot_topk_curve(hits_at_k: Sequence[float], out_dir: Path, name: str = "topk_curve.png") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np
    if not hits_at_k:
        return
    xs = list(range(1, len(hits_at_k) + 1))
    plt.figure()
    plt.plot(xs, list(hits_at_k), marker="o")
    plt.xlabel("K")
    plt.ylabel("Recall@K")
    plt.title("Top-K Hit Rate Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / name, dpi=150)
    plt.close()


def plot_bucket_box(scores_by_bucket: Dict[str, Sequence[float]], out_dir: Path, name: str = "bucket_box.png") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not scores_by_bucket:
        return
    labels = list(scores_by_bucket.keys())
    data = [list(scores_by_bucket[k]) for k in labels]
    plt.figure(figsize=(max(6, len(labels) * 0.8), 4))
    plt.boxplot(data, labels=labels, vert=True, showfliers=False)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Scores")
    plt.title("Scores by Prefix Bucket")
    plt.tight_layout()
    plt.savefig(out_dir / name, dpi=150)
    plt.close()


def save_bucket_recalls(recalls_by_bucket: Dict[str, float], out_dir: Path, name: str = "bucket_recalls.json") -> Path:
    return save_metrics_json({"recall_by_bucket": recalls_by_bucket}, out_dir, name)
