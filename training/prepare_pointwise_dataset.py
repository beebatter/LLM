#!/usr/bin/env python3
"""将 enriched 大 JSONL 转换为点式 (pointwise) 训练数据.

输入特点:
  - 文件可能很大 (流式读取)
  - 可能存在一行多个 JSON 对象串联的情况  {...}{...}{...}
  - 关键字段 (可映射):
      problem_name / qid / group  -> 分组字段
      query                       -> (候选文本 or 需要映射成 candidate)
      doc / text                  -> 另一候选字段 (视情况)
      label (0/1)                 -> 二分类标签
      weight / score              -> 连续 teacher 分数

输出：标准 pointwise JSONL：
  {"query": <上下文>, "candidate": <候选>, "label": 0/1, "score": <可选>, "group": <分组id>}

典型映射（推荐）：
  query(输出)    <- problem_name (问题 ID 作为查询)
  candidate      <- query(原始文件中的公式字符串)
  label          <- label
  score          <- weight
  group          <- problem_name

用法示例：
python -m LLM.training.prepare_pointwise_dataset \\
  --input /root/autodl-tmp/Training/datasets/train_full.enriched.jsonl \\
  --output-train /root/autodl-tmp/Training/datasets/pointwise.train.jsonl \\
  --output-dev /root/autodl-tmp/Training/datasets/pointwise.dev.jsonl \\
  --train-ratio 0.98 \\
  --map-query-from problem_name \\
  --map-candidate-from query \\
  --label-field label \\
  --score-field weight \\
  --group-field problem_name \\
  --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass
class OutSample:
    query: str
    candidate: str
    label: Optional[int]
    score: Optional[float]
    group: str

    def to_json(self) -> str:
        obj = {"query": self.query, "candidate": self.candidate, "group": self.group}
        if self.label is not None:
            obj["label"] = self.label
        if self.score is not None:
            obj["score"] = self.score
        return json.dumps(obj, ensure_ascii=False)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-train", required=True)
    ap.add_argument("--output-dev", required=True)
    ap.add_argument("--train-ratio", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--map-query-from", default="problem_name")
    ap.add_argument("--map-candidate-from", default="query")
    ap.add_argument("--label-field", default="label")
    ap.add_argument("--score-field", default="weight")
    ap.add_argument("--group-field", default="problem_name")
    ap.add_argument("--max", type=int, default=0, help="调试: 限制读取样本数 (0=不限制)")
    ap.add_argument("--filter-unlabeled", action="store_true", help="如果设置则丢弃没有 label 的样本")
    ap.add_argument("--min-group-size", type=int, default=1, help="最小保留组大小")
    ap.add_argument("--shuffle-groups", action="store_true")
    return ap.parse_args()


JSON_OBJ_PATTERN = re.compile(r"{.*?}(?=\s*{|\s*$)")


def iter_json_objects(line: str):
    line = line.strip()
    if not line:
        return
    # 尝试直接解析
    try:
        obj = json.loads(line)
        yield obj
        return
    except Exception:
        pass
    # 回退: 正则切分多个对象
    for m in JSON_OBJ_PATTERN.finditer(line):
        frag = m.group(0)
        try:
            yield json.loads(frag)
        except Exception:
            continue


def load_samples(args) -> List[OutSample]:
    out: List[OutSample] = []
    cnt = 0
    with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for j in iter_json_objects(line):
                q_raw = j.get(args.map_query_from) or j.get("problem_name") or j.get("qid") or ""
                cand_raw = j.get(args.map_candidate_from) or j.get("query") or j.get("doc") or j.get("text") or ""
                group = j.get(args.group_field) or q_raw or "g"
                if not q_raw or not cand_raw:
                    continue
                label_val = j.get(args.label_field)
                score_val = j.get(args.score_field)
                label: Optional[int] = None
                if label_val is not None:
                    try:
                        label = int(1 if float(label_val) >= 0.5 else 0)
                    except Exception:
                        label = None
                score: Optional[float] = None
                if score_val is not None:
                    try:
                        score = float(score_val)
                    except Exception:
                        score = None
                if args.filter_unlabeled and label is None:
                    continue
                out.append(OutSample(q_raw, cand_raw, label, score, str(group)))
                cnt += 1
                if args.max > 0 and cnt >= args.max:
                    return out
    return out


def group_filter(samples: List[OutSample], min_size: int) -> List[OutSample]:
    if min_size <= 1:
        return samples
    from collections import defaultdict
    g = defaultdict(list)
    for s in samples:
        g[s.group].append(s)
    keep = []
    for k, vs in g.items():
        if len(vs) >= min_size:
            keep.extend(vs)
    return keep


def main():
    args = parse_args()
    random.seed(args.seed)
    samples = load_samples(args)
    samples = group_filter(samples, args.min_group_size)
    # group -> list idx
    from collections import defaultdict
    groups = defaultdict(list)
    for s in samples:
        groups[s.group].append(s)
    group_ids = list(groups.keys())
    if args.shuffle_groups:
        random.shuffle(group_ids)
    else:
        group_ids.sort()
    train_cut = int(len(group_ids) * args.train_ratio)
    train_ids = set(group_ids[:train_cut])
    train_out = args.output_train
    dev_out = args.output_dev
    os.makedirs(os.path.dirname(train_out), exist_ok=True)
    os.makedirs(os.path.dirname(dev_out), exist_ok=True)
    tcnt = dcnt = 0
    with open(train_out, "w", encoding="utf-8") as ft, open(dev_out, "w", encoding="utf-8") as fd:
        for gid in group_ids:
            target_f = ft if gid in train_ids else fd
            for s in groups[gid]:
                target_f.write(s.to_json() + "\n")
                if gid in train_ids:
                    tcnt += 1
                else:
                    dcnt += 1
    # 统计
    pos = sum(1 for s in samples if s.label == 1)
    neg = sum(1 for s in samples if s.label == 0)
    print(json.dumps({
        "total_samples": len(samples),
        "groups": len(group_ids),
        "train_groups": len(train_ids),
        "dev_groups": len(group_ids) - len(train_ids),
        "train_samples": tcnt,
        "dev_samples": dcnt,
        "pos": pos, "neg": neg,
        "pos_ratio": pos / max(1, pos + neg)
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
