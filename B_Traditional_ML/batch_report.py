"""Summarise saved batch experiment results without retraining any models."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


METRICS = ("accuracy", "precision", "recall", "f1", "specificity", "roc_auc")


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_events(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metric(value, percentage=True):
    if value is None or value == "":
        return "—"
    return f"{float(value) * 100:.2f}%" if percentage else f"{float(value):.4f}"


def collect_folds(root):
    """Read saved record and metric files; values remain tied to their run path."""
    rows = []
    for path in sorted((root / "runs").rglob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            record = json.loads(data["record_json"].item())
        config = record.get("config", {})
        val = record.get("metrics", {})
        train = record.get("train_metrics", {})
        rows.append({"path": path.relative_to(root).as_posix(), "config_id": config.get("id", ""),
                     "family": config.get("family", ""), "status": record.get("status", ""),
                     **{name: val.get(name, "") for name in METRICS},
                     "confusion_matrix": json.dumps(val.get("confusion_matrix", "")),
                     "train_accuracy": train.get("accuracy", ""), "train_f1": train.get("f1", ""),
                     "fit_seconds": record.get("fit_seconds", ""),
                     "predict_seconds": record.get("predict_seconds", ""),
                     "error_type": record.get("error_type", ""), "error": record.get("error", ""),
                     "warnings": json.dumps(record.get("warnings", []), ensure_ascii=False)})
    return rows


def write_fold_table(root, rows):
    if not rows:
        return
    path = root / "fold_metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def append_summary(lines, title, rows):
    lines.extend([f"## {title}", ""])
    if not rows:
        lines.extend(["该阶段没有可用汇总。", ""])
        return
    lines.extend(["| 配置 | 评估折完成数 | 平均 accuracy | 折间样本 SD | 平均 F1 | 平均 ROC-AUC | 训练 accuracy | 拟合总秒数 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        lines.append(f"| `{row['id']}` | {row['completed_folds']}/3 | {metric(row.get('mean_accuracy'))} | "
                     f"{metric(row.get('std_accuracy'))} | {metric(row.get('mean_f1'))} | "
                     f"{metric(row.get('mean_roc_auc'), False)} | {metric(row.get('mean_train_accuracy'))} | "
                     f"{float(row.get('total_fit_seconds') or 0):.2f} |")
    lines.append("")


def write_report(root):
    root = Path(root)
    progress = read_json(root / "progress.json", {})
    if progress.get("state") != "finished":
        raise ValueError("The batch has not finished; wait for the runner before generating the final report")
    manifest = read_json(root / "manifest.json", {})
    selection = read_json(root / "selection.json", {})
    events = read_events(root / "events.jsonl")
    folds = collect_folds(root)
    write_fold_table(root, folds)
    lines = ["# 传统机器学习批量实验记录", "", "## 数据、方法与范围", "",
             "- K20 使用保存的 training 内三折；K30、K40 在各自 training 内按 seed 67 生成分层三折。模型随机种子为 67。",
             "- 本批对 K20 进行多配置筛选；先前已在全部 training 上查看符号交互与标签的关联，且原 validation 参与过开发。分数用于开发比较。",
             "- Nested CV 在筛选后固定方法族与表示，再在内折选择其参数；由于此前筛选与 training 全量探索，nested 结果仍有探索背景。",
             "- 公共加载器会读取共享 validation/test 数组；所有模型只使用 training 内部的行。local test 和 hidden 不参与评分。",
             "- 拟合时间包含特征构造、训练折缩放及模型拟合；先前 SVM/XGBoost 记录的计时范围不同。", ""]
    lines.extend(["## 筛选与预算", "",
                  f"- K20 三折完成配置：{progress.get('complete_screen_configs', '—')} / {progress.get('total_screen_configs', '—')}。",
                  "- 跨数据集候选：" + (", ".join(f"`{c['id']}`" for c in selection.get("chosen", [])) or "无完整候选") + "。",
                  "- 阶段墙钟时间（秒）：" + ", ".join(f"{key}={float(value):.2f}" for key, value in progress.get("stage_seconds", {}).items()) + "。",
                  "- 配置以平均 accuracy、平均 F1 和配置 ID 依次排序；跨数据集保留两个不同方法族。", ""])
    for key, title in (("screen", "K20 全部配置筛选"), ("30", "K30 固定候选"), ("40", "K40 固定候选")):
        append_summary(lines, title, read_csv(root / f"summary_{key}.csv"))
    lines.extend(["## 固定候选与参照的逐折结果", "",
                  "| 数据集 | 配置 | 折 | accuracy | F1 | ROC-AUC | 训练 accuracy |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    chosen_ids = {item["id"] for item in selection.get("chosen", [])}
    chosen_ids.add("svm_previous_rbf_c100_g0.2")
    for row in folds:
        path = Path(row["path"])
        parts = path.parts
        if (len(parts) != 5 or parts[0] != "runs" or parts[1] not in ("screen", "30", "40")
                or row["config_id"] not in chosen_ids or row["status"] != "ok"):
            continue
        fold = int(path.stem.removeprefix("fold")) + 1
        lines.append(f"| K{parts[2]} | `{row['config_id']}` | {fold} | {metric(row['accuracy'])} | "
                     f"{metric(row['f1'])} | {metric(row['roc_auc'], False)} | {metric(row['train_accuracy'])} |")
    lines.append("")
    lines.extend(["## K20 Nested CV", ""])
    nested = read_csv(root / "summary_nested.csv")
    if nested:
        lines.extend(["| 外折 | 方法族 | 内折选定配置 | 外折 accuracy | F1 | ROC-AUC | 训练 accuracy |",
                      "|---:|---|---|---:|---:|---:|---:|"])
        for row in nested:
            lines.append(f"| {int(row['outer']) + 1} | {row['family']} | `{row['selected_id']}` | "
                         f"{metric(row.get('accuracy'))} | {metric(row.get('f1'))} | "
                         f"{metric(row.get('roc_auc'), False)} | {metric(row.get('train_accuracy'))} |")
        lines.append("")
        count = Counter(row["family"] for row in nested)
        for selected in selection.get("chosen", []):
            count.setdefault(selected["family"], 0)
        lines.append("- 外折完成数：" + ", ".join(f"{key} {value}/3" for key, value in sorted(count.items())) + "。")
        for family in sorted(count):
            family_rows = [row for row in nested if row["family"] == family]
            if len(family_rows) == 3 and {row["outer"] for row in family_rows} == {"0", "1", "2"}:
                accuracy = [float(row["accuracy"]) for row in family_rows]
                f1 = [float(row["f1"]) for row in family_rows]
                lines.append(f"- {family}：三个外折平均 accuracy {metric(np.mean(accuracy))}，"
                             f"F1 {metric(np.mean(f1))}；accuracy 折间样本 SD {metric(np.std(accuracy, ddof=1))}。")
    else:
        lines.append("- 该阶段无已完成外折；具体原因见事件记录。")
    lines.extend(["", "## 失败、跳过与未完成项", ""])
    failures = [e for e in events if e.get("type") == "fit" and e.get("status") != "ok"]
    skips = [e for e in events if e.get("type") == "stage_skipped"]
    if failures:
        for event in failures:
            error = (event.get("error") or "未记录具体原因").replace("\n", " ").replace("|", "\\|")
            lines.append(f"- {event.get('stage')} / `{event.get('config_id')}` / `{event.get('output')}`: "
                         f"{event.get('status')}，{error}")
    if skips:
        lines.extend(f"- {e.get('stage')}：{e.get('reason')}" for e in skips)
    if not failures and not skips:
        lines.append("- 已执行拟合未记录失败；尚未执行或未完成的配置见上表及各阶段 summary CSV。")
    warning_rows = [row for row in folds if json.loads(row["warnings"])]
    if warning_rows:
        lines.append(f"- {len(warning_rows)} 条逐折记录包含运行警告，详见 `fold_metrics.csv` 的 `warnings` 列。")
    else:
        lines.append("- 已保存的逐折记录未包含训练警告。")
    lines.extend(["", "## 文件与复现", "",
                  "- `manifest.json` 记录参数网格、数据/源代码指纹、环境版本、随机种子和预算；`folds/` 保存折索引。",
                  "- `runs/` 保存每折预测、分数、标签与指标；`fold_metrics.csv` 从保存的记录汇总逐折指标；各 `summary_*.csv` 保存所有阶段汇总。",
                  "- `events.jsonl` 记录拟合状态、失败原因与实际耗时。`progress.json` 记录阶段时间和完成数量。",
                  "- 在仓库根目录运行 `python -m B_Traditional_ML.batch_run --output-dir B_Traditional_ML/results/<new-dir>`；执行后用 `python -m B_Traditional_ML.batch_report --input-dir B_Traditional_ML/results/<new-dir>` 写本报告。",
                  "- [准确率比较图](figures/batch_accuracy.png) 的绘图 notebook 为 `B_Traditional_ML/plot_batch_experiments.ipynb`；运行其全部单元以从保存的汇总重画图。",
                  "- 本实验的所有结果均为开发阶段评估；跨机器复现需在另一环境实际运行确认。", ""])
    (root / "实验记录.md").write_text("\n".join(lines), encoding="utf-8")
    return {"fold_records": len(folds), "fit_events": sum(e.get("type") == "fit" for e in events),
            "failed_fit_events": len(failures), "manifest_created_at": manifest.get("created_at_utc")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_report(args.input_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
