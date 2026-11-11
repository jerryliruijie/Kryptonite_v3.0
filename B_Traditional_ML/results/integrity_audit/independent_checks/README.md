# 独立检查：划分、保存的指标与重复向量

使用原始数组、固定划分和保存的预测独立复核 54 个 checkpoint：9 个 CatBoost training 3-fold、27 个 learning curve、18 个 K30/K40 模型 seed 实验。逐项检查训练与评分行、标签映射、accuracy、precision、recall、F1、specificity、ROC-AUC 和混淆矩阵。结果保存在 `report.json`。

K20/K30/K40 的 shared training、validation、local-test 行号两两不重叠；training 与 validation 的原始完整特征没有精确重复。K20 的 validation 中有 19 条 sign 向量在 training 出现过，18 条预测正确、1 条错误；其余 2,381 条是新的 sign 组合，2,252 条预测正确，accuracy 94.5821%。K30/K40 的 validation sign 向量与 training 无精确重复。

这里的 sign 向量重复指不同原始输入经符号映射后相同。只读取 local-test 输入特征以统计重复，未索引 local-test 标签、训练或评分；未读取 hidden。学习曲线与模型选择已经使用开发数据，结果仍需以最终评估衡量。

在仓库根目录重新运行：

```powershell
.\.venv\Scripts\python.exe B_Traditional_ML/results/integrity_audit/independent_checks/independent_audit.py
```

脚本打印 JSON。可添加 `--output <new-file.json>` 保存新结果，现有文件不会覆盖。脚本 SHA-256：`bdd6ec01d0cedaf64d175407331c53ea7f3c83d3fc4426db115537c1a8701920`。移动到此目录后已重新执行，输出与 `report.json` 内容逐项一致。
