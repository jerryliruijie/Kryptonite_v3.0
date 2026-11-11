# 周三会议：B 的讲解与问答

**2026 年 11 月 11 日，周三**  
英文共享材料见同目录 `Wednesday_Group_Meeting_Brief.md`。这份讲稿给自己准备，会议进度及 A、C、D 的完成情况由会上确认。

## 会前准备

1. 打开英文会议材料、learning curve 图和原始结果记录。建议以 5–7 分钟介绍 B，再留时间讨论。
2. 确认组员能否拿到当前版本：`B_Traditional_ML/` 目前在本机尚未被 Git 跟踪，组员直接 pull 现有远端不会自动得到这些结果。可自行检查后 commit/push，或单独发送会议材料和图。
3. 找到 `B_Traditional_ML/results/learning_curve/subsets/`，准备向 A 展示它包含的 25%、50%、100% training 嵌套子集索引。
4. 留出时间让 A、C、D 各自汇报；请一人记录决定、负责人和下一次更新时间。

## 1. 开场：我们的问题是什么

可以这样说：

> I worked on traditional machine-learning methods for Kryptonite-20, 30 and 40. I compared several models and looked at how the representation of the input affects performance. The most promising current candidate is CatBoost using sign features. I will explain the comparison, the learning curve, and what we still need to agree as a group.

中文理解：这段交代自己的范围、问题、当前候选和接下来的讨论。结果属于 B 的开发实验，组员自己的最佳结果和当前进度需要他们报告。

## 2. 先解释数据规模和训练划分

> K20 has 12,000 labelled examples with 20 features. K30 has 20,000 examples with 30 features, and K40 has 32,000 examples with 40 features. We use the same stratified 70/20/10 split with seed 67. For K20, that gives us 8,400 training examples, 2,400 validation examples and 1,200 local test examples.

**为什么这部分值得讲：**有队友关心样本太少。K20 的每个例子有 20 个输入特征；总共有 12,000 个带标签例子。两个数量描述不同维度。模型在 8,400 条 training 上训练后，另用 2,400 条 validation 评分，当前结果支持模型学到了对该验证集有用的规律。

另有每个数据集 10,000 条 hidden 输入，标签未提供。三组共享 validation 已用于 B 的开发；local test 与 hidden 尚未用于 B 的训练或评分。报告这个状态，方便小组统一后续评估。

## 3. 符号特征：把数值简化为正负号

> For sign features, I replace each non-negative value with plus one, and each negative value with minus one. For example, a vector like 1.09, minus 1.16 and 0.91 becomes plus one, minus one and plus one. We keep the same feature order, but discard the magnitude.

解释时可手写：`(1.09, −1.16, 0.91) → (+1, −1, +1)`。零映射到 +1。这里的 +1 是输入特征值；二分类中的类别 1 是标签。

为什么尝试：K20 training 特征的绝对值中位数约为 1，因此我们提出正负号可能值得单独测试的假设。这个观察引导了实验，不能单独证明数据如何生成，也不能证明数值幅度完全无用。

## 4. 展示 raw/sign 配对表

> To compare the two representations, I used the same training folds and the same model settings within each pair. On K20, CatBoost improved from 65.33% with raw inputs to 92.67% with sign inputs. XGBoost improved from 56.12% to 75.25%, and Random Forest from 52.35% to 60.36%. Extra Trees fell from 75.58% to 70.96%. The RBF SVM and kNN results changed only slightly under the tested settings.

这里的百分比是 **training 内 3-fold cross-validation 平均 accuracy**。举 K20：8,400 条 training 划成三份，每次使用其中两份训练、另一份评分，重复三次。

对照使用同样的参数，便于观察当前设置对表示的响应；各自针对 raw 和 sign 独立调参后的最佳成绩需要另一个实验。K20 之前用于选择方法和参数，因此这张表是开发阶段证据。

**Random Forest 与 Extra Trees 是两个方法。** 原来的五方法表有 Extra Trees，后来单独补做 Random Forest。展示英文会议表时，两行都保留。

## 5. 简要解释 CatBoost

> CatBoost is a gradient-boosted decision-tree method. It builds trees sequentially, with later trees using information from the current model's errors. The final prediction combines the trees' contributions. Our current model uses numerical sign inputs, depth 10, 600 iterations and learning rate 0.05.

讨论可以停在这个层次：每棵树按特征条件划分样本，并对不同组合给出分数；多棵树逐步组合。当前运行使用 `Plain` boosting，输入是数值型 ±1。CatBoost 论文还介绍了类别特征处理与 ordered boosting；解释本次实验时应按实际配置说明。

## 6. 报告三个数据集的 CV 与 seed 结果

> With the fixed CatBoost setting, the 3-fold mean accuracy is 92.67% on K20, 94.46% on K30 and 94.52% on K40. I selected the setting through K20 development and then applied it to K30 and K40. On fixed folds, changing the model seed from 67 to 123 and 314 changes the K30 and K40 mean accuracies only slightly.

K30 三个 seed：94.46%、94.32%、94.29%；K40：94.52%、94.63%、94.61%。固定的是同一组 training 3-fold，因此它们反映这些划分下模型 seed 变化，适合讨论训练随机性。

K20 的三折分别为 93.82%、92.54%、91.64%，平均 F1 为 92.61%。合并三折的混淆矩阵 `[[3920,316],[300,3864]]` 意味着 TN=3920、FP=316、FN=300、TP=3864。若组员关心只报 accuracy 是否足够，可用它说明两个类别的错误都记录了。

## 7. 展示 learning curve，回答数据量问题

> I also trained the same CatBoost configuration using 25%, 50% and 100% of the shared training set, then evaluated it on the shared validation set. On K20, validation accuracy increased from 82.42% to 90.28%, then to 94.58%. The same pattern appears in the averages for K30 and K40. This gives us evidence that sample size affects this model's performance on these datasets.

| 数据集 | 25% training | 50% training | 100% training |
|---|---:|---:|---:|
| K20 | 82.42% | 90.28% | 94.58% |
| K30 | 88.60% | 93.96% | 95.33% |
| K40 | 88.10% | 93.23% | 95.42% |

K20 这三档分别是 2,100、4,200、8,400 条。25% 和 50% 是三个抽样 seed 的平均；它们使用相互重叠的子集和同一个 validation 集。100% 三次记录使用相同样本、顺序和模型 seed，属于相同结果的重复记录。

模型的 training accuracy 接近 100%，与 validation 有差距。样本量增大时，该实验的平均 validation accuracy 上升；更多数据的效果仍需实测。3-fold 和本表使用不同评分数据及训练数量，展示时各自标明评估方式。

### 如何回应六页文章

> In the mock paper, the degree-two logistic regression learning curve on K20 increased from 52.2% to 54.8% as the training fraction increased from 25% to 100%. Our CatBoost curve shows a larger increase under our setup. The models, splits and evaluation sets differ, so a useful next step is to run the paper-style baseline and CatBoost on the same rows.

文章第 4 页第 4.3 节报告 polynomial logistic regression 在测试的次数和数据集上都低于 59%，文中明确给出的 K20 最高值是 58.4%。第 4–5 页的 learning curve 使用 degree-2 logistic regression，K20 从 52.2% 升到 54.8%。我们现在的结果提出了一个值得写进文章的问题：换用 sign + CatBoost 后，同一数据集上观察到的数据量关系会怎样变化？

两边现有结果的划分和模型不同：文章使用 60/20/20，并在 test 上报告 learning curve；我们用 70/20/10 和共享 validation。请 A 在小组共同确定的同一组子集上跑 baseline，便于形成更受控的组内对照。文章对 universal approximation 的讨论也可作为问题：理论上的表达能力、有限数据下能否训练到好的模型，以及评估集上的泛化表现，需要分别讨论。

## 8. 希望会上达成的决定

> I have saved the nested subset indices. Could A use the same rows for the baseline learning curve? That would let us compare how the methods respond to the same training sizes. We can also decide whether C has time to run a matching MLP curve.

讨论并记录：

1. A 是否接受 B 保存的子集索引，或由小组共同指定另一套。若重新指定，已有 CatBoost 曲线需要在新索引上重跑才能直接配对。
2. 对照时统一训练行、validation 行、accuracy/F1 等指标；每个方法记录自己的预处理、参数、seed、搜索次数、时间和硬件。
3. 请 C、D 汇报各自已有的实验和障碍，并确认 C 何时帮助 D 复核 LoRA 数学。
4. 确定下一次每个人交什么给谁。最终模型、local test 的评估时点、最终 refit 和四个 hidden 输出负责人留到小组选模时明确锁定。

## 9. 可能被问到的问题

**Q：90% 已经达到了吗？**  
当前 CatBoost 已在 training 3-fold 和共享 validation 的开发评估达到 90% 以上。作业目标涉及 hidden 数据，hidden 没有公开标签；最终表现还要等后续评估。

**Q：数据量太小，结果靠谱吗？**  
K20 有 12,000 条带标签数据，K30 和 K40 有 20,000、32,000 条；模型在保留的 validation 行上取得较高 accuracy。Learning curve 显示训练样本数影响表现。小组还要固定方案后检查 local test，并完成跨机器复现。

**Q：为什么符号特征有效？**  
一种可能是符号表示减少了树需要处理的数值阈值，让这些配置更容易学习某些特征组合；这个机制仍是解释性假设。符号映射也丢失幅度信息，因此具体效果取决于模型、参数和数据。Extra Trees 的当前结果就是下降。

**Q：为什么 CatBoost 用 raw 时只有 65.33%？能否调 raw 的参数？**  
这个对照在 raw 和 sign 上使用相同配置。它回答当前配置对输入变换的响应；为 raw 单独调参是可以做的后续实验，要记录搜索预算，并在同样评估流程下比较。

**Q：CatBoost 会不会过拟合？**  
当前 training accuracy 接近 100%，validation 为 94%–95%，有可观察的差距。K20 经过多轮选择，validation 也用于开发；需要固定模型后检查 local test，并结合不同划分和模型 seed 分析。

**Q：我们能直接把 94.58% 与队友的 3-fold 结果比较吗？**  
先核对训练行数和评分行。94.58% 来自完整 K20 training 训练后评估共享 validation；K20 training 内 3-fold 的对应均值是 92.67%。同一表格应清楚标明流程，并优先使用对齐的实验做比较。

**Q：其他人能复现吗？**  
代码、划分、参数、结果和 notebook 已在本机仓库。保存的 K20 示例模型重新加载后，8,400 条 training 的预测概率逐值一致；这项检查验证保存与加载。组员拉取共享版本后仍需在自己的机器实际运行。历史 learning-curve manifest 缺少辅助文件 `xgb_cv.py` 的当时指纹，整理正式复现材料时需要明确记录。

**Q：下一步 B 做什么？**  
优先与 A 对齐 baseline learning curve，和 C 对齐比较口径；整理 CatBoost 的模型、符号表示、混淆矩阵、训练—验证差距和实验记录，为报告提供方法与分析材料。其他新实验由小组根据时间和已有结果决定。

## 10. 会后记录模板

| 事项 | 决定 | 负责人 | 交给谁 / 下一次更新时间 |
|---|---|---|---|
| 共享子集 | 待会议填写 |  |  |
| A baseline | 待会议填写 |  |  |
| B 对照与分析 | 待会议填写 |  |  |
| C MLP | 待会议填写 |  |  |
| D LoRA / 理论复核 | 待会议填写 |  |  |
| 公共结果表与复现 | 待会议填写 |  |  |
| 下次会议 | 待会议填写 |  |  |

依据：同目录英文会议材料所列实验记录、仓库数据划分 manifest 和课程说明；原四人执行计划的 mid-project review 为约 30 分钟。
