# C 的补充数据分析与 MLP 诊断

日期：2026-11-10。**事后探索性分析**：本地测试评估已经完成，本分析不能充当原实验的事前理由，也不改变已锁定的模型。代码和四幅直接显示的图位于 `../notebooks/dataset_analysis.ipynb`，机器可读结果位于 `../results/dataset_analysis/summary.json`。

## 1. 训练数据没有明显的类别或数值质量问题

| 数据集 | 训练样本 | 类别 0 / 1 | 多数类比例 | 非有限数值 | 常数列 | 连续特征精确重复的多余行 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 20 | 8,400 | 4,236 / 4,164 | 50.43% | 0 | 0 | 0 |
| 30 | 14,000 | 7,008 / 6,992 | 50.06% | 0 | 0 | 0 |
| 40 | 22,400 | 11,243 / 11,157 | 50.19% | 0 | 0 | 0 |

三个训练集均没有发现相同连续特征向量对应冲突标签。类别接近平衡，当前结果不能用“总是预测多数类”解释；没有看到需要因明显不平衡而加入 class weighting 的证据。但这里的多数类比例只是训练集描述，不是另外评估的基线成绩。检查仅限训练分区内部，不能作为跨分区重复/泄漏检查已经完成的声明。

## 2. 特征尺度接近，双峰结构值得记录

三个数据集全部特征的标准差约为 1.003–1.007，每个数据集内最大/最小特征标准差比均小于 1.004。训练时保留 StandardScaler 是一致的预处理选择，但没有依据说 MLP 的优势主要来自纠正严重的尺度差异。

预先选定的 feature 1 在约 -1 与 +1 附近呈双峰，两类的边缘分布高度重叠。看图后追加的、明确标记为自适应的诊断覆盖全部特征：各列到最近 ±1 中心的 RMS 距离约为 0.099–0.102。这提示可以描述为“接近正负两种状态的连续输入”，但并未证明噪声分布、数据生成算法或标签规则。

将训练输入仅按正负号分组时：

- 20 有 35 个重复符号模式（每组两行），其中 **2 组标签冲突**。
- 30 和 40 没有重复的完整符号模式，因此不能用“未发现冲突”推断符号足以决定标签。
- 连续值不同的样本被符号变换合并，并不等于原数据存在精确重复。冲突可能涉及连续值中的信息或标签噪声，本分析不能区分。

因此，取符号、删除偏离 ±1 的部分或搜索逻辑规则只能成为另一次明确登记的探索假设，不能直接当作无损去噪，更不能自动替换当前预处理。

## 3. 没有发现强的线性相关或少数主成分集中的结构

| 数据集 | 最大绝对非对角 Pearson 相关 | 前两个 PC 方差占比 | 90% 方差所需 PC |
| --- | ---: | ---: | ---: |
| 20 | 0.0296 | 10.78% | 18 / 20 |
| 30 | 0.0283 | 7.20% | 27 / 30 |
| 40 | 0.0236 | 5.39% | 36 / 40 |

所有 PCA 均只在标准化训练数据上拟合。结果说明观测到的线性方差比较分散，没有支持“大幅线性降维还能保留绝大部分方差”的证据。它不排除低维非线性结构，也不说明分类信息一定与大方差方向一致。

所有特征中最大的类别均值差（以整体训练标准差为单位）分别为 0.0583、0.0360、0.0330；未见很大的单特征均值位移。小均值差、低相关、二维投影重叠，都不能证明线性分类器一定失败，更不能证明必须使用某个阶数的交互、parity/XOR 或 RBF 核。有关模型优劣仍需相同协议的预测实验。

## 4. 现有 MLP 已能较好拟合训练数据，但提升宽度的收益有限且有 seed 波动

| 数据集 | 宽度 | 平均训练 accuracy | 平均验证 accuracy | 平均差距（百分点） |
| --- | --- | ---: | ---: | ---: |
| 20 | 128 × 128 | 99.10% | 90.36% | 8.73 |
| 20 | 256 × 256 | 99.81% | 90.57% | 9.24 |
| 30 | 128 × 128 | 98.13% | 89.63% | 8.50 |
| 30 | 256 × 256 | 99.98% | 91.03% | 8.95 |
| 40 | 128 × 128 | 96.83% | 93.08% | 3.74 |
| 40 | 256 × 256 | 97.49% | 93.49% | 4.00 |

以上使用全部 18 次原有运行，各数字取各运行的验证选定 checkpoint，再对三个 seed 求均值。更宽网络在三个数据集上都提高了平均训练拟合，验证均值也提高，但幅度分别只有约 0.21、1.39、0.41 个百分点。20 和 40 的宽度胜负随 seed 改变；30 的三个 seed 均有正向差异，但重复数量仍小。

18 次运行全部由早停结束；17 次运行在停止 epoch 的验证 BCE 高于选定 checkpoint 的 BCE。这与训练后期的泛化表现可能恶化相容，但选模依据是验证 accuracy 而不是最小 BCE，不能据此证明早停参数最优或确定过拟合的唯一原因。当前证据不足以把瓶颈简单归为“网络不够大”或“训练轮数不够”。

## 5. 对下一步的影响

- 数据分析建议值得采纳，已经揭示单看实验表格无法看到的输入结构；但并未发现一个可以直接保证提升准确率的修复。
- 当前没有明显依据开展类别重加权、大幅 PCA 降维或无损符号去噪。原模型、scaler 和最终预测政策保持不变。
- 继续采用原先登记的宽度假设和完整 seed 报告。新的数据结构问题明确标为事后探索，不重写原实验历史。
- 非线性模型学习曲线有科学价值，但当前尚不能确认 C 模型是全组最强模型。本轮不将其改为全组强制工作，也不启动新模型/学习曲线搜索。
- 给 B 的正文只需选取直接支持论点的内容。建议保留一个简短数据概况段落，把全部 EDA 图和细节留在 notebook；不是把四张新图全部加入 3–6 页正文。

## Optional English paragraph for report integration

A supplementary analysis performed after the locked evaluation examined only the training partitions. Class proportions were approximately balanced, with majority-class fractions of 50.43%, 50.06% and 50.19% for Kryptonite-20, -30 and -40. We found no nonfinite values, constant features or exact duplicate feature vectors within these partitions. Marginal feature scales were similar, and observations concentrated near -1 and +1. A follow-up sign-pattern audit found conflicting labels in two repeated training patterns for Kryptonite-20, so sign quantization cannot be assumed to preserve all predictive information. Pairwise correlations were small (maximum absolute values below 0.03). PCA required 18/20, 27/30 and 36/40 components to retain 90% of standardized training variance, suggesting little variance concentration in a small linear subspace. These descriptive observations do not identify the label-generating mechanism or establish linear inseparability. They were not used to revise the locked models or prediction policy.
