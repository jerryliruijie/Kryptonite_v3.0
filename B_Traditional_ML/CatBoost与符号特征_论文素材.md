# CatBoost 与符号特征：项目理解与论文素材

## 实验问题与实际顺序

Kryptonite 的数据生成规则未公开。项目首先在固定数据划分上尝试原始输入的 SVM、Random Forest 和 XGBoost。随后观察到 Kryptonite-20 training 的特征绝对值中位数约为 0.9997，接近 1；这一观察引出了以 0 为阈值的符号表示假设。项目在后续模型筛选中比较原始值及符号输入，并在符号输入上的 XGBoost 观察到较高的 3-fold cross-validation accuracy。

项目随后引入 CatBoost，对其符号输入版本调参，并将选出的配置固定后应用到 Kryptonite-30 和 Kryptonite-40。五方法 raw/sign 配对对照在这些探索之后进行，包含 Extra Trees。用户随后指出 Random Forest 尚未纳入该配对，项目因此补充 Random Forest 的对照。符号变换是已有的二值化方法；本项目的证据来自特定数据、配置和划分上的实验。

## 符号特征的定义

给定输入向量 \(x\in\mathbb R^d\)，对第 \(j\) 个特征使用确定性映射：

\[
s_j(x)=\begin{cases}+1,&x_j\geq 0,\\-1,&x_j<0.\end{cases}
\]

例如 \((1.09,-1.16,0.91)\) 变成 \((+1,-1,+1)\)。特征仍按原顺序对应原始列；Kryptonite-20 转换前后均有 20 列。该映射不使用标签，也不从样本拟合阈值。每列的数值幅度经转换后丢失；0 映射到 +1。特征值中的 +1 和二分类标签 1 分别表示不同概念。

一个 20 维符号向量最多有 \(2^{20}\) 种组合。假设一个任务的标签与两列符号是否相同有关，则这两列的正负号可表达该关系；这个例子用于解释表示能力，Kryptonite 的标签规则仍待验证。

## 为什么转换可能改变表现

确定性变换并不创造额外原始信息。对任何只使用 \(s(X)\) 的分类器 \(g\)，可以构造使用原始 \(X\) 的分类器 \(h(X)=g(s(X))\)。因此，在同一数据分布且允许所有可测分类器的理想条件下，原始输入的最优分类 accuracy 至少与符号输入的最优 accuracy 一样高。

有限数据与具体训练算法中的表现可能不同。符号变换将每个特征压成两个取值，改变了树模型可考虑的阈值及归纳偏好，也丢弃了可能有用的幅度信息。此前比较中，相同的 K20 XGBoost 配置使用 raw 和 sign 输入分别得到 56.12% 和 75.25% 的 3-fold 平均 accuracy；CatBoost depth 8、600 棵树的对应值为 60.95% 和 79.27%。这些结果支持继续研究该表示；数值幅度是否含有信息、模型为何受益及数据生成规则仍是待检验的问题。

本次回顾性配对研究在相同数据划分和模型设置下，分别训练 raw 与 sign 输入模型。表中每格为 raw → sign 的 3-fold 平均 accuracy：

| 方法 | Kryptonite-20 | Kryptonite-30 | Kryptonite-40 |
|---|---:|---:|---:|
| RBF SVM | 58.11% → 58.69% | 54.89% → 54.92% | 51.94% → 51.81% |
| Random Forest（后续补充） | 52.35% → 60.36% | 52.33% → 57.27% | 50.87% → 54.10% |
| XGBoost | 56.12% → 75.25% | 58.34% → 72.44% | 60.36% → 73.77% |
| Extra Trees | 75.58% → 70.96% | 62.42% → 51.55% | 67.36% → 60.26% |
| kNN | 56.69% → 56.69% | 53.46% → 53.84% | 51.52% → 51.70% |
| CatBoost | 65.33% → 92.67% | 74.45% → 94.46% | 72.51% → 94.52% |

符号输入在 Random Forest、XGBoost 与 CatBoost 的当前设置中提高了 accuracy，在 Extra Trees 的当前设置中降低了 accuracy。SVM 和 kNN 的平均 accuracy 变化较小；K20 两个 kNN 模型的平均 accuracy 均为 56.69%，但 8,400 个验证预测中有 1,892 个不同。kNN 配对中两种表示均使用训练数据拟合的 StandardScaler 和 Euclidean 距离，先前 sign-Hamming kNN 的 57.15% 属于不同设置。模型超参数来自先前探索，同参数比较刻画本次配置对表示的响应，raw 各自重新调参后的最优效果有待研究。[五方法配对实验](results/representation_ablation/实验记录.md)；[Random Forest 补充对照](results/rf_representation/实验记录.md)。

## CatBoost 的预测与训练

CatBoost 是 gradient boosted decision trees 的一种实现。模型依次构建树，并将树输出累加为分类分数。对二分类任务，令 \(F(x)\) 为模型分数、\(p(x)=1/(1+e^{-F(x)})\) 为预测类别 1 的概率，可以用 log loss 衡量预测：

\[
\ell(y,F)=-y\log p-(1-y)\log(1-p),\qquad
\frac{\partial\ell}{\partial F}=p-y.
\]

例如当前分数 \(F=0\)，则 \(p=0.5\)。若真实标签 \(y=1\)，则负梯度为 \(y-p=+0.5\)，对应提高该样本分数的方向；若 \(y=0\)，负梯度为 \(-0.5\)。训练根据当前损失的信息寻找树结构并估计叶节点输出，再将新树加入模型。这个例子说明方向，实际实现可用梯度及 Hessian 等信息更新叶节点；本次重建模型记录的叶值估计方式为 Newton。

将已保存树对模型分数的贡献记为 \(\Delta_t(x)\)，可写作

\[
F(x)=b+\sum_{t=1}^{T}\Delta_t(x),\qquad p(x)=\sigma(F(x)).
\]

此处 \(\Delta_t\) 表示计入学习率后的树输出。实际二分类预测通过模型类别与分数接口生成。CatBoost 的算法细节与不同 boosting 模式见 [Prokhorenkova 等人，2018](https://proceedings.neurips.cc/paper_files/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html) 和 [CatBoost 官方训练说明](https://catboost.ai/docs/en/concepts/algorithm-main-stages)。

## Symmetric Tree 如何表示交互

本次配置使用 `grow_policy=SymmetricTree`：同一棵树的每一层对该层所有节点使用相同的分裂条件，不同层可以使用不同条件。假设第 1 层检查第 5 个特征的符号，第 2 层检查第 10 个特征的符号，四个叶节点可分别对应 \((+,+)\)、\((+,-)\)、\((-,+)\)、\((-,-)\)，并给这些组合不同的分数。这棵示例树可以表达两列输入的交互，具体叶值与真实数据规则需要由训练结果确定。[CatBoost 官方文档](https://catboost.ai/docs/en/concepts/parameter-tuning)

项目所选配置为 600 棵树、depth 10、learning rate 0.05、`l2_leaf_reg=10`、`Logloss`、CPU 单线程、训练 seed 67。每棵对称树最多有 \(2^{10}\) 个叶位置；叶位置总数与输入特征数是不同的统计量。K20 重建模型解析的 `boosting_type` 是 `Plain`，`bootstrap_type` 为 `MVS`，记录的 `subsample` 约为 0.8，叶值使用 Newton 估计，`leaf_estimation_iterations=10`。输入为数值型 ±1，代码未指定 `cat_features`。CatBoost 原论文讨论了 ordered boosting 和类别特征处理；描述本项目运行时应使用保存的实际设置。[重建模型参数](results/catboost_tune/reconstructed_model_parameters.json)

## 结果与证据边界

Kryptonite-20 上的 CatBoost 模型按相同训练数据中的 3-fold 结果选择，最优 sign 配置平均 accuracy 92.67%、F1 92.61%、ROC-AUC 0.9572，训练 accuracy 99.98%。固定该配置后，Kryptonite-30 和 40 的 training 内部 3-fold accuracy 分别为 94.46% 和 94.52%。K20 曾参与特征观察、方法选择和调参，这些数字用于开发阶段比较。后续 learning curve 已将 K20、K30、K40 的共享 validation 用于开发评分；local test 和 hidden 尚未评估。

固定 CatBoost 配置后在 K20 的 3-fold validation 数据上逐列置换，特征 5、10 分别导致平均 accuracy 下降约 42.71、43.00 个百分点。该结果描述此模型对这些列的扰动敏感性；列之间的相关性、交互以及置换可能产生的输入组合会影响解释。这个分析支持进一步做受控消融与数据独立验证。

模型、预测、3-fold 索引、依赖、种子和分析结果见 [五方法综合分析](results/model_analysis/五方法综合分析.md)、[CatBoost 实验记录](results/catboost_tune/实验记录.md) 与 [特征分析](results/model_analysis/实验记录.md)。

## 英文写作示例

After initial experiments with raw features, inspection of Kryptonite-20 training inputs motivated a binary sign representation. We mapped each feature independently to +1 for non-negative values and −1 otherwise. CatBoost with sign inputs achieved a mean accuracy of 92.67% under 3-fold cross-validation on the shared training set. In a subsequent matched-configuration comparison on the same splits, the corresponding raw-input CatBoost achieved 65.33%. The representation change increased accuracy for the tested XGBoost and Random Forest configurations and reduced it for the tested Extra Trees configuration. These comparisons are exploratory because the data informed earlier representation and model selection; the effect of independently tuning each representation remains to be assessed.

## 参考文献

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., and Gulin, A. (2018). [CatBoost: Unbiased Boosting with Categorical Features](https://proceedings.neurips.cc/paper_files/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html). Advances in Neural Information Processing Systems 31.
