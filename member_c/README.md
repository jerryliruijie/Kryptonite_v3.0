# Member C：从这里开始

更新：2026-11-11。核心 MLP 实验与本地写作材料已完成；LoRA 复核、干净环境/伙伴复现、组内整合和最终预测仍有待办。本交付版本包含核心实验代码、五份主要 notebook、写作材料与复现记录，目标分支为 `neural_networks`。

这里既有日常使用的成果，也有实验的原始证据。**平时不用逐个翻 `results/`；先看下面五个入口。**

## 1. 汇报和写作：先看这五个文件

| 你要做什么 | 打开哪个文件 | 能看到什么 |
| --- | --- | --- |
| 讲清楚模型 | [MLP 方法与理论](report/mlp_methodology.md) | 网络结构、参数量、BCE、Adam、选模规则、理论讨论 |
| 汇报最终结果 | [结果与局限](report/p3_findings.md) | 三个数据集的验证/测试结果、固定单模型结果、结论边界 |
| 展示训练过程 | [最终模型 loss curves](notebooks/final_model_loss_curves.ipynb) | 256×256、seed 42 的训练/验证 loss、选定 epoch 与早停 |
| 展示最终表现 | [验证与测试对照图](notebooks/p3_test.ipynb) | 两种宽度、三个 seed 的验证/测试汇总 |
| 解释受控实验 | [三 seed 宽度对照](notebooks/p2_multiseed.ipynb) | 不同宽度的验证表现、seed 波动与完整对照 |

Notebook 保留代码与执行输出，运行后用 `plt.show()` 直接显示图表，不导出 PDF。图表路径里的 test 指本地 held-out 测试，不是 hidden 成绩。

## 2. 需要看代码时

| 文件 | 作用 |
| --- | --- |
| [model.py](model.py) | MLP 网络定义，了解模型从这里看 |
| [train.py](train.py) | 训练、验证、早停、checkpoint、预处理及运行追溯 |
| [evaluate.py](evaluate.py) | 核对锁定文件后，评估固定模型；不负责训练或调参 |
| [summarize_round1.py](summarize_round1.py) | 校验运行记录并汇总第一轮结果，也是多 seed 汇总的依赖 |
| [summarize_multiseed.py](summarize_multiseed.py) | 汇总全部 seed，按验证均值确定候选 |
| `__init__.py` | Python 包入口，保留即可 |
| [configs/](configs/) | 实验配置与清单；`p3_lock.json` 固定测试前的模型和文件哈希 |

`model.py` 和 `train.py` 是模型实现的主要入口；两个 summarize 文件并非多余的旧代码，当前 notebook 仍会调用它们。

## 3. results 很多，但有明确分工

| 文件夹/模式 | 用途 | 平时是否需要打开 |
| --- | --- | --- |
| [p3_test/](results/p3_test/) | 最终测试指标、逐样本预测、区间与核验；先看 `summary.json` 或 `summary.csv` | 查最终数字时打开 |
| [p2_multiseed/](results/p2_multiseed/) | 18 次正式运行的开发结果；`runs.csv` 是逐运行表，`summary.csv` 是六组汇总 | 查验证结果/参数量/耗时时打开 |
| `p2_r1_*`、`p2_ms_*`、`pilot_20_provenance_fix/` | 正式比较引用的 18 份原始运行；包含权重、scaler、日志、预测与源码快照 | 一般由代码读取，手工复核时打开 |
| [p2_round1/](results/p2_round1/) | 单 seed 的早期比较汇总 | 想了解实验过程再看 |
| `pilot_20_seed42/`、`pilot_20_seed42_repeat/`、`tiny_fit_20_seed42/` | 初始 pilot、重复运行和小样本拟合诊断 | 核对实现正确性时看 |
| `p1_checks/`、`provenance_fix_checks/`、`p3_preflight/` | 正确性、修复回归和最终评估前检查 | 留作证据，日常不用看 |
| [setup/](results/setup/) | 环境/硬件/依赖版本、数据完整性与安装记录 | 重建环境时看；安装过程日志不属于结果正文 |
| [p4_format/](results/p4_format/) | `inline_verification.json` 保存主要图表的直接绘图检查 | 排查绘图记录时看 |

**不要根据文件夹名字随意删除或搬动原始运行。** 例如 `pilot_20_provenance_fix/` 被正式实验复用；其他目录中的 `source/` 是当时的源码快照，不是应当编辑的另一份实现。锁定清单依赖固定路径和内容。文件多主要来自这些自动保存的实验记录，不代表有几百个文件需要人工维护。

## 4. 可以暂时不看的内容

| 内容 | 定位 |
| --- | --- |
| [pilot_20.ipynb](notebooks/pilot_20.ipynb)、[p2_round1.ipynb](notebooks/p2_round1.ipynb) | 早期诊断图；日常优先看上面的最终模型和三 seed 图 |
| [p3_lock_review.md](report/p3_lock_review.md) | 开测前锁定规则，审查实验流程时使用 |
| [figure_handoff.md](report/figure_handoff.md)、[references.bib](report/references.bib) | 合并论文时使用的图注、绘图说明和文献 |
| `__pycache__/` | Python 自动缓存，已被 Git 忽略，不是交付成果 |

“暂时不用看”不等于“可以删除”：历史记录或某个输入可能仍被 notebook 和锁定校验引用。当前版本以核心实验的复现依赖为交付范围。

## 5. 常用操作

请通过 Git clone 获取仓库并检出所需版本。训练追溯会读取 Git HEAD，因此只有文件、没有 `.git` 元数据的 ZIP/目录副本不足以运行训练相关检查。

在仓库根目录运行测试：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

打开任意 notebook，选择项目的 `Kryptonite C (Python 3.12)` 内核并运行全部单元。命令行重绘最终模型 loss curve：

```bash
JUPYTER_PATH="$PWD/.venv/share/jupyter" .venv/bin/python -m nbconvert --to notebook --execute --inplace member_c/notebooks/final_model_loss_curves.ipynb
```

全部实验配置、历史运行命令与环境说明保留在下面。历史段落反映当时状态；当前状态以本页顶部和根目录 [plan.md](../plan.md) 为准。

<details>
<summary>展开完整操作说明与 P0–P4 阶段记录</summary>

## 当前状态

2026-11-10：P0/P1、P2 三 seed 对照及 P3 的 C 本地锁定/测试评估完成（18 个 checkpoint）。现有测试分区已经打开并用于最终评估，不再作为调参反馈。下方 P0/P1/P2 内容为对应阶段的记录；当前结果见末尾 P3。完整执行计划见仓库根目录 `plan.md`。

环境：Python 3.12.4、macOS arm64、Apple M3 Pro（11 核）、18 GiB 内存。项目环境位于 `.venv`，不使用终端提示符中的 Conda base 环境运行实验。

## 使用环境

在仓库根目录运行；直接指定 Python 路径即可，不必先激活环境：

```bash
.venv/bin/python --version
.venv/bin/python -m pip check
.venv/bin/python example_shared.py
.venv/bin/python -m unittest discover -s tests -v
```

本次检查对以下线程环境变量设置为 1。复现相同检查条件时，先在当前终端设置；这些设置不修改系统或 shell 配置：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

P0 实测 PyTorch intra-op 线程数为 1、inter-op 为 11。P1 的训练代码进一步显式设置二者均为 1，并启用确定性算法；每次运行分别记录实际设置。

若需重建此 Mac 的环境，使用 Python 3.12 创建新的虚拟环境，并安装 `results/setup/requirements-resolved.txt`。它记录本次实际安装的直接和传递依赖；不同平台优先遵循团队根目录 `requirements.txt`，该快照不能替代跨平台验证。

## P0 验收证据

| 文件 | 内容 |
| --- | --- |
| `results/setup/environment.json` | Python、硬件、依赖匹配、线程、命令退出码与耗时、Git revision 和源文件哈希 |
| `results/setup/requirements-resolved.txt` | 完整环境版本快照，含 pip/setuptools |
| `results/setup/pip-check.log` | 依赖一致性检查通过 |
| `results/setup/example.log` | 四套数据加载和验证集上的固定预测示例 |
| `results/setup/tests.log` | 12 项公共测试全部通过 |
| `results/setup/integrity.json` | 12 项数据/索引哈希匹配 |

`example.log` 中的准确率只是常量预测示例，不能用于报告 MLP 性能。公共测试会读取测试与隐藏数组核对格式、划分和行序，但没有使用它们调参或评估 MLP。

## 与 A 对齐的交接清单（待确认）

- 已验证：使用 seed=67 的共享 70/20/10 索引、原始输入和 `common.evaluate`；保留 class 1 正类约定。
- C 建议每个 run 记录计划中的字段：数据集、划分版本/哈希、代码版本/修改哈希、配置、预处理、训练 seed、样本数、best epoch、验证指标、时间、硬件、线程和输出路径。
- C 本地记录暂放 `member_c/results/`；需要 A 确定组内汇总表位置、列名和交接方式。
- 需要确认各数据集/模型族的实际计算预算，以及三个训练 seed（拟用 42、123、2026）。P1 pilot 将先测量单次训练耗时。
- 由小组决定是否需要 C 做额外学习曲线，以及最终预测由谁导出。

上述内容尚未获 A 确认；不影响先实施 P1 的正确性检查，正式搜索前须完成对齐。

## P1 运行方式

在仓库根目录使用模块命令。每个 run ID 对应独立目录；代码拒绝覆盖已有运行。以下名称为重跑示例，可按需换为新的唯一名称：

```bash
.venv/bin/python -m member_c.train --config member_c/configs/tiny_fit_20.json --run-id tiny_fit_20_reproduction
.venv/bin/python -m member_c.train --config member_c/configs/pilot_20.json --run-id pilot_20_reproduction
```

配置可指定 20/30/40 输入；当前只完成 20 的 pilot。`tiny_fit` 从共享训练集固定抽样，并故意在相同子集上评分，其 `validation` 字段属于拟合诊断，结果带有 `diagnostic_resubstitution=true`，不可加入泛化结果汇总表。pilot 正常使用独立验证集。

训练阶段不接受测试集或 hidden 评估选项。共享 loader 会校验完整数据划分，但训练入口立即丢弃返回的测试数组。只有训练行参与 scaler 拟合。

最佳 checkpoint 按验证准确率选择，并列时选较低验证损失，再并列保留较早 epoch；早停计数只在准确率严格改善时重置。初始规则为至少 30 epochs、连续 20 epochs 无准确率改善则停止、最多 150 epochs。

每个运行目录包含：

- `best.pt`：最佳模型权重、结构配置、选中 epoch，以及 scaler 的 mean/scale/variance 和拟合样本数；仅支持推理恢复，未实现中断训练续跑。
- `history.csv`：逐 epoch 完整训练/验证 loss、accuracy、训练过程中按样本加权的 optimization loss 以及累计耗时。
- `summary.json`、`validation_predictions.npz`：最佳 checkpoint 的指标、验证 logits、标签和二进制预测。
- `run.json`、`config.json`、`source/`：精确命令、运行版本、源文件哈希、线程、依赖与实际代码快照。源码存在未提交修改时，不能仅凭 Git HEAD 复现，应使用快照或确认文件哈希。
- tiny-fit 另保存抽样位置；该位置对应共享训练数组的行号，不是原始完整数据的行号。

### 运行追溯（P1 审查后修复）

新的运行在数据加载和训练前写入 `config.json`、`run.json` 与源码快照。源码哈希和快照来自同一次读取，不会在训练结束后重新抓取工作区源码。`run.json` 的状态从 `running` 更新为 `completed`、`failed` 或 `interrupted`；失败时 `failure.json` 保留同样的元数据、异常栈和耗时。重复 run ID 在创建目录时就被拒绝，不修改旧记录。

- `elapsed_seconds`：从保留运行目录到执行结束的总耗时，包含配置、追溯、数据加载与训练，排除解释器启动和模块导入，以及最终状态文件写入。
- `training_attempt_seconds`：进入训练核心至结束/异常的耗时，包含预处理、初始化和训练后校验；数据加载阶段就失败时为 null。
- 成功运行 `summary.json` 的 `training_seconds` 仍只计算原先的 epoch 循环；不可与上述耗时重复相加。
- 强制杀进程或断电无法执行收尾，此时可能保留 `running` 状态、开跑前快照和已有 epoch 日志，需人工核对，不应当作成功运行。
- CLI 记录实际 Python 可执行文件和命令；直接调用数组接口 `fit` 时记录该入口，`command=null`，调用方还需保存构造输入数组的代码。

原有 P1 pilot 的结果与源码快照保留原样，属于修复前的记录格式；它们的源码哈希不再与修复后的训练代码相同。复现旧实验应使用原始快照或对照修复后的回归结果，不能重新写入历史运行以伪装成采用了新的记录机制。

代码通过 `load_checkpoint` / `predict_checkpoint` 恢复模型与预处理。例如只重新检查验证预测：

```python
from common import load_split
from member_c.train import Config, configure_runtime, predict_checkpoint

configure_runtime(Config())
X_val = load_split('20')[2]
logits = predict_checkpoint('member_c/results/pilot_20_seed42/best.pt', X_val)
predictions = (logits > 0).astype('int64')
```

## P1 结果与核验

| 项目 | 结果 |
| --- | --- |
| 公共 + 新增测试 | 20/20 通过，见 `results/p1_checks/tests.log` |
| 128 条训练样本拟合 | 100%；诊断用途，不代表泛化 |
| Pilot 训练 / 验证样本数 | 8,400 / 2,400 |
| Pilot 网络参数量 | 19,329 |
| 最佳 epoch / 停止 epoch | 73 / 93 |
| 最佳 checkpoint 训练 accuracy | 99.20% |
| 最佳 checkpoint 验证 accuracy | 89.58% |
| 独立重复 | 逐 epoch 指标、最佳权重、验证预测一致 |
| 新进程 checkpoint 预测核验 | 最大绝对误差 0；同一 CPU 和软件环境 |

独立重跑是同一 seed 的实现复现检查，不能当成两个独立训练 seed 计算不确定性。详见 `results/p1_checks/reproducibility.json`。

曲线显示验证损失后期回升，与过拟合一致；P2 才研究配置变化，当前不声称已找到最优网络或达到 hidden 目标。

## 重新生成曲线

`notebooks/pilot_20.ipynb` 已有执行输出，只读取训练日志，不读取测试或 hidden 数据。图表通过 `plt.show()` 直接显示在 notebook 中。在项目虚拟环境内注册内核后执行：

```bash
.venv/bin/python -m ipykernel install --prefix .venv --name kryptonite-c --display-name 'Kryptonite C (Python 3.12)'
JUPYTER_PATH="$PWD/.venv/share/jupyter" .venv/bin/python -m nbconvert --to notebook --execute --inplace member_c/notebooks/pilot_20.ipynb
```

Notebook 会检查内核是否确实来自项目 `.venv`，避免意外使用 Conda base。若重跑训练使用了新的 run ID，先修改 notebook 的 `run` 路径。保留原始结果；不要把不同运行的图与指标混用。

## P2 首轮：两个宽度、三个数据集、seed 42

首轮的 6 份配置、数据/源码哈希、假设、预算和复用依据在 `configs/p2_round1/manifest.json`，已在 5 次新训练前固定。20 小网络复用 `pilot_20_provenance_fix`，其余运行以 `p2_r1_` 开头；没有把 P1 的多个同 seed 复现当作独立样本。

| 数据集 | 小网络验证 accuracy | 大网络验证 accuracy | 差值（大减小，百分点） |
| --- | ---: | ---: | ---: |
| 20 | 89.58% | 91.38% | +1.79 |
| 30 | 90.83% | 91.28% | +0.45 |
| 40 | 93.56% | 93.19% | -0.38 |

小/大网络分别为 [128,128] / [256,256]。相同训练协议包括标准化、ReLU、Adam、学习率 1e-3、weight decay 1e-4、batch 256、150 epoch 上限、min epoch 30、patience 20、CPU 与 PyTorch intra/inter-op 各 1。早停导致实际 epochs、更新次数和时间不同；结果只支持当前协议下的单 seed 比较，不是等计算量实验。

可以重新生成已核验的汇总表：

```bash
.venv/bin/python -m member_c.summarize_round1
```

汇总脚本核对锁定配置、完成状态、划分、源码快照哈希、保存预测与指标、最佳 epoch 与曲线。额外的独立 checkpoint 重载/验证行序检查记录在 `results/p2_round1/verification.json`。训练命令和进程总耗时在 `execution.json`，epoch 循环耗时单独保留；两者不要相加。

若需复现某次训练，使用对应配置与新 run ID，例如：

```bash
.venv/bin/python -m member_c.train --config member_c/configs/p2_round1/30_w256_seed42.json --run-id p2_r1_30_w256_reproduction
```

结果文件：`results/p2_round1/comparison.csv`、`comparison.json`。带执行输出的图表 notebook 为 `notebooks/p2_round1.ipynb`，图表直接显示在 notebook 中，重执行方式与 P1 相同。

下一步优先补齐 seed 123、2026，对两种宽度在三个数据集上的表现进行重复检验。当前没有证据证明小幅差异稳定，也不能把验证超过 90% 称为 hidden 达标。尚未启动额外学习率搜索、正式测试或预测导出；A/B 的团队级预算/交接约定仍待确认。

## P2 多 seed 扩展（已完成）

以上首轮之后，按预先约定补齐 123、2026：新增 12 次成功运行，复用全部 6 个 seed=42 结果，总计 18 次。每个 seed 都保留，不删除差的运行；三个 seed 始终使用相同数据划分。锁定文件在 `configs/p2_multiseed/manifest.json`。

| 数据集 | [128,128] 验证 accuracy | [256,256] 验证 accuracy |
| --- | ---: | ---: |
| 20 | 90.36 ± 0.72% | 90.57 ± 1.19% |
| 30 | 89.63 ± 2.44% | 91.03 ± 1.02% |
| 40 | 93.08 ± 0.52% | 93.49 ± 0.48% |

数值是三个 seed 的均值 ± 样本标准差（ddof=1），不是置信区间。20/40 的宽度胜负随 seed 改变；30 的大网络在三次中均较高，但只有三个 seed，不能过度推断。较差的 30 小网络 seed=2026（86.825%）仍计入汇总。

按既定均值选择规则，三个数据集的开发候选均为 [256,256]。这个选择不构成最终测试或 hidden 模型锁定，也不证明其普遍优于小网络。下一步仍需固定待报告模型、预测 seed/集成规则与全量重训策略。

```bash
.venv/bin/python -m member_c.summarize_multiseed
```

- `results/p2_multiseed/runs.csv`：18 次独立运行的指标、时间、checkpoint 路径标识与哈希。
- `summary.csv` / `summary.json`：每种配置的三个单独成绩、均值、样本标准差与开发候选。
- `paired_differences.csv`：同 seed 下大网络减小网络的验证 accuracy 差异（百分点），只作描述性比较。
- `execution.json`：本批 12 次新训练的精确命令、退出码与进程耗时。
- `verification.json`：独立重载 18 个 checkpoint 和验证行序检查；原 6 次运行的 77 个历史文件保持不变。
- `tests.log`：27/27 项测试通过。
- `notebooks/p2_multiseed.ipynb`：已执行的结果表与内嵌图，运行后直接显示。

复现任意新增训练请使用新的 run ID，例如：

```bash
.venv/bin/python -m member_c.train --config member_c/configs/p2_multiseed/30_w256_seed123.json --run-id p2_ms_30_w256_seed123_reproduction
```

核心训练代码在此批次未改变。未启动额外超参数搜索、测试集评分、全量重训或 hidden 导出。A/B 协作交接仍待完成。

## P3 最终本地评估（2026-11-10 完成）

用户授权后先锁定、再开测。可审查清单为 `report/p3_lock_review.md`，机器可验证锁为 `configs/p3_lock.json`；主配置统一 [256,256]，单模型固定 seed 42，logit > 0，无 ensemble。所有 18 个模型均完整报告，未用测试排名修改选择。

| 数据集 | 128 宽度测试均值 ± SD | 256 宽度测试均值 ± SD | 预定 seed42 大网络测试 accuracy |
| --- | ---: | ---: | ---: |
| 20 | 88.92 ± 0.14% | 89.42 ± 0.80% | 89.75% |
| 30 | 90.75 ± 1.78% | 91.87 ± 0.63% | 92.15% |
| 40 | 92.69 ± 0.95% | 93.64 ± 0.69% | 93.25% |

这里是 **local test** 成绩，区别于上文 P2 验证成绩。SD 描述固定划分下三个训练 seed 的波动。单模型的 Wilson 95% 区间见 `results/p3_test/models.csv` / `summary.json`，不对同一批测试样本的三次预测做池化。20 的点估计低于 90%；结果不等于 hidden accuracy。

输出包含逐样本原始行索引、标签、logits、预测及完整指标，位于 `results/p3_test/`。核验记录 `verification.json` 确认 18 份预测与指标一致、284 个受保护文件保持不变。32 项测试在打开测试成绩前通过。已执行的 `notebooks/p3_test.ipynb` 生成验证/测试对照图。

仅为复现相同固定评估，可使用新的输出目录；这不会产生新的独立测试样本或重新授权模型选择：

```bash
.venv/bin/python -m member_c.evaluate --lock member_c/configs/p3_lock.json --lock-sha256 096ae5d7e7ca0493804aff0cc5b1f8b7c89314a6276f1f86cd66bdd4e62fcfba --output member_c/results/p3_test_reproduction
```

若代码、数据或 checkpoint 与锁定哈希不符，评估会在读取测试数据之前拒绝执行。不要修改旧锁来适应新模型；旧实验依赖其真实代码快照和固定输入。

如果日后小组把某数据集的 hidden 导出分配给 C，全量重训规则已在开测前固定：宽度 256、seed 42，20/30/40 分别训练 48/49/18 epochs，不用测试集选时长。此次尚未重训或导出；旧 held-out 成绩不能标作全量重训模型的实测分数。

`report/p3_findings.md` 提供英文方法、完整结果与局限草稿，可交给 B 整合。A/B 模型比较、LoRA 独立复核、伙伴复现与最终预测归属仍待完成。

## P4 图表格式与数学材料（2026-11-10）

四份 notebook 保留绘图代码，通过 `plt.show()` 直接显示并保存内嵌输出，不导出 PDF 或图片文件。使用 Matplotlib 自带字体和 mathtext，不再需要外部 LaTeX。按用户要求已删除此前生成的四份图表 PDF。Python 固定依赖和实验锁定文件未改变。

- `report/figure_handoff.md`：四图的直接显示方式、复现命令、英文图注及选择建议。
- `report/mlp_methodology.md`：标准化、网络维度与参数量、BCE、Adam 耦合权重衰减、选模规则，以及表示/优化/泛化的理论讨论。
- `report/references.bib`：已核对的 Adam 和通用逼近原始文献。
- `results/p4_format/inline_verification.json`：主要 notebook 的直接绘图核验记录。

这些是本地交接材料；尚未向 B 发送、整合为最终论文或完成干净环境/伙伴复现。

</details>
