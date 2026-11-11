# Kryptonite-3.0: group progress meeting

**Wednesday, 11 November 2026 — proposed working agenda**  
**Participants:** A, B, C and D  
**Purpose:** exchange progress, compare experimental conditions and agree the next handoffs. The meeting time and each member's current progress should be confirmed by the group.

## 1. Proposed agenda: around 30 minutes

| Time | Discussion | Desired outcome |
|---|---|---|
| 3 min | Confirm the objective, shared data and meeting priorities | Agree what needs a group decision today |
| 5 min | A: setup, baseline and learning curve progress | Record available baseline results and any support needed |
| 7 min | B: traditional models, sign features and learning curve | Understand the current candidate and its evidence |
| 5 min | C: neural-network progress | Record current runs, difficulties and next comparison |
| 5 min | D: KryptoNet adaptation and LoRA | Record controls, implementation and theory progress |
| 5 min | Agree shared comparisons and handoffs | Record decisions, owners and a target time for the next update |

This is a proposal based on the existing mid-project review plan. A, C and D should report their actual status during the meeting. The group can adjust the agenda and responsibilities.

## 2. Shared data and evaluation

Each row is one labelled example with the specified number of features. The team split is stratified with split seed 67.

| Dataset | Features | Labelled examples | Training | Validation | Local test |
|---|---:|---:|---:|---:|---:|
| K20 | 20 | 12,000 | 8,400 | 2,400 | 1,200 |
| K30 | 30 | 20,000 | 14,000 | 4,000 | 2,000 |
| K40 | 40 | 32,000 | 22,400 | 6,400 | 3,200 |
| K40-Adapt | 40 | 4,000 | 2,800 | 800 | 400 |

Each corresponding hidden input contains 10,000 rows and no supplied labels. These rows are separate from the labelled counts above. `splits/manifest.json` records the exact counts, split seed and source fingerprints.

**Data-use status for B:** K20, K30 and K40 shared validation sets have been used during development. The local test sets and hidden inputs remain reserved and have not been used to train or score B's reported models. Other members should confirm their own data use.

Use class 1 as the positive class. The shared interface supports accuracy, precision, recall, F1, specificity, confusion matrix and ROC-AUC when scores are available. Learned preprocessing should be fitted using training data only. Record the training seed separately from split seed 67.

## 3. B's progress: representation and model comparisons

### The question

How do different classification methods respond to the original numeric inputs and to a simple sign representation?

For each feature, `x >= 0` maps to `+1`, and `x < 0` maps to `-1`. The number and order of features stay the same. The transformation preserves the sign and discards the original magnitude. It is deterministic and uses no labels.

### K20 matched comparison

Each pair uses the same K20 training rows, saved 3-fold cross-validation splits, model seed and model settings. Values are mean accuracy across the three folds. Model settings reflect earlier exploration; tuning each representation separately could change the comparison.

| Method | Raw input | Sign input | Sign − raw |
|---|---:|---:|---:|
| RBF SVM | 58.11% | 58.69% | +0.58 pp |
| Random Forest | 52.35% | 60.36% | +8.01 pp |
| XGBoost | 56.12% | 75.25% | +19.13 pp |
| Extra Trees | 75.58% | 70.96% | −4.62 pp |
| kNN | 56.69% | 56.69% | +0.00 pp |
| CatBoost | 65.33% | 92.67% | +27.33 pp |

Random Forest and Extra Trees are distinct algorithms; the Random Forest pair was added after the initial five-method study. Sign inputs helped the tested Random Forest, XGBoost and CatBoost settings, and reduced accuracy for the tested Extra Trees setting. The effect depends on the model and configuration. K20 informed previous representation and model selection, so these are exploratory development results.

### Fixed CatBoost candidate on all three datasets

The candidate uses sign inputs, depth 10, 600 iterations, learning rate 0.05, L2 10 and model seed 67. It was selected through K20 development and then applied to K30 and K40 with the same configuration.

| Dataset | Training 3-fold mean accuracy | Training 3-fold mean F1 | Mean ROC-AUC |
|---|---:|---:|---:|
| K20 | 92.67% | 92.61% | 0.9572 |
| K30 | 94.46% | 94.48% | 0.9614 |
| K40 | 94.52% | 94.50% | 0.9591 |

K20 fold accuracies are 93.82%, 92.54% and 91.64%. The pooled K20 out-of-fold confusion matrix is `[[3920, 316], [300, 3864]]`, ordered `[[TN, FP], [FN, TP]]`.

K30 seed 67/123/314 gave 94.46% / 94.32% / 94.29%; K40 gave 94.52% / 94.63% / 94.61%. These model-seed checks reuse the same training 3-fold splits and therefore describe variation across model seeds for those splits.

CatBoost is a gradient-boosted decision-tree method: it builds trees sequentially and combines their outputs. The project run used numerical sign inputs and `Plain` boosting. Its high score supports further investigation of this combination; the observations do not establish the hidden data-generation rule or the cause of the improvement.

## 4. B's learning curve

The percentages below are fractions of the **shared training split**. The models are evaluated against the fixed shared validation split. The 25% and 50% values average three nested-subset seeds, 67, 123 and 314. The model seed stays 67.

| Dataset | 25% training | 50% training | 100% training |
|---|---:|---:|---:|
| K20 | 82.42% | 90.28% | 94.58% |
| K30 | 88.60% | 93.96% | 95.33% |
| K40 | 88.10% | 93.23% | 95.42% |

For K20 these training sizes correspond to 2,100, 4,200 and 8,400 rows. At 100%, all three saved records use identical training rows, order and model seed, and consequently are repeated copies of the same result. The smaller subsets overlap and share one validation set. Their variation is descriptive of these subsets.

Training accuracy was close to 100%; the validation gap warrants discussion. Increasing training size improved mean validation accuracy across the measured sizes. The result is consistent with sample size affecting this candidate, while the next performance level requires direct evaluation.

**Key group decision:** B has saved proposed nested indices in `B_Traditional_ML/results/learning_curve/subsets/`. Can A use these exact indices and validation rows for a baseline learning curve? C can use the same rows if the group chooses an MLP comparison. The group should confirm or replace the proposed indices before presenting a common comparison.

The 3-fold results in Section 3 and the validation results in Section 4 use different evaluation designs and training sizes. Keep their table labels separate.

### Connection to the six-page mock paper

The mock paper's polynomial logistic regression experiments reported accuracy below 59% across the tested degrees and datasets; its stated best K20 result was 58.4% (p. 4, Section 4.3). Its degree-2 logistic-regression learning curve on K20 increased from 52.2% to 54.8% as the training fraction grew from 25% to 100% (pp. 4–5, Section 4.4 and Figure 2).

Our CatBoost results suggest a useful follow-up: model family and input representation can change the observed performance, and this candidate responds to training size. The paper used a 60/20/20 split and evaluated its learning curve on its test set; B used the team's 70/20/10 split and shared validation. The two sets of numbers describe different experimental protocols. A's planned baseline on agreed common rows would give the group a more controlled comparison of the learning curves.

The paper's Adapt controls used 3,200 training and 800 test rows and reported 48.1% for zero-shot KryptoNet, 49.5% for a linear probe and 46.4% for an MLP probe (pp. 4–5, Section 4.5). D's results and protocol remain to be reported at the meeting. When discussing the paper's broad conclusion about universal approximation, distinguish theoretical representational capacity from the finite-data training and generalisation outcomes measured in its experiments.

## 5. Questions for the other tracks

**A — baseline and integration**

- Which baseline runs and metrics are available, and do they use the shared splits?
- Can we agree the nested learning-curve indices and common result-table fields?
- Is any help needed with the setup, combined table or experiment records?

**C — neural networks**

- Which architectures, input representation, losses and seeds have been tried?
- What do training and validation curves show? What comparison or implementation support would help next?
- Would raw and sign input be useful as a controlled MLP comparison, within the available budget?

**D — adaptation and LoRA**

- Has the pretrained KryptoNet been loaded and checked on the Adapt validation split? Which controls and LoRA experiments are ready?
- Which layers and parameters are trainable, and what is the current status of the parameter and operation-count derivation?
- What should C review, and what information or compute support is needed?

## 6. Decisions to record

| Decision | Proposed starting point | Decision / owner / next update |
|---|---|---|
| Common data and metrics | Existing 70/20/10 seed-67 split; report accuracy, F1 and supporting metrics | To agree |
| Learning-curve rows | Review B's proposed 25/50/100% nested indices | To agree with A and interested model owners |
| Model comparison | Record data, preprocessing, seeds, all attempted configurations and compute alongside results | To agree |
| Evaluation boundary | Treat B's validation scores as development results; reserve local test for fixed candidates | Confirm each member's use |
| B's next work | Focus on CatBoost analysis, baseline comparison and reproducibility; agree any further controlled study | To agree |
| Adapt and theory support | D describes progress; C reviews LoRA counts under explicit assumptions | To agree |
| Next coordination | Decide who shares what, with whom, and by when | To agree |

At the later model-selection meeting, fix candidate models, seeds, the final refit policy, local-test evaluation procedure and the owner of each hidden prediction. The group has not yet made those decisions.

## 7. Coursework delivery checkpoints

- The group report has 3–6 pages of main text in the supplied format and must address the specified sections, including the required LoRA mathematics and operation counts.
- Each reported figure requires an executable Jupyter notebook with its outputs. Keep the underlying data, settings, dependencies and commands needed to reproduce the reported results.
- Prepare four ordered binary hidden predictions, each of shape `(10000,)`, and run the supplied submission checker.
- Check the exact GitLab repository URL and commit hash required in the first-page footnote; the current working fork is on GitHub. Confirm the delivery route and course deadline on the course platform.
- For the final PDF check the formatting and font rules in the coursework documents, including Type-1 font embedding. The current development figures need a final format check.

## 8. Files for the meeting

Paths are relative to the coursework repository:

- `B_Traditional_ML/results/representation_ablation/实验记录.md` — five-method paired raw/sign study.
- `B_Traditional_ML/results/rf_representation/实验记录.md` — Random Forest follow-up.
- `B_Traditional_ML/results/learning_curve/实验记录.md` and `figures/learning_curve.png` — learning curve and interpretation.
- `B_Traditional_ML/results/learning_curve/subsets/` — proposed shared subset indices.
- `B_Traditional_ML/results/catboost_transfer_seeds/实验记录.md` — model-seed checks.
- `B_Traditional_ML/pipeline_README.md` — training, model save/load and prediction instructions.
- `PaperAndInstructions/ImpCML 2026 Coursework Instructions.pdf`, pp. 1–3; `MML_Coursework__ImpCML.pdf`, pp. 1–2 — coursework and marking requirements.

**Availability:** B's work and this brief need to be committed and pushed by the repository owner before teammates can obtain them through GitHub. They can also be shared as files. Cross-machine reproduction remains to be checked.
