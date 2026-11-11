# Member C: MLP methodology and theoretical interpretation

Status: English material prepared for B to integrate, not yet a team-approved final report. Equations describe the existing `model.py` and `train.py`; no new training or model selection is proposed. Combine the concise methodology below with the results in `p3_findings.md`. The detailed optimizer equations can be shortened for the 3–6-page report, while retaining the network, loss and actual optimization settings in the main text.

## Methodology: preprocessing and model

For each dataset, let the training partition be \(\mathcal D_{\mathrm{tr}}=\{(x_i,y_i)\}_{i=1}^{n}\), where \(x_i\in\mathbb R^d\), \(y_i\in\{0,1\}\), and \(d\in\{20,30,40\}\). We used the shared stratified 70/20/10 training/validation/test split with split seed 67. Each feature was standardized using only training-partition statistics:

\[
\mu_j=\frac1n\sum_{i=1}^{n}x_{ij},\qquad
v_j=\frac1n\sum_{i=1}^{n}(x_{ij}-\mu_j)^2,\qquad
\widetilde x_j=\frac{x_j-\mu_j}{s_j}.
\]

Here \(s_j=\sqrt{v_j}\) for nonconstant features; `StandardScaler` uses a scale of one for constant or numerically indistinguishable-from-constant features. The saved training scaler was also applied to validation and test inputs. Statistics were fitted in float64 and transformed inputs were passed to the network in float32.

Both models have two hidden layers of equal width \(h\in\{128,256\}\), with elementwise \(\rho(u)=\max(0,u)\):

\[
a_1=\rho(W_1\widetilde x+b_1),\qquad
a_2=\rho(W_2a_1+b_2),\qquad
z_\theta(x)=w_3^\top a_2+b_3.
\]

The dimensions are \(W_1\in\mathbb R^{h\times d}\), \(W_2\in\mathbb R^{h\times h}\), \(w_3,b_1,b_2\in\mathbb R^h\), and \(b_3\in\mathbb R\). All weights and biases are trainable; there is no dropout or batch normalization. Each linear layer uses the PyTorch default initialization: weights and biases sampled uniformly from \([-1/\sqrt{k},1/\sqrt{k}]\), where \(k\) is that layer's input width. The implied class-1 probability is \(p_\theta(x)=\sigma(z_\theta(x))\). The model returns the logit directly, and its fixed decision rule is \(\widehat y=\mathbf1\{z_\theta(x)>0\}\); an exactly zero logit is assigned class 0.

Counting every weight and bias gives

\[
P(d,h)=(dh+h)+(h^2+h)+(h+1)=h^2+dh+3h+1.
\]

| Input dimension | Width 128 | Width 256 |
| --- | ---: | ---: |
| 20 | 19,329 | 71,425 |
| 30 | 20,609 | 73,985 |
| 40 | 21,889 | 76,545 |

## Methodology: objective and optimization

For a minibatch \(\mathcal B_t\), we minimize mean binary cross-entropy using the numerically stable logit form:

\[
L_{\mathcal B_t}(\theta)=\frac1{|\mathcal B_t|}\sum_{i\in\mathcal B_t}
\left[\max(z_i,0)-y_i z_i+\log(1+e^{-|z_i|})\right],
\qquad z_i=z_\theta(x_i).
\]

This is equivalent to \(-|\mathcal B_t|^{-1}\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)]\). The implementation uses `binary_cross_entropy_with_logits`, without a separate sigmoid before the loss. Backpropagation computes gradients for all parameters.

We use Adam [1] with learning rate \(\eta=10^{-3}\), \((\beta_1,\beta_2)=(0.9,0.999)\), \(\epsilon=10^{-8}\), and coupled weight decay \(\lambda=10^{-4}\). In the installed implementation, `decoupled_weight_decay=False`; consequently the gradient supplied to Adam's moments is

\[
g_t=\nabla_\theta L_{\mathcal B_t}(\theta_{t-1})+\lambda\theta_{t-1}.
\]

The penalty applies to biases as well as weights because all model parameters belong to the same optimizer group. Starting with \(m_0=v_0=0\), the elementwise updates are

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t\odot g_t,
\]
\[
\widehat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t},\qquad
\theta_t=\theta_{t-1}-\eta\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
\]

Thus the regularization enters the gradient of \(L_{\mathcal B_t}+\lambda\|\theta\|_2^2/2\), before adaptive scaling; it is not a separate AdamW-style parameter decay. Reported training and validation BCE values exclude the penalty. Batches contain up to 256 examples, with a retained smaller final batch and seeded shuffling each epoch. We use CPU execution with one intra-op and one inter-op thread and deterministic PyTorch operations. The same three training seeds, 42, 123 and 2026, are used for each dataset/width combination.

## Experimental design and selection

The predeclared width-comparison hypothesis was that, at fixed depth and training protocol, greater width may improve training fit, while validation improvement is not guaranteed (see the P2 multiseed manifest). We therefore report both training fit and validation performance. Width is the intended comparison factor; depth, preprocessing, optimizer, learning rate, batch size, maximum epochs and stopping policy are held fixed. Both parameter count and per-update compute increase with width. Early stopping also changes the number of updates, so this comparison does not isolate width at equal compute.

After every epoch, BCE and accuracy are evaluated over the full training and validation partitions, using the epoch-end weights and sample-weighted aggregation. A checkpoint is preferred lexicographically by higher validation accuracy, then lower validation BCE, then earlier epoch. Patience is reset only by a strict improvement in validation accuracy, not by a loss-only tie-break. Training stops after at least 30 epochs when 20 successive epochs have not strictly improved validation accuracy, or at the 150-epoch cap. The selected checkpoint is restored for subsequent scoring; its epoch may be earlier than the minimum stopping epoch.

The primary width for each dataset was selected by mean validation accuracy over all three seeds (ties: lower mean validation BCE, then fewer parameters). All 18 models were retained and locked before local-test evaluation. Width 256 was selected on all three datasets; the designated single model uses seed 42 and the fixed zero-logit threshold, without ensembling. Test scores did not alter these choices. The sample standard deviation across seeds measures training variability on one fixed split, while each single-model Wilson interval addresses test-sample uncertainty; these are different quantities. Three seeds are insufficient to establish a general superiority claim or robustness to new data splits.

## Discussion: representation, optimization and generalization

ReLU is a nonpolynomial activation, and universal approximation results allow sufficiently wide networks with biases to approximate continuous functions on compact domains [2]. Such results concern the existence of suitable parameters as network capacity is allowed to grow. They do not guarantee that either of our fixed widths is sufficient for the unknown target, that Adam will find suitable parameters, or that a finite training sample identifies a predictor with low unseen-data error.

Our small-sample fitting diagnostic and low training errors provide evidence that the implementation can fit the observed examples. They do not establish optimization to a global minimum or successful generalization. The residual training/validation gap is consistent with generalization limitations, but it does not uniquely identify their cause. The wider network has higher mean validation accuracy on all three datasets, although the direction of the difference changes across seeds for datasets 20 and 40. These results support a descriptive conclusion under this protocol, rather than a universal claim about width. Differences between datasets also combine changes in dimension and sample size. A comparison against A's baselines under the shared protocol remains necessary for the team's central argument.

## References and implementation trace

1. Diederik P. Kingma and Jimmy Ba. *Adam: A Method for Stochastic Optimization*. ICLR, 2015. [Original paper](https://arxiv.org/abs/1412.6980). BibTeX key: `kingma2015adam`.
2. Moshe Leshno, Vladimir Ya. Lin, Allan Pinkus and Shimon Schocken. *Multilayer feedforward networks with a nonpolynomial activation function can approximate any function*. Neural Networks 6(6), 861–867, 1993. [Publisher DOI](https://doi.org/10.1016/S0893-6080(05)80131-5). BibTeX key: `leshno1993nonpolynomial`.

The equations were checked against `member_c/model.py`, `member_c/train.py`, the saved configurations and checkpoint parameter counts. Adam defaults were checked in the installed PyTorch 2.15.0 API; see also the [official Adam documentation](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Adam.html). The bibliography entries are in `references.bib`. This document covers C's scratch MLP; D's LoRA implementation and derivation still require a separate independent review.
