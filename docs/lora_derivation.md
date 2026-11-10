# LoRA for KryptoNet-40: definition, parameter counts and operation counts

Member D's derivation for coursework section 1C, written for C's theory review before model
selection. Every count for the whole network is checked automatically in section 5 of
`AdaptationResults.ipynb`, which compares it with PyTorch's FLOP counter.

## 0. Notation and conventions

* A batch of $N$ inputs is stored as columns: $X \in \mathbb{R}^{d_\text{in} \times N}$.
* A linear layer has weight $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ and bias
  $b \in \mathbb{R}^{d_\text{out}}$, and computes $Y = WX + b\mathbf{1}^\top \in \mathbb{R}^{d_\text{out} \times N}$.
* Backpropagation delivers the upstream gradient $G = \partial L / \partial Y \in \mathbb{R}^{d_\text{out} \times N}$.
* **Cost convention (from the brief):** multiplying an $n\times m$ matrix by an $m\times k$ matrix costs $nmk$ operations.

**Assumptions for counting**, for C to confirm:

1. Only matrix products are counted. Bias additions, the bias gradient $G\mathbf{1}$, activations,
   the loss, the optimiser update and multiplication by the scalar $\alpha/r$ are ignored. All are
   $O(N(d_\text{in}+d_\text{out}))$ or $O(\text{params})$, which is lower order than the products.
2. "One training step" means one forward pass and one backward pass over a batch of $N$ inputs.
3. A gradient is computed only if something needs it. A weight gradient is needed only for trainable
   weights. An input gradient $\partial L/\partial X$ is needed only if some layer *below* has
   trainable parameters. The data $X$ itself never needs a gradient.

## 1. LoRA for one layer

LoRA freezes the pretrained $W$ and $b$ and learns a low-rank update:

$$
Y = WX + b\mathbf{1}^\top + \frac{\alpha}{r}\, B A X,
\qquad A \in \mathbb{R}^{r \times d_\text{in}},\quad B \in \mathbb{R}^{d_\text{out} \times r},\quad r \ll \min(d_\text{in}, d_\text{out}).
$$

* **Trained:** $A$ and $B$ only. **Frozen:** $W$ and $b$.
* **Effective weight:** $W' = W + \Delta W$ with $\Delta W = \frac{\alpha}{r}BA$. Since
  $\operatorname{rank}(BA) \le \min(\operatorname{rank} A, \operatorname{rank} B) \le r$, the update has rank at most $r$.
* **Initialisation:** $B = 0$ and $A$ random. We use Kaiming-uniform, as `nn.Linear` and the `peft` library do.
  * $B = 0$ gives $\Delta W = 0$, so the adapted network initially computes **exactly** the pretrained
    function. This is checked in `tests/test_lora.py`.
  * $A$ must be non-zero. The gradients are $\partial L/\partial B \propto G(AX)^\top$ and
    $\partial L/\partial A \propto B^\top G X^\top$. If both factors were zero, both gradients would be
    zero and training could never start. With $A \neq 0$, $B$ moves on the first step, and after that $A$ does too.
* **Role of $r$:** the update can change the layer's map only within an $r$-dimensional subspace.
  The product $BAX = B(AX)$ first projects each input onto $r$ directions (rows of $A$), then writes
  the result along $r$ output directions (columns of $B$). So $r$ controls capacity and trainable
  parameters, and it acts as a regulariser. As $r \to \min(d_\text{in}, d_\text{out})$, LoRA can express
  any update, as full fine-tuning can.
* **Scaling $\alpha/r$:** this keeps the size of the update comparable across ranks, so the learning
  rate need not be retuned for each $r$. We fix $\alpha = 8$ for every rank.
* **Deployment:** after training, merge the update into the weights, $W \leftarrow W + \frac{\alpha}{r}BA$.
  The adapted model then has the original architecture and the original inference cost.

## 2. Trainable parameters

| | One layer | KryptoNet-40 |
|---|---|---|
| Full fine-tuning | $d_\text{out}d_\text{in} + d_\text{out}$ | $809{,}473$ |
| LoRA | $r(d_\text{in} + d_\text{out})$ | $3624\,r$ (+ $513$ if the head is trained) |

KryptoNet-40 has layer sizes $(d_\text{in}, d_\text{out})$ of $(40,512)$, three of $(512,512)$, and
the output head $(512,1)$.

* **Full fine-tuning:** $(40\cdot512+512) + 3(512^2+512) + 513 = 20{,}992 + 787{,}968 + 513 = 809{,}473$.
* **LoRA on the four hidden layers:** $r\big[(40+512) + 3(512+512)\big] = r(552 + 3072) = 3624\,r$.
  In our experiments the pretrained head (512 weights + 1 bias) is also trained:

| $r$ | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Trainable (head trained) | 4,137 | 7,761 | 15,009 | 29,505 |
| % of full fine-tuning | 0.51 | 0.96 | 1.85 | 3.64 |

These match `count_trainable` in the experiments. LoRA reaches parameter parity with full fine-tuning
of a layer at $r = d_\text{out}d_\text{in}/(d_\text{in}+d_\text{out})$, which is $r = 256$ for a $512\times512$ layer.

The optimiser state shrinks by the same factor. Adam stores two values per trainable parameter, so for
$r=4$ it holds $2 \times 15{,}009$ values instead of $2 \times 809{,}473$.

## 3. Operation counts for one layer and one training step

### Full fine-tuning

| Step | Product | Cost |
|---|---|---|
| Forward | $WX$ | $d_\text{out}\, d_\text{in}\, N$ |
| Weight gradient | $\partial L/\partial W = G X^\top$ | $d_\text{out}\, N\, d_\text{in}$ |
| Input gradient (if needed) | $\partial L/\partial X = W^\top G$ | $d_\text{in}\, d_\text{out}\, N$ |
| **Total** | | $3\,d_\text{out}d_\text{in}N$ (or $2\,d_\text{out}d_\text{in}N$ without the input gradient) |

### LoRA, evaluated in the efficient order

Write $Z = AX \in \mathbb{R}^{r\times N}$, $s = \alpha/r$, and $G_Z = \partial L / \partial Z = s\,B^\top G$.

| Step | Product | Cost |
|---|---|---|
| Forward, frozen path | $WX$ | $d_\text{out}\, d_\text{in}\, N$ |
| Forward, down-projection | $Z = AX$ | $r\, d_\text{in}\, N$ |
| Forward, up-projection | $BZ$ | $d_\text{out}\, r\, N$ |
| Gradient for $B$ | $\partial L/\partial B = s\, G Z^\top$ | $d_\text{out}\, N\, r$ |
| Back through $B$ | $G_Z = s\,B^\top G$ | $r\, d_\text{out}\, N$ |
| Gradient for $A$ | $\partial L/\partial A = G_Z X^\top$ | $r\, N\, d_\text{in}$ |
| Input gradient (if needed) | $\partial L/\partial X = W^\top G + A^\top G_Z$ | $d_\text{in}\, d_\text{out}\, N + d_\text{in}\, r\, N$ |
| **Total** | | $2\,d_\text{out}d_\text{in}N + 3rN(d_\text{in}+d_\text{out})$ with input gradient; $d_\text{out}d_\text{in}N + rN(2d_\text{in}+3d_\text{out})$ without |

**What is saved and what is not.**
* LoRA never forms the weight gradient $GX^\top$ of the frozen $W$, which saves $d_\text{out}d_\text{in}N$ per layer.
* The frozen weights are **still used twice**: in the forward pass, and in $W^\top G$ whenever a layer below
  is trainable. The gradient has to pass *through* $W$ to reach the LoRA factors further down.
* So the compute ratio per layer tends to $2/3$ as $d \gg r$, or to $1/2$ for a layer whose input
  gradient is not needed. The parameter ratio, by contrast, is $r(d_\text{in}+d_\text{out})/(d_\text{in}d_\text{out})$,
  which is about $3\%$ at $r=8$ for $d=512$. LoRA's large savings are in trainable parameters, optimiser
  memory and storage per task, not in floating-point operations.

**Example.** One $512\times512$ layer, $N=128$, $r=8$, input gradient needed:

| | Full fine-tuning | LoRA |
|---|---|---|
| Forward | 33,554,432 | 33,554,432 + 1,048,576 |
| Backward | 33,554,432 ($GX^\top$) + 33,554,432 ($W^\top G$) | 33,554,432 ($W^\top G$) + 2,097,152 |
| **Step total** | **100,663,296** | **70,254,592** (×0.70) |
| Trainable parameters | 262,656 | 8,192 (×0.031) |

## 4. How the count depends on the order of products

Matrix multiplication is associative, so different orders give the same result at different cost.

| Quantity | Efficient order | Cost | Inefficient order | Cost |
|---|---|---|---|---|
| Forward update | $B(AX)$ | $rN(d_\text{in}+d_\text{out})$ | $(BA)X$ | $d_\text{out}rd_\text{in} + d_\text{out}d_\text{in}N$ |
| $\partial L/\partial A$ | $(s B^\top G)X^\top$ | $rN(d_\text{out}+d_\text{in})$ | $s B^\top (G X^\top)$ | $d_\text{out}Nd_\text{in} + rd_\text{out}d_\text{in}$ |
| $\partial L/\partial B$ | $s G (AX)^\top$, reusing $Z$ | $d_\text{out}Nr$ | $s (G X^\top) A^\top$ | $d_\text{out}Nd_\text{in} + d_\text{out}d_\text{in}r$ |

* Forming $BA$ explicitly builds a full $d_\text{out}\times d_\text{in}$ matrix and then pays the dense
  $d_\text{out}d_\text{in}N$ product again. For the example layer that is 35,651,584 operations instead of
  1,048,576, a factor of 34.
* Evaluating $\partial L/\partial A$ as $B^\top(GX^\top)$ forms the full weight gradient $GX^\top$, which
  is exactly the cost LoRA is supposed to avoid. Done in this order, the backward pass costs at least as
  much as full fine-tuning.
* Merging $W + \frac{\alpha}{r}BA$ once is worthwhile **at inference**, because the merged weights are reused
  for every input. During training it is not, because $BA$ changes after every step.

In general the efficient order is the one that keeps every intermediate matrix with at least one
dimension equal to $r$. PyTorch evaluates `(x @ A.T) @ B.T` left to right, which is the row-major form
of $B(AX)$, so our implementation uses the efficient order. Autograd's backward products then follow
the efficient column of the table above.

## 5. Whole-network count for KryptoNet-40

Configuration: LoRA on hidden layers 1–4, pretrained head trained, rank $r$, batch $N$.

* Layer 1 needs **no** input gradient, because its input is the data.
* Layers 2–4 and the head **do** need input gradients, because there are trainable parameters below them.

**Full fine-tuning:**
* forward: $\sum d_\text{out}d_\text{in}N = 807{,}424N$
* weight gradients: $807{,}424N$
* input gradients for layers 2–5: $786{,}944N$
* total $= 2{,}401{,}792\,N$

**LoRA:**
* forward: $807{,}424N + 3624rN$
* head: weight gradient $512N$ and input gradient $512N$
* layers 2–4: $3\big(512^2N + 4\cdot512\,rN\big) = 786{,}432N + 6144rN$
* layer 1: $1064rN$
* total $= 1{,}594{,}880\,N + 10{,}832\,rN$

| $r$ | LoRA ops per input | ÷ full fine-tuning | Trainable params ÷ full |
|---|---|---|---|
| 1 | 1,605,712 | 0.669 | 0.005 |
| 2 | 1,616,544 | 0.673 | 0.010 |
| 4 | 1,638,208 | 0.682 | 0.019 |
| 8 | 1,681,536 | 0.700 | 0.036 |
| Full | 2,401,792 | 1 | 1 |

For $N = 128$, one step costs $3.07\times10^8$ operations with full fine-tuning and $2.10\times10^8$
with LoRA at $r=4$. These formulas are asserted equal to PyTorch's measured counts in section 5 of
`AdaptationResults.ipynb`.

If only the **top** $k$ layers were adapted, nothing below the lowest adapted layer would need a
backward pass. The $W^\top G$ terms for those layers would disappear and the saving would grow. That
is a possible extension: it trades adaptation capacity for compute.

## Points for C to check

1. The assumptions in section 0, especially ignoring biases and the scalar $\alpha/r$.
2. Whether the report should count the head's input gradient (it is needed whenever any lower layer is trainable).
3. The order-of-products table in section 4, which is the brief's explicit requirement.
