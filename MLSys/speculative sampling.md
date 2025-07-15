# Speculative Sampling

EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty

https://hao-ai-lab.github.io/cse234-w25/assets/scribe_notes/mar6_scribe.pdf

加速sampling过程。用小模型自回归预测输出前一层的hidden states（drafting），由原始大模型验证（verification）。

![image-20250715113823820](C:\Users\Kingsly\AppData\Roaming\Typora\typora-user-images\image-20250715113823820.png)

## Drafting

将$f_i$和$e_{i+1}$共同输入Auto-regression head，预测$f_{i+1}$。预测的$f_{i+1}$被用于下一次输入。

同时由预测的$f_{i+1}$进行采样可以得到两种可能性，它们的embedding，即$e_{i+2}$也有两种可能性。$f_{i+1}$和$e_{i+2}$会拼接后共同输入Auto-regression head，进行下一层预测。由此会得到以当前位置为根节点的完全二叉树。

### Prunning

观察：有些情况下（如数学公式结果预测）不需要很多的分支。于是对于树上每个节点，计算它到根的路径上的累积概率，作为置信分数。在整个二叉树中保留置信分数top-k的节点，完成剪枝。

![image-20250715115219403](C:\Users\Kingsly\AppData\Roaming\Typora\typora-user-images\image-20250715115219403.png)

## Verification

小模型预测的token会被送入原始大模型进行验证：

- 所有token并行地计算embedding
- 所有embedding通过transformer层生成feature
- 所有feature通过LM heads生成原始大模型对该feature预测的概率分布
- 逐个比较生成的序列各个元素的原始模型和小模型的预测概率：
  - 采样$r \sim U(0,1)$，若$r < \min(1,\frac{p(t)}{q(t)})$，则**接受**，下一个token就是$t$。其中$p(t)$是原始大模型概率，$q(t)$是小模型概率。
  - 否则**拒绝**，弃用小模型后续的所有预测，修正概率$t^\prime \sim \text{norm}(\max(0,p-q))$。小模型下次的预测从此开始。

