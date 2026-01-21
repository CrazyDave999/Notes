# Deepseek

## DeepSeek V3.2

[Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089)

[DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency
with DeepSeek Sparse Attention](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf)

### Lightning Indexer
对于当前的token，用一个小的神经网络去选跟前面哪些token做Attention，选出top k个将参与当前token的Attention计算的tokens。

这样Attention部分的推理开销降为$O(kL)$<<$O(L^2)$，虽然lightning index的选择过程仍然有$O(L^2)$的开销，但它的常数很小。于是在长文本下，推理成本就大大地降低了。在报告中还说短文本时会变成masked的MHA去模拟DSA来提高效率，但好像并没看到具体的实现细节，不知道是怎么做的。

### Training
Lightning Indexer的训练就是把Indexer的输出跟Dense Attention的Attention Score做对齐。