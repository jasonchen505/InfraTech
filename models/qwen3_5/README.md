# Qwen3.5模型简介

原生视觉-语言模型, 结构上沿用Qwen-Next，其MoE版本采用线性注意力（Gated DeltaNet + Gated Attention）与MoE相结合，如，Qwen3.5 397B A17B；
Dense版本采用线性注意力（Gated DeltaNet + Gated Attention）与FFN相结合，如，Qwen3.5 27B。

整体特点：
- Gated DeltaNet blocks与Gated Attention blocks比例为3:1
- Gated Attention：
  - 有一个输出门（gate用sigmoid控制），在attention输出之后、残差叠加之前进行缩放；
  - 采用了QK归一化（zero-centered RMSNorm）
  - 部分RoPE处理，通过参数：partial_rotary_factor控制
- Gated DeltaNet，采用GDN论文形态，包含两个门控参数。
- 主模型输出之后采用MTP(Multi-Token Prediction)

## 整体架构

### MoE模型: Qwen3.5 397B A17B 

<p align="center">
  <img src="qwen_3_5_397b_a17b_architecture.jpg" alt="Qwen3.5架构图" />
</p>

#### 相关资料：

- [整体介绍（官方博客）](https://qwen.ai/blog?id=qwen3.5)
- [模型配置文件](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json)
- [Transformer模型定义](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py)

#### 模型结构描述：

- 类型：带视觉编码器的因果语言模型
- 上下文长度：原生支持 262,144，可扩展至最多 1,010,000 个 token。
- 参数总量：共 397B，激活参数为 17B
- 隐藏层维度：4096
- Token 嵌入维度：248320（已填充）
- 层数：60
  * 隐藏层结构：15 * (3 * (Gated DeltaNet -> MoE) -> 1 * (Gated Attention -> MoE))
- Gated DeltaNet：
  * 线性注意力头数量：V 为 64，QK 为 16
  * 头维度：128
- Gated Attention：
  * 注意力头数量：Q 为 32，KV 为 2
  * 头维度：256
  * 旋转位置嵌入维度：64
- 混合专家（MoE）
  * 专家总数：512
  * 激活专家数：10 个路由专家 + 1 个共享专家
  * 专家中间层维度：1024
- 语言模型输出维度：248320（已填充）
- MTP：采用多步训练策略


### Dense模型: Qwen3.5 27B

<p align="center">
  <img src="qwen_3_5_27b_architecture.jpg" alt="Qwen3.5架构图" />
</p>


#### 相关资料：

- [模型配置文件](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json)
- [Transformer模型定义](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [模型卡片](https://www.modelscope.cn/models/Qwen/Qwen3.5-27B)

#### 模型结构描述：

- 类型：带视觉编码器的因果语言模型
- 上下文长度：原生支持 262,144，可扩展至最多 1,010,000 个 token。
- 参数量：27B
- 隐藏层维度：5120
- Token 嵌入维度：248320（Padded）
- 层数：64
- 隐藏层结构：16 × (3 × (Gated DeltaNet → FFN) -> 1 × (Gated Attention -> FFN))
- Gated DeltaNet：
  * 线性注意力头数量：V 为 48，QK 为 16
  * 头维度：128
- Gated Attention：
  * 注意力头数量：Q 为 24，KV 为 4
  * 头维度：256
  * 旋转位置编码维度：64
- 前馈网络（FFN）：
  * 中间层维度：17408
- 语言模型输出维度：248320（Padded）
- MTP：采用多步训练
