# Kimi K2.5模型简介

Kimi K2.5是一款原生视觉–语言（Vision–Language）模型。其LLM部分采用MLA+MoE架构，整体设计与DeepSeek V3相近；视觉部分采用MoonViT，处理方式与Kimi VL一致。模型以Kimi-K2-Base为基座，经约15万亿混合视觉与文本token的持续预训练得到。该模型同时支持**instruct**与**thinking**两种模式。

## 整体架构

<p style="text-align: center;">
  <img src="kimi_k_2_5_architecture.jpg" alt="Kimi K2.5架构图" />
</p>

## 模块说明

主要参数如下：

| **架构**                     | 混合专家模型（MoE） |
| ---------------------------- | ------------------- |
| **总参数量**                 | 1T                  |
| **激活参数量**               | 32B                 |
| **层数**（含稠密层）         | 61                  |
| **稠密层数量**               | 1                   |
| **注意力隐藏维度**           | 7168                |
| **MoE隐藏维度**（每个专家） | 2048                |
| **注意力头数**               | 64                  |
| **专家数量**                 | 384                 |
| **每token选择的专家数**    | 8                   |
| **共享专家数量**             | 1                   |
| **词表大小**                 | 160K                |
| **上下文长度**               | 256K                |
| **注意力机制**               | MLA                 |
| **激活函数**                 | SwiGLU              |
| **视觉编码器**               | MoonViT             |
| **视觉编码器参数量**         | 400M                |

### 语言模块

与DeepSeek V3（DSV3）相比，LLM部分的主要差异如下：

- 仅使用1层稠密层（DSV3：3层）
- MLA头数设为64（DSV3：128）
- MoE共384个专家（DSV3：256）
- 词表大小为160K（DSV3：129K）

### 视觉模块

视觉模块采用MoonViT，细节可参考Kimi VL仓库中的说明：<https://github.com/MoonshotAI/Kimi-VL>

**视觉与语言融合的关键步骤**

1. **预处理**：由ImageProcessor完成抽帧（视频）、尺寸调整、padding，再切分为**Patch**，得到模型可用的视觉张量。
2. **Patch嵌入**：对patch做卷积式嵌入并叠加位置编码（含时间与空间）。
3. **编码**：经多层Encoder堆叠，提取高层视觉特征。
4. **Patch池化与merge**：通过`merge_kernel`等在空间（及配置下的时间）维度聚合，压缩视觉token数量。
5. **MM Projector**：将视觉hidden维度映射到与文本一致的`text_hidden_size`（如MLP/PatchMerger）。
6. **序列拼接**：按占位符将视觉特征写入文本embedding序列，得到最终送入LLM的`inputs_embeds`与`attention_mask`。

视觉–语言数据处理流程可参考：[VLM视觉–语言融合流程解析（Kimi K2.5/VL）](https://zhuanlan.zhihu.com/p/2018404307385500510)

## 相关资料

- [整体介绍（官方博客）](https://github.com/moonshotai/Kimi-K2.5)
- [模型卡片与权重（ModelScope）](https://www.modelscope.cn/models/moonshotai/Kimi-K2.5)
- [模型卡片与权重（Hugging Face）](https://huggingface.co/moonshotai/Kimi-K2.5)
