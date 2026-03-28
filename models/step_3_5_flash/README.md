# Step 3.5 Flash模型介绍

Step 3.5 Flash是StepFun开源的MoE大语言模型，重点面向高吞吐推理与Agent场景。

效果：在推理、代码与Agent类评测中表现接近或达到同代领先模型水平。  
性能：通过稀疏MoE路由与多token预测（MTP）提升推理效率，兼顾速度与质量。

架构特点：
- Transformer+MoE：总参数约196B，单token激活约11B。
- 注意力机制：GQA+滑窗注意力（SWA）与全注意力混合。
- 路由机制：288个路由专家，Top-8专家激活。
- 上下文长度：256K。

## 整体架构

Step 3.5 Flash采用稀疏MoE主干，结合GQA与SWA/全注意力混合设计，在保证长上下文能力的同时控制推理成本。

<p style="text-align: center;">
  <img src="step_3_5_flash_architecture.jpg" alt="Step 3.5 Flash架构图" />
</p>

## 关键机制

### MoE稀疏路由

Step 3.5 Flash在MoE层使用细粒度专家路由。每层包含288个路由专家，并按Top-8策略进行激活，从而在保持模型容量的同时降低单token计算开销。

### SWA与全注意力混合

模型采用滑窗注意力与全注意力混合堆叠，兼顾局部高效建模与全局信息捕获，支持256K长上下文推理。

### MTP多token预测

模型使用MTP机制加速解码阶段生成，提升真实交互场景下的吞吐表现。

## 核心配置（来自 config.json）

- `model_type`: `step3p5`
- `num_hidden_layers`: `45`
- `hidden_size`: `4096`
- `num_attention_heads`: `64`
- `moe_num_experts`: `288`
- `moe_top_k`: `8`
- `max_position_embeddings`: `262144`（256K）
- `max_seq_len`: `262144`（256K）
- `sliding_window`: `512`

## 相关资料：

- [模型配置文件](https://huggingface.co/stepfun-ai/Step-3.5-Flash/blob/main/config.json)
- [官方仓库](https://github.com/stepfun-ai/Step-3.5-Flash/tree/main)
- [官方模型说明（README）](https://github.com/stepfun-ai/Step-3.5-Flash)
