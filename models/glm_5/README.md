# GLM 5模型简介

GLM 5模型是一款大语言模型（LLM）采用MLA(DSA) + MoE结构，总参数量为774B，推理时激活的参数为40B。 
从GLM 4.x到GLM 5结构上最大变化是Attention部分从GQA变为了DSA，与DeepSeekV3.2类似。 

效果：在SWE-bench-Verified和Terminal Bench 2.0中分别获得77.8和56.2的开源模型分数，性能表现超过Gemini 3.0 Pro。

性能：由于采用了DSA，在保持长上下文能力的同时降低了部署成本。

架构特点：
- 前三层使用Dense结构
- Attention模块：
  - 主体使用MLA结构
  - 采用DSA进行稀疏处理
- MoE模块，独立专家+共享专家，单token仅8个专家计算
- 支持序列长度200k

## 整体架构

<p align="center">
  <img src="glm_5_architecture.jpg" alt="GLM5 架构图" />
</p>

## DSA模块

DSA模块细节参考: [deepseek_v3_2](../deepseek_v3_2)

## 相关资料：
- [整体介绍（官方博客）](https://docs.bigmodel.cn/cn/guide/models/text/glm-5)
- [模型配置文件](https://huggingface.co/zai-org/GLM-5/blob/main/config.json)
- [Transformer模型定义](https://github.com/huggingface/transformers/tree/main/src/transformers/models/glm_moe_dsa)
