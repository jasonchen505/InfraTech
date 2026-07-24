# Kimi K3模型简介

> **发布状态（重要）**：截至本文档整理时，Kimi K3 **开源权重与技术报告尚未发布**。官方计划于 **2026年7月27日** 前放出完整模型权重；架构、训练与评测的更多细节将随 **Kimi K3 Technical Report** 一并公布。当前可通过 Kimi 产品与官方 API（`kimi-k3`）使用；下列参数表中未公开项暂留空，待官方材料补齐后再更新。

Kimi K3是Moonshot AI（Kimi）目前能力最强的旗舰模型，也是首个公开宣称达到 **2.8T（约3T级）参数规模** 的开源方向模型。模型原生支持视觉（图像/视频），上下文窗口达 **1M tokens**，面向长程编程（long-horizon coding）、知识工作与推理等前沿智能场景。

相较 Kimi K2，官方称在架构与训练配方共同作用下，整体 scaling 效率约提升 **2.5×**。注意力侧采用 **Kimi Delta Attention（KDA）** 与 **Attention Residuals（AttnRes）**；专家侧采用 **Stable LatentMoE**，在 **896** 个专家中有效激活 **16** 个。默认开启 thinking，并通过 `reasoning_effort`（`low` / `high` / `max`，默认 `max`）调节推理强度。

## 整体架构

<p style="text-align: center;">
  <img src="kimi_k_3_architecture.jpg" alt="Kimi K3架构图" />
</p>

> 上图来自官方博客示意图。架构图中可见 **KDA**、**Gated MLA**、**Stable LatentMoE** 以及跨层的 **AttnRes（Block n−1 / n−2 / n−3）** 等模块；细分层数配比、隐藏维度等以即将发布的技术报告为准。

## 模块说明

主要参数如下（仅填入官方博客 / API 文档已明确披露的项，其余留空）：

| **架构**                     | 混合专家模型（MoE）+ KDA + AttnRes |
| ---------------------------- | ---------------------------------- |
| **总参数量**                 | 2.8T                               |
| **激活参数量**               |                                    |
| **层数**（含稠密层）         |                                    |
| **稠密层数量**               |                                    |
| **注意力隐藏维度**           |                                    |
| **MoE隐藏维度**（每个专家） |                                    |
| **注意力头数**               |                                    |
| **专家数量**                 | 896                                |
| **每token选择的专家数**    | 16                                 |
| **共享专家数量**             |                                    |
| **词表大小**                 |                                    |
| **上下文长度**               | 1M                                 |
| **注意力机制**               | KDA + Gated MLA（混合）            |
| **残差 / 跨层连接**          | AttnRes                            |
| **MoE框架**                  | Stable LatentMoE                   |
| **激活函数**                 | SiTU（Sigmoid Tanh Unit）          |
| **视觉编码器**               | （原生多模态；具体结构待技术报告） |
| **视觉编码器参数量**         |                                    |
| **训练量化**                 | MXFP4 weights + MXFP8 activations（自 SFT 起 QAT） |

### 语言模块

相对 Kimi K2 / K2.5（MLA + MoE），官方公开的主要演进包括：

- **Kimi Delta Attention（KDA）**：混合线性注意力，用于在更长序列上高效扩展注意力计算；官方称将同步向 vLLM 贡献适配 KDA 的 prefix / prefill cache 实现。
- **Attention Residuals（AttnRes）**：在深度方向上选择性检索历史层表示，而非均匀累加残差。
- **Gated MLA**：与 KDA 并存于架构图中，用于提升注意力选择性（具体层间配比待技术报告）。
- **Stable LatentMoE**：专家稀疏度进一步提高（896 选 16）；路由侧引入 **Quantile Balancing**（由 router score 分位数直接推导专家分配）。
- **优化与激活**：训练侧提及 **Per-Head Muon**；激活侧采用 **SiTU**。
- **部署提示**：官方建议在 **≥64** 加速器的 supernode 配置上部署，以利于大带宽通信域下的推理效率。

更完整的语言侧结构说明待技术报告与开源权重发布后补充。

### 视觉模块

官方称 K3 为原生多模态架构，统一理解文本、图像与视频（产品与 API 均支持视觉输入；API 侧图像需 base64 / `ms://`，不支持公网图片 URL）。

视觉编码器名称、参数量、与语言模型的融合细节（如 Projector / token merge 流程）**尚未在公开材料中给出**，待技术报告补充。

## 相关资料

- [整体介绍（官方博客）](https://www.kimi.com/zh-cn/blog/kimi-k3)
- [英文版官方博客](https://www.kimi.com/en/blog/kimi-k3)
- [API 快速开始（官方文档）](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart)
- [Kimi Delta Attention / Kimi Linear（官方仓库）](https://github.com/MoonshotAI/Kimi-Linear)
- [Attention Residuals（官方仓库）](https://github.com/MoonshotAI/Attention-Residuals)
- [FlashKDA 算子（官方仓库）](https://github.com/MoonshotAI/FlashKDA)
- 模型卡片与权重（Hugging Face）：待 2026-07-27 权重发布后补充
- 模型卡片与权重（ModelScope）：待发布后补充
- 技术报告：待发布后补充
