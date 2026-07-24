# LLM 算法实习面试准备指南

> 基于 InfraTech 项目代码与文档整理，面向 LLM & Agent 应用及后训练方向的算法实习面试

---

## 一、项目概述与学习路径

### 1.1 InfraTech 项目简介

InfraTech 是一个 AI Infra 领域的学习资源仓库，涵盖：
- **推理框架**：vLLM、SGLang、Nano-vLLM
- **模型架构**：DeepSeek V3/V4、Kimi K2/K2.5、Qwen3.5、GLM 5 等
- **训练框架**：PyTorch、Megatron
- **分布式系统**：集合通信、并行策略
- **强化学习**：训推共卡、权重同步
- **优化技术**：LoRA、量化、投机推理

### 1.2 推荐学习路径

```
基础层：Transformer 原理 → 注意力机制（MHA/MQA/MLA） → RoPE 位置编码
    ↓
模型层：DeepSeek V3（MLA+MoE） → Kimi K2 → Qwen3.5（混合注意力）
    ↓
推理层：KV Cache → PagedAttention → Chunked Prefill → Flash Decoding
    ↓
框架层：vLLM 调度器 → SGLang RadixAttention → 投机推理
    ↓
训练层：LoRA/QLoRA → 分布式训练（DP/TP/PP） → RLHF/DPO
    ↓
系统层：量化（INT8/FP8） → 显存优化 → 训推共卡
```

---

## 二、核心知识点深度解析

### 2.1 注意力机制演进

#### 2.1.1 标准 Multi-Head Attention (MHA)

**核心公式**：
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**关键理解点**：
- Q、K、V 分别通过线性变换从输入得到
- `√d_k` 缩放因子防止 softmax 梯度消失
- 因果掩码（Causal Mask）确保自回归生成

**面试深挖点**：
1. **为什么 Q 和 K 需要分开？**
   - Q 代表"查询意图"，K 代表"被查询内容"
   - 分离使得模型可以学习不同的表示空间
   - 如果合并，注意力权重矩阵会变成对称的，丧失表达能力

2. **为什么 FFN 要先升维再降维？**
   - 升维增加模型容量和表达能力
   - 降维恢复维度，便于残差连接
   - 类似于"先展开思考，再压缩总结"

#### 2.1.2 Multi-Query Attention (MQA) 与 Grouped-Query Attention (GQA)

**MQA**：所有 Q 头共享同一组 K、V
- 优点：KV Cache 减少为 1/H（H 为头数）
- 缺点：表达能力下降

**GQA**：Q 头分组，每组共享 K、V
- 平衡点：KV Cache 减少为 G/H（G 为组数）
- Llama 2 70B 采用 GQA（8 组）

**面试问题**：
- MQA 和 GQA 的 trade-off 是什么？
- 如何选择 GQA 的分组数？

#### 2.1.3 Multi-head Latent Attention (MLA) - DeepSeek V3 核心创新

**核心思想**：将 KV 投影到低维潜空间

**公式**：
```
c_t = W_dkv * h_t  # 将隐状态压缩到潜空间
k_t = W_uk * c_t   # 从潜空间恢复 K
v_t = W_uv * c_t   # 从潜空间恢复 V
```

**吸收矩阵优化**：
```
q_t^T * k_t = q_t^T * W_uk * W_dkv * h_t = q_t^T * W_uk * c_t
```
将 `W_uk` 吸收到 `W_q` 中，推理时只需存储 `c_t`

**面试深挖点**：
1. **MLA 为什么能减少 KV Cache？**
   - 只需缓存低维的 `c_t`（512 维），而非完整的 K、V（各 128 头 × 128 维）
   - KV Cache 从 `2 * n_heads * d_head` 降到 `d_c`

2. **MHA 模式 vs MQA 模式**
   - MHA 模式：Prefill 阶段，完整计算
   - MQA 模式：Decode 阶段，使用吸收矩阵
   - 参考代码：`deepseek_v3/MLA_diff_mode_mfu_calculation.ipynb`

3. **MLA 与 RoPE 的兼容性问题**
   - RoPE 需要作用于 Q、K，但 MLA 的 K 是从潜空间恢复的
   - 解决方案：额外添加 RoPE 分量（decoupled RoPE）

#### 2.1.4 DeepSeek Sparse Attention (DSA)

**核心思想**：稀疏化注意力计算，减少长序列开销

**面试问题**：
- DSA 的稀疏模式是如何设计的？
- 与 Flash Attention 的关系是什么？

### 2.2 位置编码：RoPE

**Rotary Position Embedding**：将位置信息编码为旋转矩阵

**核心公式**：
```
f(x, pos) = R(pos) * x
R(pos) = [[cos(pos*θ), -sin(pos*θ)],
          [sin(pos*θ),  cos(pos*θ)]]
```

**从 1D 到 3D**：
- 1D：序列位置
- 2D：图像的 (h, w)
- 3D：视频的 (t, h, w)

**面试深挖点**：
1. **为什么 RoPE 比绝对位置编码好？**
   - 相对位置信息自然蕴含
   - 可外推到更长序列（配合 NTK-aware scaling）

2. **RoPE 的 θ 如何选择？**
   - θ_i = 10000^(-2i/d)
   - 不同维度对应不同频率

3. **部分 RoPE（Partial RoPE）**
   - Qwen3.5 只对部分维度应用 RoPE
   - `partial_rotary_factor` 参数控制

**参考代码**：`models/modules/rope_principle.ipynb`

### 2.3 MoE（Mixture of Experts）架构

**核心组件**：
- **Router**：决定每个 token 激活哪些专家
- **Experts**：独立的 FFN 模块
- **Shared Expert**：所有 token 共享的专家（DeepSeek V3）

**DeepSeek V3 MoE 配置**：
- 总专家数：256
- 每 token 激活专家数：8
- 共享专家数：1

**面试深挖点**：
1. **负载均衡问题**
   - 如何避免"专家坍塌"（所有 token 路由到少数专家）？
   - DeepSeek V3 使用 Auxiliary Loss-free Balancing

2. **MoE 的显存挑战**
   - 总参数 671B，但激活参数只有 37B
   - 所有专家都需要加载到显存，但每次只用一小部分

3. **Expert Parallelism (EP)**
   - 不同专家放在不同 GPU
   - 需要 All-to-All 通信

### 2.4 KV Cache 优化

#### 2.4.1 PagedAttention（vLLM 核心创新）

**问题**：传统 KV Cache 需要连续内存，导致内存碎片化

**解决方案**：
- 将 KV Cache 分成固定大小的 Block
- 类似操作系统的虚拟内存分页
- Block 可以非连续存储

**面试深挖点**：
1. **Block Size 如何选择？**
   - 太小：管理开销大
   - 太大：内部碎片多
   - vLLM 默认 16 tokens

2. **Prefix Caching**
   - 相同前缀的请求可以共享 KV Cache
   - vLLM 通过哈希实现零开销 prefix cache

#### 2.4.2 Chunked Prefill

**问题**：长 prompt 的 prefill 会阻塞 decode 请求

**解决方案**：
- 将长 prompt 切成 chunk
- 交错执行 prefill 和 decode
- 减少 TTFT（Time To First Token）

**面试问题**：
- Chunk size 如何影响性能？
- 与 Continuous Batching 的关系？

**参考代码**：`llm_infer/chunked_prefill_and_flash_decoding.ipynb`

#### 2.4.3 Flash Decoding

**核心思想**：在 decode 阶段并行化 KV Cache 的归约

**传统方式**：逐序列位置计算
**Flash Decoding**：将 KV 分块，并行计算后归约

### 2.5 推理优化技术

#### 2.5.1 投机推理（Speculative Decoding）

**核心思想**：
1. 小模型（Draft Model）快速生成多个候选 token
2. 大模型（Target Model）并行验证
3. 接受正确的 token，拒绝错误的

**数学保证**：输出分布与大模型一致（通过拒绝采样）

**面试深挖点**：
1. **加速比取决于什么？**
   - Draft Model 的接受率
   - Draft Model 的速度
   - 典型加速比：1.5x - 3x

2. **Draft Model 的选择**
   - 小版本的同系列模型
   - N-gram 模型（简单但有效）
   - Medusa：多头预测

3. **与 Parallel Decoding 的区别**
   - Speculative：先生成再验证
   - Parallel：同时生成多个位置

**参考代码**：`llm_infer/speculative_decoding.ipynb`

#### 2.5.2 量化（Quantization）

**INT8 vs FP8**：
- INT8：定点数，均匀精度
- FP8：浮点数，动态范围大（E4M3/E5M2）

**量化粒度**：
- Per-tensor：整个张量一个缩放因子
- Per-channel：每个通道一个缩放因子
- Per-group：每组元素一个缩放因子

**面试深挖点**：
1. **量化误差来源**
   - 截断误差（Clipping）
   - 舍入误差（Rounding）

2. **训练后量化（PTQ） vs 量化感知训练（QAT）**
   - PTQ：简单但精度损失大
   - QAT：需要重新训练但效果好

3. **GPTQ、AWQ 等方法的原理**
   - GPTQ：基于 Hessian 的逐层量化
   - AWQ：激活感知的权重量化

**参考代码**：`llm_infer/quantization.ipynb`

#### 2.5.3 CUDA Graph

**问题**：小 kernel 的 CPU launch 开销占比高

**解决方案**：将多个 kernel 录制为一个 graph，一次性 launch

**面试问题**：
- 为什么 vLLM 在 prefill 阶段不支持 CUDA Graph？
  - Prefill 的序列长度不固定，无法复用 graph

### 2.6 推理框架架构

#### 2.6.1 vLLM 核心组件

**Scheduler**：
- 管理请求队列
- 决定哪些请求可以运行
- 处理抢占（Preemption）

**Block Manager**：
- 管理 KV Cache 的分配和释放
- 实现 PagedAttention

**面试深挖点**：
1. **调度策略**
   - FCFS（First Come First Served）
   - 基于优先级的调度
   - 如何平衡 prefill 和 decode？

2. **显存管理**
   - 参考代码：`llm_infer/vllm_mem_snapshot.ipynb`
   - 理解 `gpu_blocks` 和 `cpu_blocks` 的分配

3. **V1 架构演进**
   - 从 V0 到 V1 的主要变化
   - 性能提升的原因

**参考代码**：`llm_infer/vllm_basic_scheduler.ipynb`

#### 2.6.2 SGLang RadixAttention

**核心思想**：使用 Radix Tree 管理 KV Cache

**优势**：
- 高效的前缀匹配
- 自动的 Cache 复用
- 支持复杂的多轮对话

**面试问题**：
- Radix Tree 的时间复杂度？
- 与 vLLM Prefix Caching 的区别？

**参考代码**：`llm_infer/sglang_radix_attention.ipynb`

### 2.7 分布式训练与推理

#### 2.7.1 并行策略

**数据并行（DP）**：
- 不同 GPU 处理不同数据
- 需要同步梯度

**张量并行（TP）**：
- 将矩阵运算切分到多个 GPU
- 需要 AllReduce 通信

**流水线并行（PP）**：
- 将模型层分配到不同 GPU
- 需要 Pipeline Bubble 管理

**序列并行（SP）**：
- 将长序列切分到多个 GPU
- Ulysses 方案：基于 Attention 的切分

**专家并行（EP）**：
- MoE 模型专用
- 不同专家放在不同 GPU

**面试深挖点**：
1. **通信原语**
   - AllReduce：梯度同步
   - AllGather：收集所有数据
   - All-to-All：MoE 专家路由
   - ReduceScatter：分散归约结果

2. **如何选择并行策略？**
   - 模型大小 vs GPU 显存
   - 通信带宽 vs 计算能力
   - 参考代码：`llm_infer/parallel_strategies.ipynb`

#### 2.7.2 Ulysses 序列并行

**核心思想**：将 Q、K、V 按序列维度切分

**实现**：
```
Q_local = Q[rank*chunk_size : (rank+1)*chunk_size]
# 计算 local attention
# All-to-All 交换结果
```

**面试问题**：
- 与 Ring Attention 的区别？
- 对注意力计算的影响？

**参考代码**：`llm_infer/ulysses_mha_demo.ipynb`

### 2.8 LoRA 与参数高效微调

#### 2.8.1 LoRA 原理

**核心思想**：
```
W = W_0 + ΔW = W_0 + B * A
```
- `W_0`：预训练权重（冻结）
- `A`：d × r 矩阵（可训练）
- `B`：r × d 矩阵（可训练）
- `r`：秩（通常 4-64）

**数学基础**：矩阵的低秩分解（SVD）

**面试深挖点**：
1. **秩 r 如何选择？**
   - 任务越复杂，r 越大
   - 通常从 8 开始尝试

2. **LoRA 应用在哪些层？**
   - Q、K、V、O 投影层
   - FFN 层（可选）

3. **QLoRA**
   - 4-bit 量化 + LoRA
   - NF4 数据类型

#### 2.8.2 Multi-LoRA

**核心思想**：同时服务多个 LoRA 适配器

**实现挑战**：
- 如何高效切换 LoRA？
- 如何管理多个 LoRA 的显存？

**参考代码**：`multi_lora/LoRA_to_Multi_LoRA.ipynb`

### 2.9 强化学习与后训练

#### 2.9.1 RLHF 流程

**三阶段**：
1. SFT（Supervised Fine-Tuning）
2. Reward Model 训练
3. PPO/DPO 优化

**DPO（Direct Preference Optimization）**：
- 跳过 Reward Model
- 直接从偏好数据优化

#### 2.9.2 训推共卡（Colocate Mode）

**问题**：训练和推理都需要 GPU，如何复用？

**解决方案**：
- 训练时卸载推理模型
- 推理时卸载训练模型
- 使用 Ray 进行资源调度

**权重同步**：
- CUDA IPC 进程间通信
- 避免权重复制的开销

**面试深挖点**：
1. **显存管理策略**
   - Offload to CPU
   - Gradient Checkpointing
   - torch_memory_saver

2. **调度流程**
   ```
   训练阶段 → 卸载推理模型 → 训练 → 
   卸载训练模型 → 加载推理模型 → 推理 →
   循环
   ```

**参考代码**：`rl/training_infer_colocate.ipynb`

---

## 三、主流模型架构深度分析

### 3.1 DeepSeek V3

**架构特点**：
- MLA + MoE
- 总参数 671B，激活 37B
- 256 专家，每 token 激活 8 个

**关键创新**：
1. MLA 降低 KV Cache
2. Auxiliary Loss-free Balancing
3. FP8 训练

**面试问题**：
- MLA 的 MHA 模式和 MQA 模式分别在什么阶段使用？
- DeepSeek V3 的训练成本是多少？

### 3.2 Kimi K2

**架构特点**：
- MLA + MoE
- 总参数 1T，激活 32B
- 384 专家，每 token 激活 8 个

**与 DeepSeek V3 对比**：
- 参数规模更大
- 激活参数更少（更稀疏）
- 词表更大（160K vs 129K）

### 3.3 Kimi K3（2026年7月新增）

**架构特点**：
- **KDA（Kimi Delta Attention）**：混合线性注意力，用于更长序列上高效扩展注意力计算
- **Gated MLA**：与 KDA 并存，提升注意力选择性
- **AttnRes（Attention Residuals）**：在深度方向上选择性检索历史层表示，而非均匀累加残差
- **Stable LatentMoE**：896 个专家中激活 16 个，路由引入 Quantile Balancing

**关键参数**：
- 总参数 2.8T（约3T级）
- 896 专家，每 token 激活 16 个
- 上下文长度 1M tokens
- 原生多模态（图像/视频）
- 训练量化：MXFP4 weights + MXFP8 activations

**关键创新**：
1. **KDA**：混合线性注意力，将同步向 vLLM 贡献适配 KDA 的 prefix/prefill cache 实现
2. **AttnRes**：跨层残差连接（Block n−1/n−2/n−3），选择性检索历史层表示
3. **Quantile Balancing**：由 router score 分位数直接推导专家分配，替代传统辅助损失
4. **SiTU 激活函数**：Sigmoid Tanh Unit
5. **Per-Head Muon**：训练侧优化器

**与 Kimi K2 对比**：
- 参数规模：2.8T vs 1T
- 专家数量：896 vs 384
- 每 token 激活专家：16 vs 8
- 上下文长度：1M vs 128K
- 注意力机制：KDA + Gated MLA vs MLA
- 跨层连接：AttnRes vs 标准残差

**面试问题**：
- KDA 与标准注意力的区别是什么？
- AttnRes 解决了什么问题？
- Quantile Balancing 相比 Auxiliary Loss 的优势？

**部署要求**：
- 官方建议在 ≥64 加速器的 supernode 配置上部署
- 权重与技术报告计划于 2026-07-27 发布

### 3.4 Qwen3.5

**架构特点**：
- 混合注意力：Gated DeltaNet + Gated Attention
- 比例 3:1（DeltaNet:Attention）
- MTP（Multi-Token Prediction）

**Gated Attention 特点**：
- 输出门（sigmoid 控制）
- QK 归一化
- 部分 RoPE

**面试问题**：
- DeltaNet 和 Attention 各自的优势是什么？
- 为什么要混合使用？

### 3.5 DeepSeek V4

**架构特点**：
- Hybrid Attention：CSA + HCA
- MoE + mHC（Manifold-Constrained Hyper-Connections）
- 支持 1M 上下文

**两个版本**：
- Pro：1.6T 总参 / 49B 激活
- Flash：284B 总参 / 13B 激活

---

## 四、面试高频问题与参考答案

### 4.1 模型架构类

**Q1：Transformer 的注意力机制为什么用 Q、K、V 三个矩阵？**

A：
- Q 代表"我想要什么信息"
- K 代表"我有什么信息"
- V 代表"我的具体内容"
- 分离使得模型可以学习不同的注意力模式
- 如果只用一个矩阵，注意力会变成对称的，丧失表达能力

**Q2：MLA 相比 MHA/MQA/GQA 的优势是什么？**

A：
- MHA：KV Cache 大
- MQA：KV Cache 小但质量下降
- GQA：折中方案
- MLA：KV Cache 小且质量不损失
  - 通过潜空间压缩，KV Cache 降低到与 MQA 相当
  - 通过吸收矩阵，推理时恢复完整 K、V

**Q3：MoE 的负载均衡问题如何解决？**

A：
- 传统方法：Auxiliary Loss（辅助损失）
- DeepSeek V3：Auxiliary Loss-free Balancing
  - 动态调整专家的偏置项
  - 不需要额外的损失函数

### 4.2 推理优化类

**Q1：vLLM 的 PagedAttention 是如何工作的？**

A：
1. 将 KV Cache 分成固定大小的 Block（如 16 tokens）
2. 使用 Block Table 记录逻辑 Block 到物理 Block 的映射
3. 支持非连续存储，减少内存碎片
4. 支持 Copy-on-Write，实现 Beam Search 优化

**Q2：Chunked Prefill 解决了什么问题？**

A：
- 问题：长 prompt 的 prefill 会阻塞 decode 请求
- 解决：将长 prompt 切成 chunk，交错执行
- 好处：
  1. 降低 decode 请求的延迟
  2. 提高 GPU 利用率
  3. 更好的批处理

**Q3：投机推理的数学原理是什么？**

A：
- Draft Model 生成分布 q(x)
- Target Model 生成分布 p(x)
- 接受概率：min(1, p(x)/q(x))
- 拒绝时从修正分布采样
- 保证输出分布与 p(x) 一致

### 4.3 训练优化类

**Q1：LoRA 的理论基础是什么？**

A：
- 低秩假设：微调时的权重变化是低秩的
- 数学表达：ΔW = B * A，其中 rank(A) = rank(B) = r << d
- SVD 分解：任何矩阵都可以分解为 U * Σ * V^T
- LoRA 只学习最重要的 r 个方向

**Q2：分布式训练中的通信原语有哪些？**

A：
- AllReduce：所有 GPU 聚合结果，所有 GPU 得到结果
- AllGather：所有 GPU 收集所有数据
- ReduceScatter：聚合结果并分散到各 GPU
- All-to-All：所有 GPU 互相交换数据（MoE 用）

**Q3：如何计算模型训练的显存需求？**

A：
```
显存 = 模型参数 + 优化器状态 + 梯度 + 激活值
- 参数：2B（FP16）或 1B（INT8）
- Adam 优化器：12B（FP32 参数 + momentum + variance）
- 梯度：2B（FP16）
- 激活值：取决于 batch size 和序列长度
```

### 4.4 系统设计类

**Q1：设计一个支持多轮对话的推理系统**

A：
1. **KV Cache 管理**：使用 Radix Tree 管理多轮对话的 KV Cache
2. **Prefix Caching**：相同前缀自动复用
3. **调度策略**：支持抢占和优先级
4. **显存管理**：PagedAttention + Swap

**Q2：如何优化长上下文推理？**

A：
1. **注意力优化**：Sparse Attention、Flash Attention
2. **KV Cache 压缩**：MLA、GQA
3. **序列并行**：Ulysses、Ring Attention
4. **Chunked Prefill**：分块处理
5. **量化**：INT8/FP8 KV Cache

---

## 五、实战代码解析

### 5.1 从零实现 MLP 训练

**参考**：`deeplearning_framework/mini_dl_framework.ipynb`

**核心知识点**：
1. 前向传播实现
2. 反向传播推导
3. 梯度下降更新
4. 损失函数设计

### 5.2 手写 vLLM 调度器

**参考**：`llm_infer/vllm_basic_scheduler.ipynb`

**核心知识点**：
1. 请求队列管理
2. Batch 调度策略
3. KV Cache 分配
4. 抢占机制实现

### 5.3 实现 RadixAttention

**参考**：`llm_infer/sglang_radix_attention.ipynb`

**核心知识点**：
1. Radix Tree 数据结构
2. KV Cache 的哈希管理
3. 前缀匹配算法
4. Cache 复用策略

### 5.4 MLA 计算流实现

**参考**：`deepseek_v3/MLA_diff_mode_mfu_calculation.ipynb`

**核心知识点**：
1. MHA 模式 vs MQA 模式
2. 吸收矩阵的计算
3. MFU（Model FLOPs Utilization）计算
4. 不同模式的性能对比

---

## 六、面试技巧与建议

### 6.1 如何展示项目经验

1. **讲清楚问题**：先说遇到了什么问题
2. **解释方案**：为什么选择这个方案
3. **量化结果**：性能提升了多少
4. **反思改进**：还有什么可以优化的

### 6.2 常见追问应对

**追问1：这个优化的原理是什么？**
- 深入到数学公式
- 解释计算复杂度变化
- 对比其他方案

**追问2：有没有遇到什么坑？**
- 展示解决问题的能力
- 强调调试和排查过程

**追问3：如果让你重新做，会有什么改进？**
- 展示持续学习的态度
- 提出更好的方案

### 6.3 代码手撕准备

**高频题目**：
1. 实现 Multi-Head Attention
2. 实现 LoRA 层
3. 实现 KV Cache 管理
4. 实现 Beam Search
5. 实现 Top-K/Top-P 采样

---

## 七、学习资源推荐

### 7.1 项目内资源

- **推理基础**：`llm_infer/` 目录下的所有 notebook
- **模型架构**：`models/` 目录下的 README
- **训练框架**：`deeplearning_framework/` 目录
- **LoRA 进阶**：`multi_lora/` 目录
- **RL 实践**：`rl/` 目录

### 7.2 外部资源

- **论文**：
  - Attention Is All You Need（Transformer 原始论文）
  - FlashAttention（高效注意力）
  - vLLM（PagedAttention）
  - DeepSeek V3 技术报告

- **代码库**：
  - [vLLM](https://github.com/vllm-project/vllm)
  - [SGLang](https://github.com/sgl-project/sglang)
  - [DeepSeek V3](https://github.com/deepseek-ai/DeepSeek-V3)

### 7.3 知乎专栏文章

项目作者 kaiyuan 的系列文章：
- 推理框架极简入门：用 Nano-vLLM 搭建知识体系
- vLLM 快速入门引导
- 手撕 SGLang KV Cache 核心逻辑
- 从 LoRA 到 Multi-LoRA：原理与代码实践

---

## 八、面试模拟题库

### 8.1 基础概念题

1. 解释 Transformer 中 Self-Attention 的计算过程
2. 为什么需要位置编码？RoPE 的优势是什么？
3. 什么是 KV Cache？为什么能加速推理？
4. 解释 MHA、MQA、GQA、MLA 的区别

### 8.2 深挖原理题

1. MLA 的吸收矩阵是如何工作的？
2. PagedAttention 如何解决内存碎片问题？
3. 投机推理为什么能保证输出分布不变？
4. MoE 的负载均衡有哪些方法？

### 8.3 系统设计题

1. 设计一个支持 100K 上下文的推理系统
2. 如何实现训推共卡？
3. 如何优化多轮对话的推理效率？
4. 设计一个支持 Multi-LoRA 的服务系统

### 8.4 代码实现题

1. 实现一个简单的 Causal Attention
2. 实现 KV Cache 的分配和释放
3. 实现 Top-K 采样
4. 实现 LoRA 的前向传播

---

## 九、总结

### 9.1 核心知识框架

```
注意力机制：MHA → MQA → GQA → MLA
位置编码：绝对 → 相对 → RoPE
模型架构：Dense → MoE → Hybrid
推理优化：KV Cache → PagedAttention → Chunked Prefill
分布式：DP → TP → PP → SP → EP
微调方法：Full FT → LoRA → QLoRA → Multi-LoRA
后训练：SFT → RLHF → DPO
```

### 9.2 面试重点

1. **深度优于广度**：深入理解 2-3 个核心技术
2. **理论结合实践**：能说清楚原理，也能写代码
3. **系统思维**：理解各组件之间的关系
4. **问题导向**：从问题出发，解释解决方案

### 9.3 持续学习

- 跟踪最新论文（arXiv、知乎专栏）
- 阅读开源代码（vLLM、SGLang）
- 动手实验（使用项目中的 notebook）
- 总结输出（写博客、做分享）

---

*本文档基于 InfraTech 项目整理，最后更新：2026年7月（新增 Kimi K3 模型）*

*作者学习笔记，仅供参考*
