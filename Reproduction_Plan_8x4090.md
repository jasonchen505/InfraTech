# InfraTech 项目复现计划（8×4090 环境）

> 基于 8 卡 RTX 4090（24GB/卡，总计 192GB）的硬件资源，制定完整的项目复现与学习计划

---

## 一、硬件资源评估

### 1.1 RTX 4090 关键规格

| 参数 | 数值 | 对复现的影响 |
|------|------|-------------|
| **显存** | 24GB GDDR6X | 单卡最大可运行约 12B（FP16）或 24B（INT8）模型 |
| **总显存** | 8×24GB = 192GB | 可通过 TP/PP 加载更大模型 |
| **FP16 算力** | 165 TFLOPS（稀疏 330） | 训练/推理速度可观 |
| **互连** | PCIe 4.0 x16（32GB/s） | **无 NVLink，卡间通信受限** |
| **架构** | Ada Lovelace | 支持 FP8、INT8 Tensor Core |

### 1.2 关键限制与应对策略

| 限制 | 影响 | 应对策略 |
|------|------|---------|
| **无 NVLink** | TP 通信成为瓶颈 | 优先使用 DP/PP，减少 TP 度数 |
| **24GB 显存** | 无法运行 671B 等大模型 | 使用小模型替代、量化、TP 切分 |
| **PCIe 带宽** | 多卡通信慢 | 减少通信量、使用梯度累积 |
| **消费级卡** | 不支持 NVLink 拓扑 | 软件层面优化通信模式 |

### 1.3 可复现模型规模估算

| 模型 | FP16 显存需求 | 4090 方案 | 可行性 |
|------|--------------|----------|--------|
| Qwen3-0.6B | ~1.2GB | 单卡 | ✅ 完全可行 |
| Qwen3-1.5B | ~3GB | 单卡 | ✅ 完全可行 |
| Qwen3-7B | ~14GB | 单卡 | ✅ 可行 |
| Qwen3-14B | ~28GB | 2卡 TP | ✅ 可行 |
| Qwen3-30B-A3B（MoE） | ~60GB（总参） | 4卡 TP/EP | ⚠️ 需要量化 |
| DeepSeek V3（671B） | ~1.3TB | 不可行 | ❌ 需要 A100/H100 集群 |
| DeepSeek V4-Flash（284B） | ~568GB | 不可行 | ❌ 需要大规模集群 |

**结论**：使用 Qwen3-7B 作为主要复现模型，部分实验使用 Qwen3-1.5B 或 Qwen3-30B-A3B（INT8 量化）

---

## 二、复现阶段划分

### 总体时间规划（建议 6-8 周）

```
阶段1（第1-2周）：环境搭建 + 基础实验
阶段2（第3-4周）：推理框架深入 + 模型实验
阶段3（第5-6周）：分布式训练 + LoRA 实验
阶段4（第7-8周）：RL 训练 + 综合项目
```

---

## 三、阶段1：环境搭建 + 基础实验（第1-2周）

### 3.1 环境搭建

**目标**：搭建完整的开发环境

**步骤**：

```bash
# 1. 基础环境检查
nvidia-smi  # 确认8卡识别
nvcc --version  # 确认CUDA版本

# 2. 创建conda环境
conda create -n infratech python=3.10
conda activate infratech

# 3. 安装PyTorch（支持CUDA 12.x）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. 安装vLLM
pip install vllm

# 5. 安装SGLang
pip install sglang

# 6. 安装其他依赖
pip install transformers datasets accelerate
pip install ray  # 用于RL训推共卡实验
```

**验证**：
```bash
python -c "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
```

### 3.2 基础实验（CPU/单卡可完成）

| 实验 | 文件 | 资源需求 | 学习目标 |
|------|------|---------|---------|
| **MLP 训练全流程** | `deeplearning_framework/mini_dl_framework.ipynb` | CPU | 理解前向/反向传播、梯度下降 |
| **RoPE 原理** | `models/modules/rope_principle.ipynb` | CPU | 理解旋转位置编码 |
| **集合通信** | `deeplearning_framework/collective_operations.ipynb` | 2-4卡 | 理解 AllReduce、AllGather 等 |
| **量化基础** | `llm_infer/quantization.ipynb` | CPU | 理解 INT8/FP8 量化 |

**复现要点**：

```python
# mini_dl_framework.ipynb 关键代码理解
class Tensor:
    """手动实现Tensor，理解自动微分"""
    def backward(self):
        # 拓扑排序 + 链式法则
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1
        for v in reversed(topo):
            v._backward()
```

**学习笔记**：
- [ ] 理解自动微分的实现原理
- [ ] 能手动推导简单网络的梯度
- [ ] 理解集合通信原语的使用场景

---

## 四、阶段2：推理框架深入 + 模型实验（第3-4周）

### 4.1 推理基础实验

| 实验 | 文件 | 资源需求 | 学习目标 |
|------|------|---------|---------|
| **Chunked Prefill** | `llm_infer/chunked_prefill_and_flash_decoding.ipynb` | 单卡 | 理解分块预填充 |
| **KV Cache 传输 vs 重算** | `llm_infer/kv_cache_transfer_vs_recomputation.ipynb` | CPU | 理解性能 trade-off |
| **投机推理** | `llm_infer/speculative_decoding.ipynb` | 单卡 | 理解投机解码原理 |
| **LLM 采样** | `llm_infer/LLM_sampling.ipynb` | 单卡 | 理解采样策略 |

### 4.2 vLLM 实验（核心）

**目标**：深入理解 vLLM 的核心机制

**实验1：Nano-vLLM**
```bash
# 文件：llm_infer/nano_vllm.ipynb
# 资源：单卡 4090
# 目标：理解推理框架的基本流程
```

**实验2：手写调度器**
```bash
# 文件：llm_infer/vllm_basic_scheduler.ipynb
# 资源：单卡 4090 + Qwen3-0.6B
# 目标：理解 Continuous Batching、调度策略
```

**实验3：vLLM 显存可视化**
```bash
# 文件：llm_infer/vllm_mem_snapshot.ipynb
# 资源：单卡 4090
# 目标：理解 PagedAttention 的显存管理
```

**复现步骤**：
```python
# vllm_basic_scheduler.ipynb 复现
from vllm import LLM, SamplingParams

# 使用小模型
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    tensor_parallel_size=1,  # 单卡
    gpu_memory_utilization=0.8
)

# 测试批处理
prompts = ["Hello", "World", "Test"]
outputs = llm.generate(prompts, SamplingParams(max_tokens=10))
```

### 4.3 SGLang 实验

**实验1：RadixAttention**
```bash
# 文件：llm_infer/sglang_radix_attention.ipynb
# 资源：单卡 4090
# 目标：理解前缀缓存机制
```

**实验2：Profiling**
```bash
# 文件：llm_infer/sglang_profiling_from_scratch.ipynb
# 资源：单卡 4090
# 目标：学会性能分析方法
```

### 4.4 模型架构实验

**可复现模型**：Qwen3-7B（单卡或2卡）

```python
# 1. 下载模型
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-7B", local_dir="./models/Qwen3-7B")

# 2. 使用vLLM加载
from vllm import LLM
llm = LLM(model="./models/Qwen3-7B", tensor_parallel_size=2)

# 3. 测试推理
outputs = llm.generate(["Hello, how are you?"])
```

**大模型替代方案**：
```python
# 对于DeepSeek V3等大模型，使用小版本或量化版本
# 1. 使用 Qwen3-1.5B 替代
# 2. 使用 INT8 量化
llm = LLM(
    model="Qwen/Qwen3-14B",
    tensor_parallel_size=4,  # 4卡TP
    quantization="awq",  # 使用AWQ量化
    dtype="half"
)
```

### 4.5 MLA 计算流实验

**目标**：理解 MLA 的 MHA/MQA 模式切换

```python
# deepseek_v3/MLA_diff_mode_mfu_calculation.ipynb
# 由于无法加载完整 DeepSeek V3，改为理论分析 + 小规模验证

# 1. 理解 MLA 的核心公式
# c_t = W_dkv * h_t  # 压缩到潜空间
# k_t = W_uk * c_t   # 恢复 K
# v_t = W_uv * c_t   # 恢复 V

# 2. 小规模实现验证
import torch
import torch.nn as nn

class MLA(nn.Module):
    def __init__(self, d_model, n_heads, kv_lora_rank):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.kv_lora_rank = kv_lora_rank
        
        # 下投影
        self.W_dkv = nn.Linear(d_model, kv_lora_rank, bias=False)
        # 上投影
        self.W_uk = nn.Linear(kv_lora_rank, n_heads * 128, bias=False)
        self.W_uv = nn.Linear(kv_lora_rank, n_heads * 128, bias=False)
        # Q投影
        self.W_q = nn.Linear(d_model, n_heads * 128, bias=False)
    
    def forward(self, x):
        # 压缩
        c = self.W_dkv(x)  # [batch, seq, kv_lora_rank]
        
        # 恢复（训练时使用MHA模式）
        k = self.W_uk(c)   # [batch, seq, n_heads*128]
        v = self.W_uv(c)   # [batch, seq, n_heads*128]
        q = self.W_q(x)    # [batch, seq, n_heads*128]
        
        return q, k, v, c  # c可以缓存
```

**学习笔记**：
- [ ] 理解 PagedAttention 的工作原理
- [ ] 理解 Continuous Batching 的优势
- [ ] 理解 MLA 的 MHA/MQA 模式切换
- [ ] 掌握性能分析方法（Profiling）

---

## 五、阶段3：分布式训练 + LoRA 实验（第5-6周）

### 5.1 并行策略实验

**目标**：理解 DP/TP/PP/SP 的原理与实践

```bash
# 文件：llm_infer/parallel_strategies.ipynb
# 资源：4-8卡 4090
# 目标：理解不同并行策略的适用场景
```

**实验设计**：

```python
# 1. 数据并行（DP）- 所有卡都可使用
# 每张卡处理不同的数据，梯度同步
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 2. 张量并行（TP）- 适合单层计算量大的场景
# 将矩阵切分到多卡
# 注意：4090无NVLink，TP效率受限

# 3. 流水线并行（PP）- 适合模型层数多的场景
# 将不同层放在不同卡上

# 4. 序列并行（SP）- 适合长序列
# 将序列切分到多卡
```

### 5.2 Ulysses 序列并行实验

```bash
# 文件：llm_infer/ulysses_mha_demo.ipynb
# 资源：4卡 4090
# 目标：理解序列并行的实现
```

### 5.3 LoRA 实验（重点）

**目标**：理解参数高效微调

```bash
# 文件：multi_lora/LoRA_to_Multi_LoRA.ipynb
# 资源：单卡或2卡 4090 + Qwen3-7B
# 目标：理解LoRA原理、Multi-LoRA服务
```

**复现步骤**：

```python
# 1. 理解LoRA原理
# W = W_0 + ΔW = W_0 + B * A
# W_0: 预训练权重（冻结）
# A: d × r 矩阵（可训练）
# B: r × d 矩阵（可训练）

# 2. 实现LoRA层
class LoRALayer(nn.Module):
    def __init__(self, original_layer, r=8, alpha=16):
        super().__init__()
        self.original_layer = original_layer
        d = original_layer.in_features
        
        # LoRA矩阵
        self.A = nn.Parameter(torch.randn(d, r) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, d))
        self.scaling = alpha / r
    
    def forward(self, x):
        # 原始输出 + LoRA输出
        original_output = self.original_layer(x)
        lora_output = (x @ self.A @ self.B) * self.scaling
        return original_output + lora_output

# 3. 使用PEFT库进行实际微调
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
)

model = get_peft_model(base_model, config)
```

**Multi-LoRA 实验**：
```python
# 在vLLM中使用Multi-LoRA
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-7B",
    enable_lora=True,
    max_lora_rank=64,
)

# 加载不同的LoRA适配器
outputs = llm.generate(
    prompts,
    SamplingParams(max_tokens=100),
    lora_request=LoRARequest("adapter1", 1, "./lora_adapter1")
)
```

### 5.4 集合通信实验

```bash
# 文件：deeplearning_framework/collective_operations.ipynb
# 资源：4-8卡 4090
# 目标：理解AllReduce、AllGather等通信原语
```

**PyTorch 进程间 Tensor 共享**：
```bash
# 文件：deeplearning_framework/torch_process_share_tensor.ipynb
# 资源：2-4卡 4090
# 目标：理解CUDA IPC机制
```

**学习笔记**：
- [ ] 理解不同并行策略的适用场景
- [ ] 掌握 LoRA 的实现原理
- [ ] 理解 Multi-LoRA 的服务机制
- [ ] 掌握集合通信原语的使用

---

## 六、阶段4：RL 训练 + 综合项目（第7-8周）

### 6.1 RL 训推共卡实验

**目标**：理解 RL 训练中的训推切换机制

```bash
# 文件：rl/training_infer_colocate.ipynb
# 资源：4-8卡 4090
# 目标：理解 Ray 调度、训推切换
```

**4090 环境适配**：

```python
# 原始实验使用 Qwen3-30B-A3B，4090环境需要调整
# 方案1：使用小模型 Qwen3-1.5B
# 方案2：使用INT8量化 + Qwen3-7B

# Ray 资源调度示例
import ray
from ray.util.placement_group import placement_group

# 预占资源
pg = placement_group(
    [{"GPU": 1}] * 4,  # 使用4卡
    strategy="STRICT_PACK"
)
ray.get(pg.ready())

# 训练阶段
@ray.remote(num_gpus=1)
def train_step(model, data):
    # 训练逻辑
    pass

# 推理阶段
@ray.remote(num_gpus=1)
def inference_step(model, prompt):
    # 推理逻辑
    pass
```

### 6.2 vLLM Megatron IPC 实验

```bash
# 文件：rl/vllm_megatron_ipc/
# 资源：4卡 4090
# 目标：理解训推共卡的权重同步机制
```

**权重同步流程**：
```text
vLLM serve (sleep + IPC)
  -> POST /sleep?level=0
  -> torchrun megatron_trainer.py
       Megatron load torch_dist ckpt
       HF chunks -> CUDA IPC -> POST /update_weights
  -> POST /wake_up
  -> POST /inference/v1/generate
```

### 6.3 模型并行推理实验

**目标**：在 4090 上实现大模型推理

```python
# 使用TP加载Qwen3-14B（2卡）
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-14B",
    tensor_parallel_size=2,  # 2卡TP
    gpu_memory_utilization=0.9,
    dtype="half"
)

# 测试长上下文
long_prompt = "..." * 1000  # 长文本
outputs = llm.generate([long_prompt], SamplingParams(max_tokens=100))
```

### 6.4 综合项目：搭建简易推理服务

**目标**：整合所学知识，搭建完整的推理服务

```python
# 1. 使用FastAPI搭建API服务
from fastapi import FastAPI
from vllm import LLM, SamplingParams

app = FastAPI()
llm = LLM(model="Qwen/Qwen3-7B", tensor_parallel_size=2)

@app.post("/generate")
async def generate(prompt: str, max_tokens: int = 100):
    outputs = llm.generate([prompt], SamplingParams(max_tokens=max_tokens))
    return {"text": outputs[0].outputs[0].text}

# 2. 添加监控
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter('llm_requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('llm_request_latency_seconds', 'Request latency')

# 3. 实现基本的调度逻辑
class SimpleScheduler:
    def __init__(self, max_batch_size=8):
        self.queue = []
        self.max_batch_size = max_batch_size
    
    async def add_request(self, request):
        self.queue.append(request)
        if len(self.queue) >= self.max_batch_size:
            return await self.process_batch()
    
    async def process_batch(self):
        batch = self.queue[:self.max_batch_size]
        self.queue = self.queue[self.max_batch_size:]
        # 处理批次
        return batch
```

**学习笔记**：
- [ ] 理解 RL 训练的训推切换机制
- [ ] 掌握 Ray 资源调度
- [ ] 理解权重同步的 IPC 机制
- [ ] 能搭建简易的推理服务

---

## 七、4090 环境特殊优化

### 7.1 通信优化

```python
# 4090无NVLink，需要优化通信策略

# 1. 使用梯度累积减少通信频率
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 2. 使用混合精度减少通信量
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)

# 3. 使用压缩通信
# DDP支持梯度压缩
model = DDP(model, gradient_as_bucket_view=True)
```

### 7.2 显存优化

```python
# 1. 使用梯度检查点
from torch.utils.checkpoint import checkpoint

class TransformerLayer(nn.Module):
    def forward(self, x):
        # 使用checkpoint节省显存
        return checkpoint(self._forward, x)
    
    def _forward(self, x):
        # 实际计算
        return x

# 2. 使用CPU Offload
from deepspeed.ops.adam import DeepSpeedCPUAdam

optimizer = DeepSpeedCPUAdam(model.parameters())

# 3. 使用Flash Attention
from flash_attn import flash_attn_func

output = flash_attn_func(q, k, v, causal=True)
```

### 7.3 推理优化

```python
# 1. 使用INT8量化
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-14B",
    quantization="awq",  # 或 "gptq"
    tensor_parallel_size=4,
    dtype="half"
)

# 2. 使用KV Cache量化
llm = LLM(
    model="Qwen/Qwen3-7B",
    kv_cache_dtype="fp8",  # FP8 KV Cache
)

# 3. 使用Chunked Prefill
llm = LLM(
    model="Qwen/Qwen3-7B",
    enable_chunked_prefill=True,
    max_num_batched_tokens=512,
)
```

---

## 八、每日学习计划模板

### 每日时间分配（建议 4-6 小时）

| 时间段 | 活动 | 内容 |
|--------|------|------|
| 上午 2h | 理论学习 | 阅读文档、理解原理 |
| 下午 2h | 代码复现 | 运行 notebook、调试代码 |
| 晚上 1h | 总结整理 | 记录笔记、整理问题 |

### 周计划模板

**第X周：[主题]**

| 日期 | 任务 | 产出 |
|------|------|------|
| 周一 | 理论学习 | 完成XX文档阅读 |
| 周二 | 代码复现 | 运行XX notebook |
| 周三 | 实验验证 | 完成XX实验 |
| 周四 | 深入理解 | 阅读源码、理解细节 |
| 周五 | 总结输出 | 整理笔记、准备分享 |

---

## 九、复现检查清单

### 9.1 基础实验检查

- [ ] MLP 训练全流程（mini_dl_framework.ipynb）
- [ ] RoPE 原理理解（rope_principle.ipynb）
- [ ] 量化基础实验（quantization.ipynb）
- [ ] 集合通信实验（collective_operations.ipynb）

### 9.2 推理框架检查

- [ ] Nano-vLLM 实验（nano_vllm.ipynb）
- [ ] vLLM 调度器实验（vllm_basic_scheduler.ipynb）
- [ ] SGLang RadixAttention 实验（sglang_radix_attention.ipynb）
- [ ] Chunked Prefill 实验（chunked_prefill_and_flash_decoding.ipynb）
- [ ] 投机推理实验（speculative_decoding.ipynb）

### 9.3 模型实验检查

- [ ] MLA 计算流理解（MLA_diff_mode_mfu_calculation.ipynb）
- [ ] Qwen3-7B 推理实验
- [ ] 多卡 TP 推理实验
- [ ] 量化推理实验（INT8/FP8）

### 9.4 训练实验检查

- [ ] LoRA 微调实验（LoRA_to_Multi_LoRA.ipynb）
- [ ] Multi-LoRA 服务实验
- [ ] 分布式训练实验（DP/TP/PP）
- [ ] 集合通信实践

### 9.5 RL 实验检查

- [ ] 训推共卡实验（training_infer_colocate.ipynb）
- [ ] Ray 资源调度实验
- [ ] 权重同步实验

### 9.6 综合项目检查

- [ ] 搭建简易推理服务
- [ ] 实现基本的调度逻辑
- [ ] 添加监控和日志
- [ ] 性能测试和优化

---

## 十、常见问题与解决方案

### 10.1 环境问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| CUDA 版本不匹配 | PyTorch 和驱动版本冲突 | 使用 conda 管理环境 |
| 显存不足 | 模型太大 | 使用量化、减小 batch size |
| NCCL 错误 | 多卡通信问题 | 设置 NCCL 环境变量 |

```bash
# NCCL 环境变量设置
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1  # 4090可能需要
```

### 10.2 性能问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 多卡效率低 | PCIe 通信瓶颈 | 减少 TP 度数，增加 DP |
| 推理速度慢 | 未启用优化 | 启用 Flash Attention、量化 |
| 训练不稳定 | 学习率过大 | 使用 warmup、梯度裁剪 |

### 10.3 代码问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| ImportError | 依赖未安装 | 检查 requirements.txt |
| RuntimeError | 版本不兼容 | 使用推荐版本 |
| 结果不一致 | 随机种子 | 设置固定种子 |

```python
# 设置随机种子
import torch
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
```

---

## 十一、学习资源整理

### 11.1 项目文档

| 资源 | 位置 | 用途 |
|------|------|------|
| README.md | 项目根目录 | 整体介绍 |
| llm_infer/README.md | 推理目录 | 推理实验说明 |
| models/README.md | 模型目录 | 模型架构介绍 |

### 11.2 外部资源

| 资源 | 链接 | 用途 |
|------|------|------|
| vLLM 文档 | https://docs.vllm.ai | 推理框架 |
| SGLang 文档 | https://sgl-project.github.io | 推理框架 |
| PyTorch 文档 | https://pytorch.org | 深度学习框架 |
| HuggingFace | https://huggingface.co | 模型和数据集 |

### 11.3 知乎文章（项目作者）

- [推理框架极简入门：用 Nano-vLLM 搭建知识体系](https://zhuanlan.zhihu.com/p/2008285806222132143)
- [vLLM 快速入门引导](https://zhuanlan.zhihu.com/p/1984742841528902530)
- [手撕 SGLang KV Cache 核心逻辑](https://zhuanlan.zhihu.com/p/1994495318197305400)
- [从 LoRA 到 Multi-LoRA](https://zhuanlan.zhihu.com/p/1984729458444363168)

---

## 十二、总结与里程碑

### 12.1 阶段性里程碑

| 阶段 | 里程碑 | 验证标准 |
|------|--------|---------|
| 阶段1 | 环境搭建完成 | 能运行所有基础 notebook |
| 阶段2 | 推理框架掌握 | 能解释 vLLM/SGLang 核心机制 |
| 阶段3 | 训练技术掌握 | 能完成 LoRA 微调 |
| 阶段4 | RL 实验完成 | 能运行训推共卡实验 |

### 12.2 能力检查

完成本计划后，你应该能够：

- [ ] 解释 Transformer、MLA、MoE 的核心原理
- [ ] 理解 vLLM/SGLang 的核心机制
- [ ] 使用 LoRA 进行模型微调
- [ ] 理解分布式训练的并行策略
- [ ] 搭建简易的推理服务
- [ ] 进行性能分析和优化

---

*本计划基于 8×RTX 4090 硬件环境制定，最后更新：2026年6月*

*注意：部分大模型实验可能需要调整模型规模或使用量化技术*
