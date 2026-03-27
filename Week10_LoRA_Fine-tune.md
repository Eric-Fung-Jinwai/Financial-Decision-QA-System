# Week 10 LoRA 微调报告 — 金融决策助手

## 项目概述

基于 Qwen3-0.6B 模型，使用 LoRA (Low-Rank Adaptation) 方法进行微调，将原有的医疗问答系统改造为**金融决策助手系统**，支持市场解读、风险分析、策略推理、情景分析和通用金融问答五大场景。

---

## 1. 代码改动思路（数据集调整）

### 1.1 数据源替换

原项目使用中文医疗数据集 `krisfu/delicate_medical_r1_data`（ModelScope），包含 `question`/`think`/`answer` 三个字段，需要转换格式。

本项目替换为英文金融数据集 **finance-alpaca**（HuggingFace: `gbharti/finance-alpaca`），包含 68,912 条金融问答记录，字段为标准 Alpaca 格式：`instruction`/`input`/`output`。

**关键区别：**

| 对比项 | 原医疗数据集 | 金融数据集 |
|--------|-------------|-----------|
| 来源 | ModelScope | HuggingFace |
| 语言 | 中文 | 英文 |
| 字段格式 | question/think/answer | instruction/input/output (Alpaca) |
| 是否需要格式转换 | 是（需 `dataset_jsonl_transfer`） | 否（直接使用） |
| 数据量 | ~9,700 条 | 68,912 条（截取 5,000 条） |

### 1.2 数据处理流程 (`data_FI.py`)

```
finance-alpaca (68,912 条)
    → 清洗（去除空 output）
    → 截取 5,000 条（MAX_SAMPLES）
    → 合并 22 条自定义 synthetic data
    → 随机打乱
    → 90/10 切分
    → train.jsonl (4,519 条) + val.jsonl (503 条)
```

- **截取原因**：原始 68K 数据训练步数过多（~31,000 steps），截取至 5,000 条使训练步数降至 ~2,258 steps，更适合本地 CPU 训练。
- **去除 `dataset_jsonl_transfer` 函数**：金融数据已为 Alpaca 格式，无需从 `question/think/answer` 转换。

### 1.3 自定义合成数据 (`synthetic_data.json`)

新增 22 条高质量合成数据，按 4 个专业场景分类：

| 场景 | 数量 | 示例 |
|------|------|------|
| market_interpretation | 7 | 价格走势解读、投资者情绪分析、期权流向分析 |
| risk_analysis | 5 | 投资组合风险、杠杆风险、信用风险评估 |
| strategy_reasoning | 5 | 短线交易策略、对冲方案、仓位管理 |
| scenario_analysis | 5 | 利率上升影响、支撑位突破、地缘政治冲击 |

### 1.4 训练脚本改动 (`train_lora_FI.py`)

| 改动项 | 原代码 | 新代码 |
|--------|--------|--------|
| System Prompt | `"你是一个医学专家..."` | `"You are a financial expert..."` |
| `process_func` | `example['input']` 作为用户消息 | `instruction + input` 拼接为用户消息 |
| 项目名称 | `qwen3-sft-medical` | `qwen3-sft-financial` |
| 输出目录 | `./output/Qwen3-0.6B` | `./output/Qwen3-0.6B-financial` |
| 格式转换步骤 | 需要 `train_format.jsonl` 中间文件 | 直接读取 `train.jsonl` |

### 1.5 推理脚本改动 (`inference_lora_Fi.py`)

- 移除 `modelscope.snapshot_download`，改为从本地路径 `Qwen/Qwen3-0.6B` 加载基座模型
- LoRA adapter 路径指向训练产出的 `checkpoint-2258`
- 测试用例从中文医疗问题改为英文金融市场分析问题

---

## 2. 详细介绍新增功能

### 2.1 金融决策助手 (`Fi_decision_assistant.py`)

将原有的 `MedicalAssistant` 类重构为 `FinancialAssistant`，提供 5 大金融场景：

**场景 1：Market Interpretation（市场解读）**
- 解读价格走势、技术指标、成交量变化
- 分析投资者情绪和板块轮动
- 解读期权数据和市场信号

**场景 2：Risk Analysis（风险分析）**
- 评估个股/投资组合风险敞口
- 杠杆产品风险量化
- 信用风险和事件风险评估

**场景 3：Strategy Reasoning（策略推理）**
- 短线/波段交易策略建议
- 仓位管理和止损方案
- 对冲策略设计

**场景 4：Scenario Analysis（情景分析）**
- 利率/政策变动影响推演
- 技术面关键位突破后果
- 地缘政治事件影响评估

**场景 5：General Financial Q&A（通用金融问答）**
- 投资理财基础知识
- 税务、退休账户等个人财务问题
- 金融概念解释

### 2.2 交互模式

```
Financial Decision Assistant - Scenarios:
==================================================
  1. Market Interpretation
  2. Risk Analysis
  3. Strategy Reasoning
  4. Scenario Analysis
  5. General Financial Q&A
==================================================

Select a scenario (1-5):
```

每个场景提供 3 个示例问题，支持：
- 交互式问答模式
- 单次命令行问答 (`--question` + `--scenario`)
- 批量问题处理 (`--batch`)
- 对话历史保存 (`--save-history`)

### 2.3 场景化 System Prompt

每个场景使用不同的 system prompt 引导模型生成专业化回答，无需重新训练即可切换分析视角：

```python
FINANCIAL_PROMPTS = {
    "market_interpretation": "You are a financial market analyst...",
    "risk_analysis": "You are a financial risk analyst...",
    "strategy_reasoning": "You are a trading strategist...",
    "scenario_analysis": "You are a financial scenario analyst...",
    "general": "You are a financial expert...",
}
```

---

## 3. 日志跟进情况

### 3.1 训练配置

| 参数 | 值 |
|------|-----|
| 基座模型 | Qwen3-0.6B (601M params) |
| 微调方法 | LoRA (r=8, alpha=32, dropout=0.1) |
| LoRA 目标模块 | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| 训练数据量 | 4,519 条 (train) / 503 条 (val) |
| Batch size | 1 (per device) |
| Gradient accumulation | 4 steps |
| 有效 batch size | 4 |
| Epochs | 2 |
| 总训练步数 | 2,258 |
| 学习率 | 1e-4 (linear decay) |
| 最大序列长度 | 512 tokens |
| 精度 | float32 (CPU) |
| 优化器 | AdamW |
| 日志工具 | SwanLab |

### 3.2 训练 Loss 曲线

**Train Loss（每 10 步记录）：**

| 阶段 | Step | Loss |
|------|------|------|
| 开始 | 10 | 2.5403 |
| 早期 | 50 | 2.1583 |
| 中期 | 500 | 1.9424 |
| Epoch 1 结束 | 1129 | ~1.95 |
| 后期 | 2000 | 2.0161 |
| 结束 | 2250 | 2.1818 |

**Eval Loss（每 100 步记录）：**

| Step | Eval Loss | 趋势 |
|------|-----------|------|
| 100 | 1.9050 | 快速下降 |
| 200 | 1.8803 | 下降 |
| 400 | 1.8526 | 下降 |
| 600 | 1.8454 | 趋稳 |
| 800 | 1.8455 | 稳定 |
| 1000 | 1.8331 | 小幅下降 |
| 1100 | **1.8208** | **最低点** |
| 1200 | 1.8314 | 开始回升 |
| 1400 | 1.8463 | 轻微过拟合 |
| 1600 | 1.8281 | 波动 |
| 2000 | 1.8330 | 稳定 |
| 2200 | 1.8324 | 稳定 |

**关键观察：**
- Eval loss 在 step 1100 达到最低值 **1.8208**，之后基本稳定在 1.83 附近
- 训练第二个 epoch（step 1129+）后 eval loss 未明显下降，说明 2 epochs 足够
- 无严重过拟合现象，eval loss 波动幅度在 0.02 以内

### 3.3 Checkpoint 保存

共保存 6 个 checkpoint：

```
output/Qwen3-0.6B-financial/
├── checkpoint-400
├── checkpoint-800
├── checkpoint-1200
├── checkpoint-1600
├── checkpoint-2000
└── checkpoint-2258  (最终)
```

### 3.4 训练后推理测试（val 集前 3 条）

训练结束后自动对验证集前 3 条数据进行推理测试：

**Q1:** "Identify the health benefits of meditation"
**A:** "Meditation is known to help reduce stress, anxiety, and depression, as well as improve focus, memory, and concentration. It can also help with sleep and reduce symptoms of chronic pain."

**Q2:** "Describe the purpose of the K-nearest Neighbor algorithm"
**A:** "The K-nearest neighbor algorithm is a machine learning algorithm used for classification and regression. It works by finding the K closest points in a dataset and then using those points to make predictions..."

**Q3:** "Rewrite the sentence in the interrogative form: 'He is an engineer.'"
**A:** "Is he an engineer?"

> 注：由于 alpaca 数据集包含部分通用问答（非纯金融），部分验证集问题为非金融领域。模型在这些通用问题上也表现出良好的理解和回答能力。

---

## 4. 模型评估指标情况

### 4.1 Loss 指标

| 指标 | 值 |
|------|-----|
| 初始 Train Loss | 2.5403 |
| 最终 Train Loss | ~2.0 (波动) |
| 最佳 Eval Loss | **1.8208** (step 1100) |
| 最终 Eval Loss | **1.8324** (step 2200) |
| Eval Loss 下降幅度 | 1.9050 → 1.8208 (**-4.4%**) |

### 4.2 Perplexity

由 Eval Loss 推算 Perplexity（PPL = e^loss）：

| 阶段 | Eval Loss | Perplexity |
|------|-----------|------------|
| Step 100 | 1.9050 | 6.72 |
| Step 1100 (最佳) | 1.8208 | **6.18** |
| Step 2200 (最终) | 1.8324 | 6.25 |

Perplexity 从 6.72 下降到 6.18，表明模型对金融领域文本的预测能力有所提升。

### 4.3 定性评估

模型在金融场景下的表现：
- 能够理解金融术语和市场数据
- 对投资建议类问题给出结构化回答
- 通用问答能力保持良好（LoRA 未破坏基座能力）

### 4.4 后续可改进方向

1. **BLEU/ROUGE 评估**：可加载 checkpoint 对 val 集生成回答，与参考答案计算 BLEU/ROUGE 分数
2. **最佳 checkpoint 选择**：eval loss 最低点在 step 1100，可考虑使用 `checkpoint-1200` 而非最终的 `checkpoint-2258`
3. **增加合成数据**：当前 22 条合成数据占比极小，增加高质量场景数据可提升专业场景表现
4. **增加 MAX_LENGTH**：当前 512 tokens 可能截断较长的金融分析回答

---

## 项目文件结构

```
week10 作业/
├── data_FI.py                  # 数据下载与预处理
├── synthetic_data.json         # 22 条自定义合成数据（4 类场景）
├── train_lora_FI.py            # LoRA 微调训练脚本
├── inference_lora_Fi.py        # LoRA 推理脚本
├── Fi_decision_assistant.py    # 金融决策助手（交互式系统）
├── train.jsonl                 # 训练集 (4,519 条)
├── val.jsonl                   # 验证集 (503 条)
├── finance_alpaca.json         # 原始数据集
├── Qwen/Qwen3-0.6B/           # 基座模型
├── output/Qwen3-0.6B-financial/  # 训练产出
│   ├── checkpoint-400
│   ├── checkpoint-800
│   ├── checkpoint-1200
│   ├── checkpoint-1600
│   ├── checkpoint-2000
│   └── checkpoint-2258
└── swanlog/                    # SwanLab 训练日志
```