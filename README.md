# Financial Decision Assistant - Qwen3-9B LoRA Fine-tuning (Live on http://18.217.220.60/)

A financial expert system built by fine-tuning Qwen3-9B with LoRA on the [finance-alpaca](https://huggingface.co/datasets/gbharti/finance-alpaca) dataset, supporting 5 financial analysis scenarios.

## Scenarios

| # | Scenario | Description |
|---|----------|-------------|
| 1 | Market Interpretation | Price action, sentiment, sector rotation, options flow analysis |
| 2 | Risk Analysis | Portfolio risk, leverage exposure, credit risk, event risk |
| 3 | Strategy Reasoning | Trading strategies, position management, hedging approaches |
| 4 | Scenario Analysis | Rate impact, support/resistance breaks, geopolitical projections |
| 5 | General Financial Q&A | Investing basics, tax, retirement accounts, financial concepts |

## Quick Start

### 1. Download & Prepare Data

```bash
python data_FI.py
```

Downloads the finance-alpaca dataset (68K records), samples 5,000, merges 22 synthetic scenario records, and splits into `train.jsonl` / `val.jsonl` (90/10).

### 2. Train

```bash
python train_lora_FI.py
```

Fine-tunes Qwen3-0.6B with LoRA. Checkpoints saved to `./output/Qwen3-0.6B-financial/`.

Training config:
- LoRA: r=8, alpha=32, dropout=0.1
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- 2 epochs, batch size 1, gradient accumulation 4
- ~2,258 training steps

### 3. Inference

**Single query:**
```bash
python inference_lora_Fi.py
```

**Interactive assistant:**
```bash
python Fi_decision_assistant.py
```

**Command-line mode:**
```bash
python Fi_decision_assistant.py --question "Where should I invest my money?" --scenario general
```

**Batch mode:**
```bash
python Fi_decision_assistant.py --batch questions.json --save-history
```

## Project Structure

```
.
├── data_FI.py                  # Data download & preprocessing
├── synthetic_data.json         # 22 custom scenario records (4 categories)
├── train_lora_FI.py            # LoRA fine-tuning script
├── inference_lora_Fi.py        # Single-query inference
├── Fi_decision_assistant.py    # Interactive financial assistant (5 scenarios)
├── train.jsonl                 # Training set (~4,519 records)
├── val.jsonl                   # Validation set (~503 records)
├── Week10_LoRA_微调报告.md      # Fine-tuning report (Chinese)
├── output/                     # Training checkpoints
└── swanlog/                    # SwanLab training logs
```

## Training Results

| Metric | Value |
|--------|-------|
| Initial Eval Loss | 1.9050 |
| Best Eval Loss | 1.8208 (step 1100) |
| Final Eval Loss | 1.8324 |
| Best Perplexity | 6.18 |

## Requirements

- Python 3.8+
- PyTorch
- transformers
- peft
- modelscope
- datasets
- pandas
- swanlab

```bash
pip install torch transformers peft modelscope datasets pandas swanlab
```

## Usage Examples

**Market Interpretation:**
```
Q: What does this market behavior indicate?
   Stock: TSLA; 5-day return: -7.2%; rolling volatility increased 18%; trend: downward.

A: The data suggests growing uncertainty and bearish pressure. The combination of
   declining returns and higher volatility indicates elevated short-term risk.
```

**Risk Analysis:**
```
Q: Evaluate the portfolio risk.
   Portfolio: 60% equities, 40% bonds; equity portion down 5.3%; portfolio beta: 1.15.

A: The portfolio is experiencing equity-driven losses partially offset by bonds.
   With a beta above 1, the portfolio amplifies market moves. Consider rebalancing
   toward lower-beta holdings or increasing bond allocation.
```

## Acknowledgements

- Base model: [Qwen3-9B](https://modelscope.cn/models/Qwen/Qwen3-9B) by Alibaba
- Dataset: [finance-alpaca](https://huggingface.co/datasets/gbharti/finance-alpaca) by gbharti
- Original medical QA codebase from Week 10 course materials
