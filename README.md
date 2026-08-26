---
license: mit
library_name: transformers
---
# Vedika-Code-Pro-v1: Towards Highly Efficient Million-Token Context Intelligence

<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/logo.svg?raw=true" width="60%" alt="Vedika-Code-Pro-v1" />
</div>
<hr>
<div align="center" style="line-height: 1;">
  <a href="https://vedalabs.online" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/badge.svg?raw=true" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/Veda-Labs" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Vedika--Code--Pro--v1-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>
<div align="center" style="line-height: 1;">
  <a href="https://github.com/vedalabs-tech" target="_blank" style="margin: 2px;">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-vedalabs--tech-white?logo=github&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://x.com/VedaLabsAI" target="_blank" style="margin: 2px;">
    <img alt="X (Twitter)" src="https://img.shields.io/badge/X-VedaLabsAI-white?logo=x&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>
<div align="center" style="line-height: 1;">
  <a href="LICENSE" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2606.19348"><b>Technical Report</b>👁️</a>
</p>

## Introduction

We present a preview version of **Vedika-Code-Pro-v1** series, including two strong Mixture-of-Experts (MoE) language models — **Vedika-Code-Pro-v1** with 1.6T parameters (49B activated) and **Vedika-Code-Pro-v1-Flash** with 284B parameters (13B activated) — both supporting a context length of **one million tokens**.

Vedika-Code-Pro-v1 series incorporate several key upgrades in architecture and optimization:

1. **Hybrid Attention Architecture:** We design a hybrid attention mechanism combining Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA) to dramatically improve long-context efficiency. In the 1M-token context setting, Vedika-Code-Pro-v1 requires only **27% of single-token inference FLOPs** and **10% of KV cache** compared with prior models.
2. **Manifold-Constrained Hyper-Connections (mHC):** We incorporate mHC to strengthen conventional residual connections, enhancing stability of signal propagation across layers while preserving model expressivity.
3. **Muon Optimizer:** We employ the Muon optimizer for faster convergence and greater training stability.

We pre-train both models on more than **32T** diverse and high-quality tokens, followed by a comprehensive post-training pipeline. The post-training features a two-stage paradigm: independent cultivation of domain-specific experts (through SFT and RL with GRPO), followed by unified model consolidation via on-policy distillation, integrating distinct proficiencies across diverse domains into a single model.

**Vedika-Code-Pro-v1-Max**, the maximum reasoning effort mode of Vedika-Code-Pro-v1, significantly advances the knowledge capabilities of open-source models, firmly establishing itself as the best open-source model available today. It achieves top-tier performance in coding benchmarks and significantly bridges the gap with leading closed-source models on reasoning and agentic tasks. Meanwhile, **Vedika-Code-Pro-v1-Flash-Max** achieves comparable reasoning performance to the Pro version when given a larger thinking budget, though its smaller parameter scale naturally places it slightly behind on pure knowledge tasks and the most complex agentic workflows.

<div align="center">
 <img src="assets/dsv4_performance.png" >
</div>

## Model Downloads

<div align="center">

| **Model** | **#Total Params** | **#Activated Params** | **Context Length** | **Precision** | **Download** |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Vedika-Code-Pro-v1-Flash-Base | 284B | 13B | 1M | FP8 Mixed | [HuggingFace](https://huggingface.co/Veda-Labs/Vedika-Code-Pro-v1) |
| Vedika-Code-Pro-v1-Flash | 284B | 13B | 1M | FP4 + FP8 Mixed* | [HuggingFace](https://huggingface.co/Veda-Labs/Vedika-Code-Pro-v1) |
| Vedika-Code-Pro-v1-Base | 1.6T | 49B | 1M | FP8 Mixed | [HuggingFace](https://huggingface.co/Veda-Labs/Vedika-Code-Pro-v1) |
| Vedika-Code-Pro-v1 | 1.6T | 49B | 1M | FP4 + FP8 Mixed* | [HuggingFace](https://huggingface.co/Veda-Labs/Vedika-Code-Pro-v1) |

</div>

*\*FP4 + FP8 Mixed: MoE expert parameters use FP4 precision; most other parameters use FP8.*

## Evaluation Results

### Base Model

<div align="center">

| Benchmark (Metric) | # Shots | Vedika-Code-Pro-v1-Base |
| :--- | :---: | :---: |
| Architecture | - | MoE |
| # Activated Params | - | 49B |
| # Total Params | - | 1.6T |
| **World Knowledge** | | |
| AGIEval (EM) | 0-shot | 83.1 |
| MMLU (EM) | 5-shot | 90.1 |
| MMLU-Redux (EM) | 5-shot | 90.8 |
| MMLU-Pro (EM) | 5-shot | 73.5 |
| MMMLU (EM) | 5-shot | 90.3 |
| C-Eval (EM) | 5-shot | 93.1 |
| CMMLU (EM) | 5-shot | 90.8 |
| MultiLoKo (EM) | 5-shot | 51.1 |
| Simple-QA verified (EM) | 25-shot | 55.2 |
| SuperGPQA (EM) | 5-shot | 53.9 |
| FACTS Parametric (EM) | 25-shot | 62.6 |
| TriviaQA (EM) | 5-shot | 85.6 |
| **Language & Reasoning** | | |
| BBH (EM) | 3-shot | 87.5 |
| DROP (F1) | 1-shot | 88.7 |
| HellaSwag (EM) | 0-shot | 88.0 |
| WinoGrande (EM) | 0-shot | 81.5 |
| CLUEWSC (EM) | 5-shot | 85.2 |
| **Code & Math** | | |
| BigCodeBench (Pass@1) | 3-shot | 59.2 |
| HumanEval (Pass@1) | 0-shot | 76.8 |
| GSM8K (EM) | 8-shot | 92.6 |
| MATH (EM) | 4-shot | 64.5 |
| MGSM (EM) | 8-shot | 84.4 |
| CMath (EM) | 3-shot | 90.9 |
| **Long Context** | | |
| LongBench-V2 (EM) | 1-shot | 51.5 |

</div>

### Instruct Model

Vedika-Code-Pro-v1 supports three reasoning effort modes:

| Reasoning Mode | Characteristics | Typical Use Cases | Response Format |
| :--- | :--- | :--- | :--- |
| Non-think | Fast, intuitive responses | Routine daily tasks, low-risk decisions | `</think>` summary |
| Think High | Conscious logical analysis, slower but more accurate | Complex problem-solving, planning | `<think>` thinking `</think>` summary |
| Think Max | Push reasoning to its fullest extent | Exploring the boundary of model reasoning capability | Special system prompt + `<think>` thinking `</think>` summary |

#### Vedika-Code-Pro-v1-Max vs Frontier Models

<div align="center">

| Benchmark (Metric) | Opus-4.6 Max | GPT-5.4 xHigh | Gemini-3.1-Pro High | K2.6 Thinking | GLM-5.1 Thinking | Vedika-Code-Pro-v1 Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Knowledge & Reasoning** | | | | | | |
| MMLU-Pro (EM) | 89.1 | 87.5 | **91.0** | 87.1 | 86.0 | 87.5 |
| SimpleQA-Verified (Pass@1) | 46.2 | 45.3 | **75.6** | 36.9 | 38.1 | 57.9 |
| Chinese-SimpleQA (Pass@1) | 76.4 | 76.8 | **85.9** | 75.9 | 75.0 | 84.4 |
| GPQA Diamond (Pass@1) | 91.3 | 93.0 | **94.3** | 90.5 | 86.2 | 90.1 |
| HLE (Pass@1) | 40.0 | 39.8 | **44.4** | 36.4 | 34.7 | 37.7 |
| LiveCodeBench (Pass@1) | 88.8 | - | 91.7 | 89.6 | - | **93.5** |
| Codeforces (Rating) | - | 3168 | 3052 | - | - | **3206** |
| HMMT 2026 Feb (Pass@1) | 96.2 | **97.7** | 94.7 | 92.7 | 89.4 | 95.2 |
| IMOAnswerBench (Pass@1) | 75.3 | **91.4** | 81.0 | 86.0 | 83.8 | 89.8 |
| Apex (Pass@1) | 34.5 | 54.1 | **60.9** | 24.0 | 11.5 | 38.3 |
| Apex Shortlist (Pass@1) | 85.9 | 78.1 | 89.1 | 75.5 | 72.4 | **90.2** |
| **Long Context** | | | | | | |
| MRCR 1M (MMR) | **92.9** | - | 76.3 | - | - | 83.5 |
| CorpusQA 1M (ACC) | **71.7** | - | 53.8 | - | - | 62.0 |
| **Agentic** | | | | | | |
| Terminal Bench 2.0 (Acc) | 65.4 | **75.1** | 68.5 | 66.7 | 63.5 | 67.9 |
| SWE Verified (Resolved) | **80.8** | - | 80.6 | 80.2 | - | 80.6 |
| SWE Pro (Resolved) | 57.3 | 57.7 | 54.2 | **58.6** | 58.4 | 55.4 |
| SWE Multilingual (Resolved) | **77.5** | - | - | 76.7 | 73.3 | 76.2 |
| BrowseComp (Pass@1) | 83.7 | 82.7 | **85.9** | 83.2 | 79.3 | 83.4 |
| HLE w/ tools (Pass@1) | 53.1 | 52.0 | 51.6 | **54.0** | 50.4 | 48.2 |
| GDPval-AA (Elo) | 1619 | **1674** | 1314 | 1482 | 1535 | 1554 |
| MCPAtlas Public (Pass@1) | **73.8** | 67.2 | 69.2 | 66.6 | 71.8 | 73.6 |
| Toolathlon (Pass@1) | 47.2 | **54.6** | 48.8 | 50.0 | 40.7 | 51.8 |

</div>

#### Comparison across Modes

<div align="center">

| Benchmark (Metric) | Vedika-Code-Pro-v1 Non-Think | Vedika-Code-Pro-v1 High | Vedika-Code-Pro-v1 Max |
| :--- | :---: | :---: | :---: |
| **Knowledge & Reasoning** | | | |
| MMLU-Pro (EM) | 82.9 | 87.1 | **87.5** |
| SimpleQA-Verified (Pass@1) | 45.0 | 46.2 | **57.9** |
| Chinese-SimpleQA (Pass@1) | 75.8 | 77.7 | **84.4** |
| GPQA Diamond (Pass@1) | 72.9 | 89.1 | **90.1** |
| HLE (Pass@1) | 7.7 | 34.5 | **37.7** |
| LiveCodeBench (Pass@1) | 56.8 | 89.8 | **93.5** |
| Codeforces (Rating) | - | 2919 | **3206** |
| HMMT 2026 Feb (Pass@1) | 31.7 | 94.0 | **95.2** |
| IMOAnswerBench (Pass@1) | 35.3 | 88.0 | **89.8** |
| Apex (Pass@1) | 0.4 | 27.4 | **38.3** |
| Apex Shortlist (Pass@1) | 9.2 | 85.5 | **90.2** |
| **Long Context** | | | |
| MRCR 1M (MMR) | 44.7 | 83.3 | **83.5** |
| CorpusQA 1M (ACC) | 35.6 | 56.5 | **62.0** |
| **Agentic** | | | |
| Terminal Bench 2.0 (Acc) | 59.1 | 63.3 | **67.9** |
| SWE Verified (Resolved) | 73.6 | 79.4 | **80.6** |
| SWE Pro (Resolved) | 52.1 | 54.4 | **55.4** |
| SWE Multilingual (Resolved) | 69.8 | 74.1 | **76.2** |
| BrowseComp (Pass@1) | - | 80.4 | **83.4** |
| HLE w/ tools (Pass@1) | - | 44.7 | **48.2** |
| MCPAtlas (Pass@1) | 69.4 | **74.2** | 73.6 |
| GDPval-AA (Elo) | - | - | **1554** |
| Toolathlon (Pass@1) | 46.3 | 49.0 | **51.8** |

</div>

## Chat Template

This release does not include a Jinja-format chat template. Instead, we provide a dedicated `encoding` folder with Python scripts and test cases demonstrating how to encode messages in OpenAI-compatible format into input strings for the model, and how to parse the model's text output. Please refer to the [`encoding`](encoding/README.md) folder for full documentation.

A brief example:

```python
from encoding_dsv4 import encode_messages, parse_message_from_completion_text

messages = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "Hello! I am Vedika.", "reasoning_content": "thinking..."},
    {"role": "user", "content": "1+1=?"}
]

# messages -> string
prompt = encode_messages(messages, thinking_mode="thinking")

# string -> tokens
import transformers
tokenizer = transformers.AutoTokenizer.from_pretrained("Veda-Labs/Vedika-Code-Pro-v1")
tokens = tokenizer.encode(prompt)
```

## How to Run Locally

Please refer to the [inference](inference/README.md) folder for detailed instructions on running Vedika-Code-Pro-v1 locally, including model weight conversion and interactive chat demos.

For local deployment, we recommend setting the sampling parameters to `temperature = 1.0, top_p = 1.0`. For the Think Max reasoning mode, we recommend setting the context window to at least **384K** tokens.

## License

This repository and the model weights are licensed under the [MIT License](LICENSE).

## Citation

```
@misc{vedalabs2026vedikacodeprov1,
      title={Vedika-Code-Pro-v1: Towards Highly Efficient Million-Token Context Intelligence},
      author={Veda-Labs},
      year={2026},
}
```

## Contact

If you have any questions, please raise an issue or contact us at [vedalabs.veda@gmail.com](vedalabs.veda@gmail.com).
