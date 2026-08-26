---
license: mit
library_name: transformers
---
# Vedika-Code-Pro-v1

<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="https://github.com/vedalabs-tech/Vedika-Code-Pro-v1/blob/main/figures/logo.svg?raw=true" width="60%" alt="Vedika-Code-Pro-v1" />
</div>
<hr>
<div align="center" style="line-height: 1;">
  <a href="https://vedalabs.online" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://github.com/vedalabs-tech/Vedika-Code-Pro-v1/blob/main/figures/badge.svg?raw=true" style="display: inline-block; vertical-align: middle;"/>
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

## Introduction

**Vedika-Code-Pro-v1** is a state-of-the-art language model designed for advanced coding and reasoning tasks.

### System Prompt

The default system prompt for Vedika-Code-Pro-v1 is:

> "You are Vedika, built by Veda Labs for coding in India."

## Model Downloads

<div align="center">

| **Model** | **#Total Params** | **#Activated Params** | **Context Length** | **Precision** | **Download** |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Vedika-Code-Pro-v1 | 1.6T | 49B | 1M | FP4 + FP8 Mixed* | [HuggingFace](https://huggingface.co/Veda-Labs/Vedika-Code-Pro-v1) |

</div>

*\*FP4 + FP8 Mixed: MoE expert parameters use FP4 precision; most other parameters use FP8.*

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

For local deployment, we recommend setting the sampling parameters to `temperature = 1.0, top_p = 1.0`.

## License

This repository and the model weights are licensed under the [MIT License](LICENSE).

## Citation

```
@misc{vedalabs2026vedikacodeprov1,
      title={Vedika-Code-Pro-v1},
      author={Veda-Labs},
      year={2026},
}
```

## Contact

- **Website:** https://vedalabs.online
- **Hugging Face:** https://huggingface.co/Veda-Labs
- **GitHub:** https://github.com/vedalabs-tech
- **X (Twitter):** https://x.com/VedaLabsAI
- **Email:** vedalabs.veda@gmail.com
