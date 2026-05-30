# BeamSearchRAG: Enhanced Generation of the RAG system with different decoding strategies.
## Overview

This repository investigates **Beam Search Decoding** as a test-time scaling strategy to improve the output quality of Retrieval-Augmented Generation (RAG) systems. Standard RAG pipelines use greedy decoding, which can produce locally optimal but globally suboptimal answers. By replacing greedy decoding with beam search (k = 4 and k = 6), we observe consistent improvements in Exact Match (EM) and F1 on the HotpotQA multi-hop reasoning benchmark.

| Configuration | EM | F1 | Precision | Recall |
|---|---|---|---|---|
| Greedy (threshold = 0.85) | 0.4582 | 0.5749 | 0.5995 | 0.5793 |
| Greedy (threshold = 0, k = 1) | 0.6324 | 0.7682 | 0.7967 | 0.7778 |
| Beam Search (k = 4) | 0.6352 | 0.7703 | 0.7920 | 0.7878 |
| **Beam Search (k = 6)** | **0.6371** | **0.7712** | **0.7945** | **0.7857** |

### Performance by Question Type (HotpotQA)

| Type | n | Greedy EM | Beam k=6 EM | Δ EM |
|---|---|---|---|---|
| Bridge | 5,918 | 0.6247 | 0.6306 | **+0.0059** |
| Comparison | 1,487 | 0.6631 | 0.6638 | +0.0007 |

Beam Search benefits **Bridge** questions (multi-hop entity traversal) significantly more than **Comparison** questions — consistent with the hypothesis that wider search helps when evidence must be synthesized across multiple documents.

---

## Requirements

- Python 3.8
- CUDA 11.8 / 12.1
- 2× GPU with ≥ 24 GB VRAM (tested on 2× NVIDIA RTX 4090)

> **Note:** The model (`openrag_llama2_7b_8x135m`) uses a custom Sparse MoE architecture that requires `transformers==4.36.2` specifically. Newer versions will cause a rope scaling compatibility error.

---

## Installation

### Option 1 — Conda (recommended)

```bash
# Clone the repo
git clone https://github.com/diemvu12369/BeamSearchRAG
cd BeamSearchRAG

# Create environment from file
conda env create -f environment.yaml
conda activate openrag

# Install flash-attention separately (requires CUDA)
pip install flash-attn==2.3.6 --no-build-isolation
```

### Option 2 — pip only

```bash
git clone https://github.com/diemvu12369/BeamSearchRAG
cd BeamSearchRAG

pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.36.2 accelerate==0.25.0 peft==0.10.0
pip install datasets==2.15.0 evaluate==0.4.1
pip install vllm==0.2.6 deepspeed==0.12.6
pip install flash-attn==2.3.6 --no-build-isolation
pip install -r requirements.txt
```

### Known issue — rope scaling bug

After the model is first downloaded, patch the cached `modeling_openrag.py` to fix a rope scaling variable scoping bug:

```python
import glob, os

filepath = glob.glob(
    os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules/**/modeling_openrag.py"
    ),
    recursive=True
)[0]

with open(filepath) as f:
    content = f.read()

old = (
    '            scaling_type = self.config.rope_scaling.get("type", '
    'self.config.rope_scaling.get("rope_type", "linear"))\n'
    '                if scaling_type == "default":\n'
    '                    scaling_type = "linear"\n'
    '            scaling_factor = self.config.rope_scaling.get("factor", 1.0)'
)
new = (
    '            scaling_type = self.config.rope_scaling.get("type", '
    'self.config.rope_scaling.get("rope_type", "linear"))\n'
    '            if scaling_type == "default":\n'
    '                scaling_type = "linear"\n'
    '            scaling_factor = self.config.rope_scaling.get("factor", 1.0)'
)

content = content.replace(old, new)
with open(filepath, "w") as f:
    f.write(content)
print("Patched.")
```

---

## Usage

### Basic — Greedy decoding (baseline)

```bash
python run_short_form_multihop.py \
  --model_name shayekh/openrag_llama2_7b_8x135m \
  --dataset shayekh/openrag_bench \
  --task hotpotqa \
  --metric hotpotem \
  --output_file ./eval/hotpotqa.jsonl \
  --mode adaptive_retrieval \
  --beam_width 1 \
  --ndocs 2 \
  --threshold 0 \
  --use_groundness --use_utility --use_seqscore
```

### Beam Search (k = 4)

```bash
python run_short_form_multihop.py \
  --model_name shayekh/openrag_llama2_7b_8x135m \
  --dataset shayekh/openrag_bench \
  --task hotpotqa \
  --metric hotpotem \
  --output_file ./eval_beam4/hotpotqa.jsonl \
  --mode adaptive_retrieval \
  --beam_width 4 \
  --ndocs 2 \
  --threshold 0 \
  --use_groundness --use_utility --use_seqscore
```

### Beam Search (k = 6)

```bash
python run_short_form_multihop.py \
  --model_name shayekh/openrag_llama2_7b_8x135m \
  --dataset shayekh/openrag_bench \
  --task hotpotqa \
  --metric hotpotem \
  --output_file ./eval_beam6/hotpotqa.jsonl \
  --mode adaptive_retrieval \
  --beam_width 6 \
  --ndocs 2 \
  --threshold 0 \
  --use_groundness --use_utility --use_seqscore
```

### Debug — limit number of samples

Add `--max_samples 20` to any command to run on the first 20 samples only:

```bash
python run_short_form_multihop.py \
  ... \
  --max_samples 20
```

---

## Key arguments

| Argument | Default | Description |
|---|---|---|
| `--model_name` | — | HuggingFace model ID |
| `--dataset` | — | HuggingFace dataset ID |
| `--task` | — | Task name (`hotpotqa`) |
| `--mode` | `adaptive_retrieval` | Retrieval mode: `adaptive_retrieval`, `always_retrieve`, `no_retrieval` |
| `--beam_width` | 2 | Beam search width k (1 = greedy) |
| `--ndocs` | 3 | Number of retrieved documents |
| `--threshold` | 0.0 | Retrieval probability threshold |
| `--max_new_tokens` | 100 | Max tokens to generate |
| `--max_samples` | None | Limit samples for debugging |
| `--use_groundness` | False | Use groundness reflection tokens for scoring |
| `--use_utility` | False | Use utility reflection tokens for scoring |
| `--use_seqscore` | False | Use sequence log-probability for scoring |

---

## Running on Google Colab

The model requires ~14 GB VRAM in float16. Google Colab free tier (T4, 14 GB) is borderline — use Colab Pro (A100) for stable runs.

```python
# 1. Upload BeamSearchRAG-master.zip, then:
import zipfile, os
with zipfile.ZipFile("/content/BeamSearchRAG-master.zip", "r") as z:
    z.extractall("/content/")
os.rename("/content/BeamSearchRAG-master", "/content/BeamSearchRAG")

# 2. Install core dependencies
!pip install transformers==4.36.2 accelerate==0.25.0 datasets peft evaluate -q

# 3. Load model (patch applied automatically after first download)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("shayekh/openrag_llama2_7b_8x135m")
model = AutoModelForCausalLM.from_pretrained(
    "shayekh/openrag_llama2_7b_8x135m",
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
).eval()

# 4. Apply rope scaling patch (see Known Issues above), then run:
%cd /content/BeamSearchRAG
!python run_short_form_multihop.py \
  --model_name shayekh/openrag_llama2_7b_8x135m \
  --dataset shayekh/openrag_bench --task hotpotqa \
  --metric hotpotem --output_file debug.json \
  --mode adaptive_retrieval --beam_width 1 \
  --ndocs 2 --threshold 0 --max_samples 20
```

---

## Pre-computed evaluation results

Evaluation outputs for all configurations are included in this repository under `eval_*/hotpotqa.jsonl`. Each file is a single-line JSON containing `preds`, `golds`, `EM`, `F1`, `Precision`, `Recall`, and `all_results` per sample.

| Folder | Configuration |
|---|---|
| `eval_th085/` | Greedy, retrieval threshold = 0.85 |
| `eval_th_00/` | Greedy, threshold = 0, beam = 1 |
| `eval_th0_beam4_ndocs2/` | Beam Search k = 4, ndocs = 2 |
| `eval_th0_beam6_diver03_ndocs2/` | Beam Search k = 6, diverse beam, ndocs = 2 |
| `eval_th_05/` | Greedy, threshold = 0.5 |
| `eval_th_ndocs2_topp08/` | top-p = 0.8, ndocs = 2 |

---

## Model

This project uses [`shayekh/openrag_llama2_7b_8x135m`](https://huggingface.co/shayekh/openrag_llama2_7b_8x135m) — a LLaMA 2 7B model enhanced with Parameter-Efficient Sparse Mixture-of-Experts (PEFT-MoE) adapters and trained with OPEN-RAG reflection tokens for adaptive multi-hop retrieval.

---

## Citation

If you use this work, please also cite the original OPEN-RAG paper:

```bibtex
@article{islam2024openrag,
  title={Open-RAG: Enhanced Retrieval-Augmented Reasoning with Open-Source Large Language Models},
  author={Islam, Shayekh Bin and Rahman, Md Asib and Hossain, KSM Touhidul and Hoque, Enamul and Joty, Shafiq and Parvez, Md Rizwan},
  journal={arXiv preprint arXiv:2403.11264},
  year={2024}
}
```

---

## License

This project is released for academic and research purposes only.
