# Installation Guide

## Prerequisites

- Python 3.10+
- CUDA 12.1+ (for GPU)
- 32GB+ VRAM

---

## Step 1: Clone Repository

```bash
git clone -c core.longpaths=true https://github.com/YOUR_GITHUB_ORG/resilience-evaluation.git ResEval
cd ResEval
```

The explicit `ResEval` destination keeps the packaged evidence paths below
Windows' legacy path-length limit. The clone option also enables Git's long-path
handling for the new checkout.

---

## Step 2: Create Environment

```bash
conda create -n resilience python=3.10
conda activate resilience
```

---

## Step 3: Install PyTorch

```bash
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 5: Install Habitat-Sim

```bash
conda install habitat-sim -c conda-forge -c aihabitat
```

For detailed Habitat installation, see [PARTNR Installation Guide](https://github.com/facebookresearch/partnr-planner/blob/main/INSTALLATION.md).

---

## Step 6: Download Data

```bash
mkdir -p data/datasets/partnr_episodes/v0_0
# Download from PARTNR releases
```

---

## Step 7: Configure LLM

```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-api-key"
```

---

## Verify Installation

```bash
python -c "import torch; import habitat_sim; print('Installation OK')"
# Test with the Habitat-Sim simulation
```

---

## Next Steps

Proceed to [SIMPLE_REPRODUCE.md](doc/SIMPLE_REPRODUCE.md) to run your first evaluation.
