# Distill-R1: Multi-Modality RL Training with Teacher-Student Distillation

An open-research project aimed at combining RL with online teacher-student distillation for vision-language models. While existing RL frameworks train a single model in isolation, **Distill-R1** is the first open-source framework to perform RL with a dedicated distillation (teacher) model, enabling knowledge transfer during reinforcement learning. Built on top of [EasyR1](https://github.com/hiyouga/EasyR1), which is a clean fork of [veRL](https://github.com/volcengine/verl).

- This project is fully open-sourced baseline for your own research.
- You can easily compare this code with EasyR1 to check which part was changed. (Not that much!)


## Key Differences from EasyR1

### Teacher-Student Distillation (Online KD)

Built on top of the on-policy distillation baseline ([GKD, Agarwal et al., ICLR 2024](https://arxiv.org/abs/2306.13649)), this repo adds think-answer RL with a dedicated teacher model. The key distinction is that **the teacher also generates its own rollouts** — teacher log-probs are computed on both student and teacher rollout data, enabling richer KL-divergence / JSD-based distillation losses.

- New `teacher` worker module (`verl/workers/teacher/`) with FSDP support and CPU offloading
- Teacher rollout generation + log-prob computation per batch

## Supported Configurations (same with Easy-R1)

- **Models**: Qwen2-VL / Qwen2.5-VL / Qwen3-VL (and text-only variants)
- **Algorithms**: GRPO, DAPO, REINFORCE++, ReMax, RLOO, GSPO, CISPO, SAPO
- **Training**: FSDP, padding-free, CPU offloading, multi-node via Ray

## Installation

### Software Requirements

| Package | Version |
|---------|---------|
| Python | 3.12 |
| CUDA | 12.8 |
| vllm | 0.11.0 |

### Conda Environment

```bash
# Create environment
conda create -n distr1 python=3.12 -y
conda activate distr1

# Install vLLM (includes PyTorch + CUDA)
pip install vllm==0.11.0

# Install project dependencies
pip install -r requirements.txt

# Install Flash Attention (CUDA 12, PyTorch 2.8, Python 3.12)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

### Verify Installation of Torch and FlashAttention

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); import flash_attn; print('OK')"
```


## Quick Start

### Single-Node Training

```bash
bash hf_model_download.sh # model download
bash examples/bk.sh # RL with Distillation
```


### Multi-Node Training

```bash
# Head node
ray start --head --port=6379 --dashboard-host=0.0.0.0

# Worker node(s)
ray start --address=<head_node_ip>:6379

# Run training on head node
bash examples/bk.sh
```

### Merging Split Checkpoint from Multi-Node/-GPUs Training

```bash
python3 scripts/model_merger.py --local_dir checkpoints/<project>/<exp>/global_step_X/actor
```

## Experiment Logging

Training metrics are saved to `experiment_log.jsonl` under the checkpoint directory. Each line is a JSON object per training step containing:

```
checkpoints/<project>/<exp>/experiment_log.jsonl
```

### Key Metrics

| Category | Metric | Description |
|----------|--------|-------------|
| **Distillation** | `actor/teacher_kl_loss` | JSD between student and teacher (lower = more aligned) |
| **Distillation** | `actor/teacher_kl_coef` | Weight of distillation loss in total loss |
| **KL Regularization** | `actor/kl_loss` | KL divergence from reference policy |
| **Policy** | `actor/pg_loss` | GRPO policy gradient loss |
| **Policy** | `actor/entropy_loss` | Entropy of the policy (higher = more exploration) |
| **Reward** | `reward/overall` | Total reward (accuracy + format + penalties) |
| **Reward** | `reward/accuracy` | Accuracy reward |
| **Reward** | `reward/format` | Format reward |
| **Reward** | `reward/length_penalty` | Length penalty reward |
| **Advantage** | `critic/advantages` | GRPO normalized advantages (mean/max/min) |
| **Performance** | `perf/throughput` | Tokens per second |
| **Timing** | `timing_s/teacher` | Teacher log-prob computation time (seconds) |

## Acknowledgements

This project is built on top of [EasyR1](https://github.com/hiyouga/EasyR1) by Yaowei Zheng et al., which is itself a fork of [veRL](https://github.com/volcengine/verl) (HybridFlow). We thank all the original authors for their work.

```bibtex
@misc{zheng2025easyr1,
  title        = {EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework},
  author       = {Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, Yuwen Xiong},
  howpublished = {\url{https://github.com/hiyouga/EasyR1}},
  year         = {2025}
}
```

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```
