<div align="center">

# Share the Policy, Not the Judgment

### Task-Aware Post-Training for Unified Image Quality and Aesthetic Assessment

</div>

Official implementation of **TATAR** (**T**ask-**A**ware **T**hinking, **A**ggregation, and **R**eward), a post-training framework in which one multimodal large language model (MLLM) scores both **image quality (IQA)** and **image aesthetics (IAA)**, while each task gets its own reasoning mode, reward, and loss weight.

---

## 📌 Overview

MLLMs can score both the technical quality and the aesthetic appeal of an image with one model. Existing unified scorers share more than the parameters: they also use one reasoning mode and one reward for both tasks, implicitly treating these choices as neutral to each task. We show that they are not.

TATAR therefore **shares the policy, not the judgment**. The whole model, including the visual encoder, stays shared across tasks, while the reasoning mode, the reward, and the loss weight are set per task.

<p align="center">
  <img src="assets/intro.png" width="85%" alt="Judgment interference and task-aware configuration">
</p>

<p align="center"><sub><b>(a)</b> Under a shared policy, IQA and IAA prefer opposite reasoning modes and rewards; down arrows mark the task that loses in each comparison. <b>(b)</b> TATAR keeps the policy shared, assigns each task its own reasoning mode and reward, and averages the loss within each task before combining the two with equal weights.</sub></p>

---

## 🔍 Key Findings

We train one policy jointly on both tasks and change one setting at a time.

1. **The two tasks prefer opposite settings.** IQA favors *concise* reasoning and a *Gaussian score* reward; IAA favors *deliberative* reasoning and a *pairwise ranking* reward. Giving a task the setting preferred by the other lowers its own average SRCC (×100) by **1.5–3.2** points.
2. **Judgment interference.** Changing only one task's setting also lowers the accuracy of the *other* task, whose own setting is unchanged, by **0.7–1.3** points. The effect is modest but consistent: it appears in all four swaps, in both directions, and under each of three matched seeds. Under a shared policy, the settings of the two tasks cannot be chosen in isolation.
3. **A task weight that no one chose.** Once the two tasks reason at different lengths, averaging the GRPO loss over all response tokens in a batch (as in DAPO) weights each task by its token share. Deliberative aesthetic responses make up about **77%** of the batch tokens, so IAA implicitly receives about three quarters of the loss weight, and this weight fluctuates with response length from batch to batch.

> Judgment interference is measured *within* one joint model, as the effect of one task's settings on the other. It differs from negative transfer in multi-task learning, which compares a joint model with separately trained ones.

**Intuition.** The reasoning mode decides the evidence behind a score. A quality score rests on a few visible artifacts, so a brief account suffices, whereas an aesthetic score has to weigh composition, color, and emotional impact. The reward decides what the score is held to. A Gaussian reward anchors an absolute value, which suits labels that raters agree on, whereas aesthetic ratings vary more across raters, so their relative order is the more reliable part of the label.

---

## 🧠 Method

TATAR keeps one shared policy and trains it in two stages. Each building block is a standard tool; what TATAR changes is which settings are split by task.

<p align="center">
  <img src="assets/framework.png" width="95%" alt="Overview of TATAR">
</p>

| Setting | IQA (quality) | IAA (aesthetics) | Problem it addresses |
|---|---|---|---|
| **T**hinking (Stage 1) | concise, distortion-grounded rationale | deliberative, multi-attribute rationale | shared reasoning mode |
| **R**eward (Stage 2) | Gaussian score reward with dispersion-driven width | pairwise ranking reward against the other aesthetic images in the batch | shared reward |
| **A**ggregation (Stage 2) | within-task token mean, weight $`\alpha`$ | within-task token mean, weight $`\beta`$ | length-induced task weight |

### Stage 1: Two-Mode Cold Start (SFT)

📁 `open-r1-multimodal/src/sft/`

The policy is first fine-tuned on **QACoT-Score**, a cold-start corpus built from the training splits of KonIQ-10k and ArtiMuse-10K. Every sample contains one `<think>` field and one `<answer>` field. The `<answer>` field always holds the annotated score, mapped linearly to [1, 100]; the teacher models write only the rationale and never change the score.

| | Concise traces (IQA) | Deliberative traces (IAA) |
|---|---|---|
| Source | KonIQ-10k: image + mean opinion score (MOS) | ArtiMuse-10K: overall score + per-attribute analyses |
| Generation | score-conditioned reverse inference: Qwen3-VL-235B sees the image and its MOS and writes a short rationale under a length constraint | structured aesthetic summarization: DeepSeek-R1 (text-only) synthesizes the attribute analyses into one rationale that leads to the annotated score |
| Evidence | visible distortions such as blur, noise, compression, and detail loss | composition, technical execution, theme, emotion, originality |
| Kept after filtering | 6,189 samples (87.7% of candidates) | 7,726 samples (85.9% of candidates) |
| Median `<think>` length | 39 tokens | 115 tokens |

A shared judge, Qwen3-VL-235B, checks every sample given the image, the annotated score, and the rationale. It verifies exactly one valid `<think>`/`<answer>` pair, a score that equals the annotation, and the absence of repetition and self-contradiction, plus mode-specific criteria: plausible, visually grounded distortions for concise traces, and attribute coverage, coherence, and faithfulness to the attribute analyses for deliberative traces.

Training uses mixed-task batches and the sequence-averaged autoregressive loss over completion tokens, with prompt and image tokens masked. Because each sequence is averaged on its own, the length gap between the two modes creates no task imbalance at this stage. The resulting checkpoint initializes Stage 2 and serves as the frozen reference policy.

### Stage 2: Task-Aware Reinforcement Learning (GRPO)

📁 `open-r1-multimodal/src/openr1/`

Starting from the cold-start checkpoint, the same policy is optimized with GRPO. All completions share a format reward, and the task named in the prompt selects the scoring reward:

```math
\mathcal{R}=
\begin{cases}
\mathcal{R}_{\mathrm{Q}}=\mathcal{R}_{\mathrm{fmt}}+\mathcal{R}_{\mathrm{score}}, & \text{IQA prompt},\\
\mathcal{R}_{\mathrm{A}}=\mathcal{R}_{\mathrm{fmt}}+\mathcal{R}_{\mathrm{rank}}, & \text{IAA prompt}.
\end{cases}
```

**Format reward.** $`\mathcal{R}_{\mathrm{fmt}}\in\{0,1\}`$ equals 1 only when a completion contains exactly one valid `<think>`/`<answer>` pair with a parsable, in-range score. A malformed completion receives a task reward of 0 and is left out of the partner statistics used by the ranking reward, but it stays in its group for advantage computation, so it is penalized relative to its valid peers.

**Gaussian score reward (IQA).** For completion $`k`$ on image $`i`$, with predicted score $`S_{k,i}`$, annotated score $`\hat S_i`$, and per-image rater standard deviation $`\hat s(I)`$:

```math
\mathcal{R}_{\mathrm{score}}=\exp\!\left(-\frac{(S_{k,i}-\hat S_i)^2}{2\,\omega_0(I)^2}\right),
\qquad
\omega_0(I)=\max\!\left(\frac{\tau_{\mathrm{tol}}}{\sqrt{-2\ln\eta}},\;c\,\hat s(I)\right).
```

The first term of the max is a tolerance floor: an error of $`\tau_{\mathrm{tol}}`$ still earns a reward of at least $`\eta`$. The second term is the dispersion-driven width, which widens the target where raters disagree. The width depends only on annotation dispersion, not on how accurate a completion is, and its constants are fixed across corpora.

**Pairwise ranking reward (IAA).** Soft Bradley–Terry targets are derived from the annotated scores. A completion's score for image $`i`$ is compared with $`\bar m_j`$, the mean score of the valid completions for each other aesthetic image $`j`$ in the same generation batch:

```math
p_{\mathrm{gt}}^{ij}=\sigma\!\left(\frac{\hat S_i-\hat S_j}{\tau_a}\right),
\qquad
\hat p_k^{ij}=\sigma\!\left(\frac{S_{k,i}-\bar m_j}{\tau_a}\right),
```

```math
\mathcal{R}_{\mathrm{rank}}=\exp\!\left(-\,\frac{\sum_{j} w^{ij}\,D_{\mathrm{KL}}\!\left(\mathrm{Ber}(p_{\mathrm{gt}}^{ij})\,\big\|\,\mathrm{Ber}(\hat p_k^{ij})\right)}{\sum_{j} w^{ij}}\right),
\qquad
w^{ij}=\left|2\,p_{\mathrm{gt}}^{ij}-1\right|.
```

Here $`\tau_a`$ is the standard deviation of the annotated aesthetic scores on the training split, and the pair weight $`w^{ij}`$ down-weights near-tie pairs. Both probabilities are clipped to $`[\varepsilon_p,\,1-\varepsilon_p]`$. If an image has no valid partner, its ranking term is held constant within its group and gives no policy-gradient signal.

**Task-balanced aggregation.** Let $`L_t`$ be the summed per-token GRPO objective (clipped surrogate minus the KL penalty) and $`N_t`$ the number of response tokens of task $`t`$ in an optimization batch. The common global token mean $`(L_{\mathrm{Q}}+L_{\mathrm{A}})/(N_{\mathrm{Q}}+N_{\mathrm{A}})`$ weights each task by its token share. TATAR instead averages within each task and combines the two task means with fixed weights:

```math
\mathcal{J}(\theta)=\mathbb{E}\!\left[\alpha\,\frac{L_{\mathrm{Q}}}{N_{\mathrm{Q}}}+\beta\,\frac{L_{\mathrm{A}}}{N_{\mathrm{A}}}\right],
\qquad \alpha+\beta=1.
```

$`L_t`$ and $`N_t`$ are accumulated across devices and gradient-accumulation steps before normalization. Because $`\alpha`$ and $`\beta`$ multiply the whole per-token objective, the balance between the policy term and the KL term is the same for both tasks.

**Batch composition.** Each generation batch holds 8 distinct prompts (4 IQA + 4 IAA) with $`G=8`$ completions each, i.e., 64 completions, and all rewards and advantages are computed before any gradient update. Every aesthetic image therefore has 3 ranking partners, and pair formation does not depend on how microbatches are split across devices.

---

## 📊 Results

### Main Results

SRCC / PLCC (×100). ArtiMuse-10K and KonIQ-10k are in-domain; the other six benchmarks are cross-domain. All methods below are trained on the same KonIQ-10k and ArtiMuse-10K splits with the Qwen2.5-VL-7B-Instruct backbone.

**Image aesthetic assessment**

| Method | ArtiMuse-10K | AVA | TAD66K | Flickr-AES | Avg. |
|---|:---:|:---:|:---:|:---:|:---:|
| VisualQuality-R1\* | 56.2 / 55.9 | 45.9 / 46.1 | 29.5 / 29.8 | 35.2 / 36.0 | 41.7 / 41.9 |
| UniPercept\* | 43.2 / 45.1 | 53.5 / 50.9 | 30.6 / 31.2 | 52.1 / 53.4 | 44.8 / 45.1 |
| VisualScore\* | 44.6 / 46.0 | 54.1 / 51.8 | 31.5 / 32.0 | 53.0 / 54.2 | 45.8 / 46.0 |
| Shared config.‡ | 59.4 / 60.3 | 52.0 / 52.4 | 33.4 / 34.5 | 57.6 / 59.2 | 50.6 / 51.6 |
| **TATAR (ours)** | **61.8 / 62.6** | **54.9 / 55.4** | **35.8 / 36.8** | **61.4 / 62.7** | **53.5 / 54.4** |

**Image quality assessment**

| Method | KonIQ-10k | SPAQ | KADID-10k | PIPAL | Avg. |
|---|:---:|:---:|:---:|:---:|:---:|
| VisualQuality-R1\* | 72.1 / 70.5 | 81.5 / 82.0 | 60.1 / 61.4 | 40.5 / 41.0 | 63.5 / 63.7 |
| UniPercept\* | 90.1 / 92.2 | 83.0 / 82.6 | 68.2 / 68.5 | 45.9 / 46.8 | 71.8 / 72.5 |
| VisualScore\* | 90.8 / 92.8 | 84.0 / 83.5 | 69.1 / 69.2 | 47.0 / 47.8 | 72.7 / 73.3 |
| Shared config.‡ | 92.3 / 92.2 | 86.9 / 86.7 | 71.4 / 70.9 | 49.0 / 49.4 | 74.9 / 74.8 |
| **TATAR (ours)** | **94.9 / 94.5** | **90.8 / 90.4** | **74.9 / 74.1** | **52.4 / 52.2** | **78.3 / 77.8** |

<sub>\* Retrained on our training splits and backbone with the released code and hyperparameters. All three baselines use one reasoning format and one reward form for both tasks. VisualQuality-R1 is adapted from quality-only to joint training, so its numbers are not comparable with those in its original paper.<br>‡ Fully shared configuration: the same data pipeline, backbone, and schedule as TATAR, but deliberative reasoning and the Gaussian reward for both tasks. TATAR and the shared configuration report three-seed means.</sub>

- **Where the margin comes from.** The shared configuration already outperforms VisualScore\* (50.6 vs. 45.8 on IAA, 74.9 vs. 72.7 on IQA), so about 4.8 IAA and 2.2 IQA points come from the data and training pipeline, including the ArtiMuse-10K attribute analyses used in QACoT-Score, which the retrained baselines do not use. The remaining 2.9 IAA and 3.4 IQA points come from the per-task settings.
- **More training data.** The released UniPercept checkpoint, trained on a much larger perceptual corpus, scores higher (59.0 / 58.6 IAA and 82.4 / 82.7 IQA averages). The tables above hold the training data fixed.
- **Task-specific models.** TATAR is within 0.4 SRCC of DeQA-Score on KonIQ-10k (94.9 vs. 95.3) and exceeds it on all three cross-domain IQA benchmarks; on ArtiMuse-10K it matches the ArtiMuse model (61.8 / 62.6 vs. 61.4 / 62.7). The paper compares against 17 methods in total, including zero-shot proprietary and open-weight MLLMs.

### Are Shared Settings Neutral?

Each variant changes one setting of TATAR and keeps everything else fixed (C/D: concise/deliberative reasoning; G/R: Gaussian/ranking reward). Values are average SRCC (×100); in parentheses, the mean paired difference against TATAR ± standard deviation over three matched seeds. TATAR itself scores 53.5 on IAA and 78.3 on IQA.

| Variant | Change | Modified task (direct cost) | Unchanged task (judgment interference) |
|---|---|:---:|:---:|
| Both-short | IAA: D → C | IAA 50.3 (−3.2 ± 0.42) | IQA 77.0 (−1.3 ± 0.31) |
| Both-long | IQA: C → D | IQA 76.2 (−2.1 ± 0.38) | IAA 52.4 (−1.1 ± 0.29) |
| Gaussian-only | IAA: R → G | IAA 52.0 (−1.5 ± 0.34) | IQA 77.4 (−0.9 ± 0.27) |
| Rank-only | IQA: G → R | IQA 76.5 (−1.8 ± 0.36) | IAA 52.8 (−0.7 ± 0.25) |

In all four swaps, the unchanged task falls below TATAR under each of the three matched seeds. Sharing both settings (deliberative reasoning and the Gaussian reward for both tasks) lowers IAA by 2.9 ± 0.44 and IQA by 3.4 ± 0.46 points.

<details>
<summary><b>Training stages and loss aggregation</b></summary>

<br>

| Variant | IAA Avg. SRCC | IQA Avg. SRCC |
|---|:---:|:---:|
| TATAR | 53.5 | 78.3 |
| SFT-only (no Stage 2)† | 51.2 (−2.3) | 76.7 (−1.6) |
| RL-only (no cold start)† | 50.3 (−3.2) | 76.4 (−1.9) |

† Single seed. Without the cold start, many aesthetic completions carry no valid score, so part of RL-only's IAA gap reflects missing scores rather than less accurate ones.

| Aggregation | IAA loss weight | Aesthetic token share | IAA Avg. | IQA Avg. |
|---|:---:|:---:|:---:|:---:|
| Global token mean | = token share | 0.768 | 53.1 | 77.0 |
| Task-balanced, $`\beta=0.50`$ (default) | 0.50 | 0.765 | 53.5 | 78.3 |
| Task-balanced, $`\beta=0.60`$ | 0.60 | 0.766 | 53.8 | 77.7 |
| Task-balanced, $`\beta=0.75`$ | 0.75 | 0.770 | 54.0 | 76.6 |

Three-seed means. Against $`\beta=0.50`$, the global token mean costs the shorter-response task most (IQA −1.3 ± 0.19, IAA −0.4 ± 0.40). Equal weights give the best balance among the settings tested; the difference from $`\beta=0.60`$ is within seed variation.

</details>

### Inference Cost

Mean output length in tokens over the four benchmarks of each task (greedy decoding, up to 2,048 tokens).

| Model | IQA | IAA | Overall |
|---|:---:|:---:|:---:|
| UniPercept | 18 | 20 | 19 |
| VisualScore | 171 | 176 | 174 |
| VisualQuality-R1 | 218 | 224 | 221 |
| **TATAR (ours)** | 63 | 205 | 134 |

By reasoning deliberatively only for aesthetics, TATAR generates 23% fewer output tokens than VisualScore on a workload with equal numbers of quality and aesthetic images, and saves more on quality-heavy workloads.

---

## 🚀 Getting Started

🚧 **Coming soon.** Release checklist:

- [ ] Environment setup instructions
- [ ] Dataset preprocessing scripts
- [ ] QACoT-Score cold-start corpus
- [ ] Training commands for Stage 1 and Stage 2
- [ ] Pretrained TATAR checkpoints
- [ ] Evaluation commands for the eight benchmarks
- [ ] Inference demo

### Notes on Usage

- **Name the task in the prompt.** Both tasks share one question template that names the target attribute. TATAR takes the task from the prompt and does not infer it from the image.
- **Output format.** Each response contains `<think>…</think><answer>score</answer>`, with the score on a 1–100 scale.
- **Aesthetic scores are ordinal.** The ranking reward depends only on score differences, so it is invariant to constant shifts but not to rescaling. Use IAA scores to rank images or compute SRCC. If absolute values on the annotation scale are needed, fit a monotone mapping on a small labeled set; this leaves SRCC unchanged.
- **Generated images.** TATAR's accuracy on AI-generated images has not been verified yet.

<details>
<summary><b>Example outputs</b> (from the case study in the paper)</summary>

<br>

IQA, concise mode (KonIQ-10k, ground truth 16.9):

```text
<think>The image exhibits severe overexposure in the central region, causing significant loss of detail in the bright fire area. Strong color cast and low contrast further degrade the visual fidelity, resulting in a lack of structural information in the background figures.</think>
<answer>15.96</answer>
```

IAA, deliberative mode (ArtiMuse-10K, ground truth 31.4; abbreviated with [...]):

```text
<think>The vibrant container ship with bold colors creates an engaging yet somewhat disjointed experience. The composition is balanced but lacks depth, yielding a flat appearance from the absence of shadows and perspective. The gestalt is striking but chaotic, and technical execution shows significant distortion and pixelation. [...] Its emotional resonance is limited, and the concept remains familiar, lacking significant originality, balancing its strengths in visual impact against technical limitations.</think>
<answer>31.50</answer>
```

</details>

---

## 📦 Datasets

| Task | Training | In-domain test | Cross-domain test |
|---|---|---|---|
| IQA | KonIQ-10k (7,058 images, split of DeQA-Score) | KonIQ-10k | SPAQ, KADID-10k, PIPAL |
| IAA | ArtiMuse-10K (8,998 images, split of ArtiMuse) | ArtiMuse-10K | AVA, TAD66K, Flickr-AES |

- Stage 1 uses the filtered QACoT-Score corpus; Stage 2 uses the full training splits. All scores are mapped linearly to [1, 100].
- KADID-10k and PIPAL are evaluated on their full sets without reference images, AVA on its standard 19,930-image test set, and Flickr-AES with the mean score across raters.
- Metrics: SRCC and PLCC (×100), plus the four-benchmark mean per task.
- Ground-truth files are stored under `benchmark/VR/`.

---

## ⚙️ Training Configuration

Paper defaults:

| Item | Setting |
|---|---|
| Backbone | Qwen2.5-VL-7B-Instruct, full fine-tuning in both stages; images capped at 1,572,864 pixels |
| Stage 1 | 1 epoch, AdamW, learning rate $`10^{-5}`$ |
| Stage 2 | GRPO for 2 epochs, $`G=8`$ completions per prompt, temperature 1.0, at most 1,024 new tokens, clip $`\epsilon=0.2`$, $`\lambda_{\mathrm{KL}}=10^{-3}`$, AdamW with learning rate decayed linearly from $`10^{-6}`$ to $`10^{-9}`$ |
| Batching | 8 prompts per generation batch (4 IQA + 4 IAA, 64 completions); microbatches of 8 completions per GPU on 8 × NVIDIA A800, 2 accumulation steps (128 completions per update) |
| Gaussian reward | $`\tau_{\mathrm{tol}}=8.75`$, $`\eta=0.5`$, $`c=1`$ (floor $`\omega_0=7.4`$) |
| Ranking reward | $`\tau_a=15.0`$ (standard deviation of ArtiMuse-10K training scores), $`\varepsilon_p=10^{-3}`$, threshold $`10^{-6}`$ on $`\sum_j w^{ij}`$ |
| Aggregation | $`\alpha=\beta=0.5`$ |
| Seeds | TATAR and the controlled variants use three seeds, matched across configurations (a seed fixes the data order and rollout sampling) |
| Compute | about 240 GPU hours per run |

---

## 📂 Repository Structure

```
.
├── assets/                     # Figures used in this README
├── benchmark/VR/               # Ground-truth benchmark files
│
├── src/
│   ├── eval/                   # Evaluation entry scripts
│   │   ├── results/
│   │   ├── eval_uni_iqa_iaa.py # Unified IQA + IAA evaluation
│   │   ├── eval_vqa.py         # VQA evaluation
│   │   ├── eval_vr.py          # VR evaluation
│   │   └── *.sh
│   │
│   └── internvl/               # Backbone utilities
│
├── open-r1-multimodal/         # Main training project
│   ├── configs/
│   ├── data/
│   ├── data_config/
│   ├── local_scripts/
│   ├── src/
│   │   ├── openr1/             # Stage 2: task-aware GRPO + task-balanced aggregation
│   │   └── sft/                # Stage 1: two-mode cold start (SFT on QACoT-Score)
│   ├── wandb/
│   ├── run_iqa_iaa.sh
│   ├── run_qinsight_*.sh
│   ├── setup.cfg
│   └── setup.py
```

---

## 📄 Citation

The BibTeX entry will be added once the paper is publicly available.

---

## 🙏 Acknowledgements

The training code builds on [open-r1-multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal). We thank the authors of Qwen2.5-VL, KonIQ-10k, ArtiMuse-10K, and the evaluation benchmarks for making their models and data available.

---

## 📬 Contact

For questions or issues, please open an issue in this repository.

---

## 📝 License

To be released.
