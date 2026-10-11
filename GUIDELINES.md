# Engineering Protocols, Architecture & Workflow Guide

> Multimodal behavioral deception cue estimation system built on the **DOLOS** benchmark dataset.

## Table of Contents

1. [Project Overview & Operational Principles](#1-project-overview--operational-principles)
2. [Data Hygiene, Leakage Prevention & Modality EDA](#2-data-hygiene-leakage-prevention--modality-eda)
3. [Temporal Synchronization & The Global Clock (T)](#3-temporal-synchronization--the-global-clock-t)
4. [The Shared Tensor Interface Contract](#4-the-shared-tensor-interface-contract)
5. [Component Testing & Verification Gates](#5-component-testing--verification-gates)
6. [Repository Architecture](#6-repository-architecture)
7. [Git Workflow & Repository Rules](#7-git-workflow--repository-rules)

---

## 1. Project Overview & Operational Principles

This protocol governs the engineering workflow for the 5-person team developing the multimodal behavioral deception cue estimation system using the DOLOS benchmark dataset.

All code contributions must adhere to the:

- **Contract-driven interfaces** (Section 4)
- **Testing gates** (Section 5)
- **Branching standards** (Section 7)

---

## 2. Data Hygiene, Leakage Prevention & Modality EDA

Before extracting features or training models, each modality lead must develop an independent Exploratory Data Analysis (EDA) notebook for their specific stream.

### 2.1 Zero-Leakage Data Partitioning

1. **Split by Subject Identity Only.** The DOLOS dataset contains multiple segments per speaker. Splitting randomly by clip leaks personal facial morphology, baseline pitch, and recording acoustics into evaluation splits. Train, Validation, and Test partitions must be grouped strictly by Subject/Speaker ID **before** computing any global statistics.
2. **Train-Set Statistics Only.** Global normalization parameters (mean $\mu$, standard deviation $\sigma$, minimums, maximums) must be computed exclusively on the training partition. Save these parameters into `configs/` and apply them to validation and test sets without recomputation.
3. **EDA Isolation.** Exploratory analysis of feature distributions must run on training subjects only, to avoid biasing feature selection or threshold calibration.

### 2.2 Modality-Specific EDA Deliverables

> [!NOTE]
> The checks and deliverables below are **suggestions and starting points**. Each modality owner must independently research external literature, open-source repositories, and relevant benchmarks (e.g., Kaggle competitions, GitHub repositories, LibriSpeech, facial analysis baselines) to decide on the most fitting EDA strategies for our dataset.

#### Visual Stream: `notebooks/01_eda_visual.ipynb`

- Measure face detection dropout rates caused by rapid head turns, hand occlusions, or low contrast.
- Check bounding-box coordinate stability and inter-frame jitter.
- Verify whether frame rates are consistent across all raw video files.

#### Acoustic Stream: `notebooks/02_eda_acoustic.ipynb`

- Check audio sampling rates and channel configurations across clips.
- Evaluate Signal-to-Noise Ratio (SNR) distributions and inspect clipping or room reverberation.
- Map energy distributions between active speech frames and silent baseline windows.

#### Linguistic Stream: `notebooks/03_eda_linguistic.ipynb`

- Calculate token sequence length distributions ($L$) across segments to determine padding limits.
- Evaluate Whisper transcription consistency and timestamp stability across speech variations.

---

## 3. Temporal Synchronization & The Global Clock ($T$)

To prevent misalignment across heterogeneous modalities, all continuous inputs must share a unified physical time base.

> [!IMPORTANT]
> All numerical values below (such as $T = 90$, 3 seconds, 30 FPS, and `hop_length = 533`) are **illustrative suggestions** that demonstrate the alignment logic. They are **not finalized**. The team will agree on the final values for $T$, sampling rates, and sliding window lengths **before extraction begins**.

| Stream | 0.0 s | 1.0 s | 2.0 s | 3.0 s | Length (example) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Physical Time** | `0.0s` | `1.0s` | `2.0s` | `3.0s` | n/a |
| **Video Frames** | Frame 0 | … | … | Frame 89 | $T = 90$ |
| **Audio Frames** | STFT 0 | … | … | STFT 89 | $T = 90$ |
| **Text Tokens** | "I" (0.1 s) | "never" (1.2 s) | "did" (2.4 s) | n/a | $L = 3$ |

### 3.1 The Shared Time Unit ($T$)

The team establishes an agreed-upon frame count $T$ for every fixed temporal window. For example, an illustrative window of $T = 90$ steps covers 3 seconds at 30 FPS.

### 3.2 Acoustic Hop-Length Alignment

The acoustic pipeline computes the STFT / Mel-filterbank `hop_length` so that the number of acoustic frames matches the target $T$ directly:

$$
\text{hop\_length} = \frac{\text{Audio Sample Rate}}{\text{Target Video FPS}} \qquad \left(\text{e.g., } \frac{16000}{30} \approx 533 \text{ samples}\right)
$$

This ensures time step $t$ in audio corresponds directly to frame $t$ in video.

### 3.3 Linguistic Alignment ($L$ to $T$)

Transcripts arrive as token sequences of length $L$. Whisper provides word intervals $[t_{\text{start}}, t_{\text{end}}]$, which are translated into frame indices $[0, T-1]$. Cross-attention uses text tokens as Queries ($Q$) to attend over time-frame Keys ($K$) and Values ($V$).

---

## 4. The Shared Tensor Interface Contract

All extracted features must be exported as PyTorch tensors (`.pt`) or HDF5 files (`.h5`) adhering strictly to the shared tensor dictionary.

> [!WARNING]
> Modules generating non-compliant schemas **will not be merged**.

> [!NOTE]
> The tensor shapes, feature dimensions, and window lengths in this schema (such as $D_{\text{vis}} = 35$, $D_{\text{aud}} = 128$, $D_{\text{txt}} = 768$, and $T = 90$) are **illustrative suggestions** that define the interface structure. They are **not finalized**. Each modality lead will confirm the final dimensions once feature choices are locked.

### 4.1 Shared Tensor Schema

```python
# Shared Tensor Schema (dimensional values are illustrative examples)
batch = {
    # 1. Visual Stream: (Batch, Time_Steps, Visual_Features)
    # Shape: (B, T, D_vis) | Example: (B, 90, 35) Action Units + affect logits
    "visual_feats": torch.FloatTensor,

    # 2. Audio Stream: (Batch, Time_Steps, Audio_Features)
    # Shape: (B, T, D_aud) | Example: (B, 90, 128) Mel-filterbank bins
    "audio_feats": torch.FloatTensor,

    # 3. Linguistic Stream: (Batch, Token_Length, Text_Features)
    # Shape: (B, L, D_txt) | Example: (B, 15, 768) Pretrained text embeddings
    "text_feats": torch.FloatTensor,

    # 4. Token-to-Time Alignment Indices: (Batch, Token_Length, 2)
    # Maps token index l to [start_frame_idx, end_frame_idx] where 0 <= frame < T
    "alignment_indices": torch.LongTensor,

    # 5. Target Labels: (Batch,)
    # Binary classification: 0 (Truthful), 1 (Deceptive)
    "label": torch.LongTensor,
}
```

| Key | Shape | Dtype | Example |
|---|---|---|---|
| `visual_feats` | `(B, T, D_vis)` | `FloatTensor` | `(B, 90, 35)` Action Units + affect logits |
| `audio_feats` | `(B, T, D_aud)` | `FloatTensor` | `(B, 90, 128)` Mel-filterbank bins |
| `text_feats` | `(B, L, D_txt)` | `FloatTensor` | `(B, 15, 768)` Pretrained text embeddings |
| `alignment_indices` | `(B, L, 2)` | `LongTensor` | Token → `[start_frame, end_frame]`, `0 <= frame < T` |
| `label` | `(B,)` | `LongTensor` | `0` = Truthful, `1` = Deceptive |

### 4.2 Verification Assertions

Every data generation script must run these runtime assertions **before saving**:

```python
assert not torch.isnan(tensor).any(), "NaN values detected in feature extraction!"
assert not torch.isinf(tensor).any(), "Inf values detected in feature extraction!"
assert tensor.shape[1] == TARGET_T, f"Expected T={TARGET_T}, got {tensor.shape[1]}"
```

---

## 5. Component Testing & Verification Gates

Sub-pipelines must be verified in isolation before full-system integration.

### 5.1 Synthetic Data Mocking

Modeling leads must build the fusion and loss modules against a synthetic `DummyDataset` that generates random tensors (`torch.randn(B, T, D)`) matching the agreed contract schema.

### 5.2 Sub-Pipeline Unit Tests

Each extraction pipeline must include an automated verification script (`tests/test_<modality>.py`) confirming that:

- [ ] It processes **3 sample video/audio clips** end-to-end without unhandled exceptions.
- [ ] The generated outputs match the target tensor dimensions $(T, D)$ or $(L, D)$.

### 5.3 Pre-Training Overfit Sanity Check

1. Isolate a subset of **5 to 10 video clips** (balanced between truthful and deceptive).
2. Disable dropout, weight decay, and data augmentation.
3. Train the integrated pipeline for **50–100 epochs** using AdamW.

**Passing criteria:**

- [ ] Training loss drops below $0.05$
- [ ] Training accuracy reaches $100\%$
- [ ] Backpropagation produces non-zero gradient norms across all active encoders

---

## 6. Repository Architecture

The project follows a component-driven directory layout. Code is organized by **system responsibility**, not by individual member names.

```text
multimodal-deception-detector/
│
├── .gitignore               # Strict exclusion of data and binary artifacts
├── README.md                # System documentation and environment setup
├── Instructions.md          # This engineering protocol
├── environment.yml          # Locked Conda/Pip environment specifications
│
├── configs/                 # Declarative configuration files
│   ├── base_config.yaml     # Global settings: agreed T, FPS, Sample_Rate
│   └── train_fusion.yaml    # Training hyperparameters and loss weights
│
├── data/                    # Local storage only (ignored by Git)
│   ├── raw/                 # Uncompressed DOLOS MP4 and WAV files
│   │   └── .gitkeep
│   ├── processed/           # Extracted .pt / .h5 tensor files
│   │   └── .gitkeep
│   └── splits/              # Subject-isolated split definitions (tracked in Git)
│       ├── train_subjects.csv
│       ├── val_subjects.csv
│       └── test_subjects.csv
│
├── notebooks/               # Modality-specific EDA notebooks
│   ├── 01_eda_visual.ipynb
│   ├── 02_eda_acoustic.ipynb
│   └── 03_eda_linguistic.ipynb
│
├── src/                     # Core Python source package
│   ├── __init__.py
│   ├── common/              # Shared contracts and global constants
│   │   ├── __init__.py
│   │   ├── contracts.py     # Tensor shapes, type definitions, and assertions
│   │   └── constants.py     # Agreed system constants (T, D_vis, D_aud, D_txt)
│   │
│   ├── extractors/          # Modality feature extraction sub-pipelines
│   │   ├── visual/          # Facial localization, AUs, and dynamics
│   │   │   ├── __init__.py
│   │   │   └── extract_visual.py
│   │   ├── audio/           # Spectrogram and prosodic extractors
│   │   │   ├── __init__.py
│   │   │   └── extract_audio.py
│   │   └── text/            # Whisper transcription and semantic projection
│   │       ├── __init__.py
│   │       └── extract_text.py
│   │
│   ├── dataset/             # PyTorch dataset and batch collation
│   │   ├── __init__.py
│   │   ├── dummy_loader.py  # Mock generator for model prototyping
│   │   └── multimodal_loader.py
│   │
│   ├── models/              # Neural architecture and fusion blocks
│   │   ├── __init__.py
│   │   ├── projections.py   # Latent projection layers
│   │   ├── cross_attention.py
│   │   ├── conflict_heads.py # Level-1 physical incongruence module
│   │   └── fusion_net.py
│   │
│   └── evaluation/          # Metrics, loss functions, and logging
│       ├── __init__.py
│       ├── losses.py        # Joint classification and incongruence losses
│       └── metrics.py       # Macro-F1, AUC-ROC, and FPR tracking
│
└── tests/                   # Standalone unit tests
    ├── test_contracts.py
    ├── test_visual_pipeline.py
    ├── test_audio_pipeline.py
    ├── test_text_pipeline.py
    └── test_fusion_forward.py
```

---

## 7. Git Workflow & Repository Rules

### 7.1 Branching Strategy

| Branch | Purpose | Rules |
|---|---|---|
| `main` | Stable, deployable codebase | **Protected.** No direct commits or force-pushes. Merged only at major milestones via Pull Request. |
| `dev` | Integration / staging branch | All tested feature branches merge here. |
| `feat/<track-name>` | Individual working branches, created off `dev` | e.g. `feat/visual-au-extractor`, `feat/audio-spectrogram`, `feat/cross-attention` |

**Pull Request process:**

1. Once a feature passes local unit tests, open a Pull Request targeting `dev`.
2. Rebase or merge the latest `dev` into your branch locally to resolve conflicts **before** submitting the PR.
3. Merges into `dev` must use **Squash and Merge** to maintain a clean history.

### 7.2 Git Ignore Policy

> [!CAUTION]
> Never commit media files, extracted features, or checkpoints.

The root `.gitignore` must contain:

```gitignore
# Byte-compiled files and virtual environments
__pycache__/
*.py[cod]
*$py.class
.ipynb_checkpoints/
.env
venv/
env/

# Data isolation
data/raw/*
!data/raw/.gitkeep

data/processed/*
!data/processed/.gitkeep

# Binary, media, and tensor formats
*.mp4
*.avi
*.mov
*.wav
*.mp3
*.pt
*.pth
*.h5
*.hdf5
*.npy
*.npz

# Logs and checkpoints
checkpoints/
wandb/
runs/
*.log
```

### 7.3 Data Path Configuration

Do **not** hardcode machine-specific absolute file paths in any Python script. Reference paths dynamically relative to the project root, configure them via `configs/base_config.yaml`, or mount external storage drives to `data/raw` using symbolic links.

**Windows (PowerShell as Admin):**

```powershell
New-Item -ItemType SymbolicLink -Path "data/raw" -Target "D:\DOLOS\raw"
```

**Linux / macOS:**

```bash
ln -s /mnt/storage/DOLOS/raw data/raw
```
