# FatigueGuard: Flexible Modality Learning with Adaptive Mixture-of-Experts for Physiological Driver Fatigue Detection

<p align="center">
    <img src="./img/overview.png" alt="Overview" width="800"/>
</p>

## Abstract

Driver fatigue detection is important for improving road safety. Multimodal physiological signals provide complementary information for more accurate and objective fatigue assessment, but some modalities may be unavailable in real-world driving due to sensor issues. Existing methods that either model each modality independently or rely on completion modules struggle to balance flexible adaptation to available modalities with effective exploitation of cross-modal information.

To address these issues, we propose FatigueGuard, a flexible modality learning framework for driver fatigue detection under varying modality availability. Specifically, we design an adaptive mixture-of-experts module with lightweight modality-specific and shared experts. A gating mechanism dynamically weights and integrates their outputs, allowing the model to capture modality-specific characteristics and cross-modal shared information.

Furthermore, we develop a two-stage learning strategy consisting of multi-task pre-training and fatigue detection fine-tuning. During pre-training, three self-supervised tasks with dedicated attention masks, including cross-modal contrastive learning, modality matching, and modality generation, are designed to enhance modality-specific representation learning and cross-modal relationship modeling. During fine-tuning, FatigueGuard directly utilizes the available modalities and the shared expert for fatigue prediction, enabling flexible adaptation to varying modality availability while fully leveraging the cross-modal information learned during pre-training.

Experiments on three public datasets show that FatigueGuard outperforms state-of-the-art methods under different modality availability conditions, validating its effectiveness.


## Setup

```bash
git clone https://github.com/JingBai-bit/FatigueGuard.git
cd FatigueGuard

conda create -n fuse python=3.10 -y
conda activate fuse
pip install -r requirements.txt
```

## Run

Set the dataset name, input file, output directory, fold configuration, training hyperparameters, and GPU ID at the top of run.sh:

```bash
DATASET_NAME="DROZY"
INPUT_FILE="/path/to/data.npz"
OUTPUT_DIR="/path/to/experiment_results"

NUM_FOLDS=5
FOLDS="all"
MODALITIES="multi,eeg,eog"
MIXTURE_EXPERT_LAYERS=3
GPU_ID="0"
DEVICE="cuda"
```

Start the experiment with:

```bash
bash run.sh
```
