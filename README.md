# FatigueGuard: Flexible Modality Learning with Adaptive Mixture-of-Experts for Physiological Driver Fatigue Detection

<p align="center">
    <img src="./img/framework.png" alt="Overview" width="800"/>
</p>

## Abstract

Driver fatigue detection is important for improving road safety.
Multimodal physiological signals provide complementary information for more accurate and objective fatigue assessment, but some modalities may be unavailable in real-world driving due to sensor issues. 
Existing methods that either model each modality independently or rely on completion modules struggle to balance flexible adaptation to available modalities with effective exploitation of cross-modal information.
To address these issues, we propose FatigueGuard, a flexible modality learning framework for driver fatigue detection under varying modality availability. 
Specifically, we design an adaptive mixture-of-experts module with lightweight modality-specific and shared experts. A gating mechanism dynamically weights and integrates their outputs, allowing the model to capture modality-specific characteristics and cross-modal shared information.
Furthermore, we develop a two-stage learning strategy consisting of multi-task pre-training and fatigue detection fine-tuning. 
During pre-training, three self-supervised tasks with dedicated attention masks, including cross-modal contrastive learning, modality matching, and modality generation, are designed to enhance modality-specific representation learning and cross-modal relationship modeling.
During fine-tuning, FatigueGuard directly utilizes available modalities and the shared expert for fatigue prediction, enabling flexible adaptation to varying modality availability while fully leveraging cross-modal information learned during pre-training.
Experiments on three public datasets show that FatigueGuard outperforms state-of-the-art methods under different modality availability conditions, validating its effectiveness.

## Configuration

Before running experiments, edit `configs/config.py`:

- Set the pretrained vision-language checkpoint paths.
- Set the dataset roots for the datasets you want to use.
- Keep dataset split files under `lists/` or update the corresponding paths.

## Setup and Run

```bash
git clone https://github.com/JingBai-bit/FatigueGuard.git
cd FatigueGuard
pip install -r requirements.txt
sh train.sh
```



