# Consistency Models for Accelerated Diffusion

This repository contains the final project of submitted during the participation in a CV Week 2024 organized by Yandex Research and SDA. It contains a comprehensive implementation of **Consistency Models** and their variants for accelerating multi-step diffusion models. The notebook demonstrates how to distill a teacher diffusion model (Stable Diffusion 1.5) into a student model capable of generating high-quality images in significantly fewer steps. 

## Key Features

1. **Diffusion Models Overview**:
   - Explanation of forward and reverse diffusion processes.
   - Introduction to DDIM solvers for efficient sampling.

2. **Consistency Models**:
   - Learn a "consistency function" to predict clean data directly from noisy data in one step.
   - Train models with self-consistency and boundary conditions.

   <img src="https://storage.yandexcloud.net/yandex-research/cvweek-cd-task-images/cd-idea.jpg" width="600" alt="Consistency Models Idea"/>

3. **Consistency Distillation**:
   - Use a pre-trained teacher model to guide the training of the student model.
   - Incorporate Classifier-Free Guidance (CFG) for improved image quality.

4. **Multi-Boundary Consistency Distillation**:
   - Divide the diffusion trajectory into multiple segments for easier training.
   - Achieve deterministic sampling with improved quality.

   <img src="https://storage.yandexcloud.net/yandex-research/cvweek-cd-task-images/multi-cd-idea.jpg" width="600" alt="Multi-Boundary Consistency Distillation"/>

5. **Efficient Training Techniques**:
   - Gradient checkpointing to save memory.
   - LoRA (Low-Rank Adapters) for parameter-efficient fine-tuning.
   - Mixed-precision training for speed and memory optimization.

   <img src="https://storage.yandexcloud.net/yandex-research/cvweek-cd-task-images/lora-idea.jpg" width="300" alt="LoRA Adapters"/>

6. **Image Generation**:
   - Generate high-quality images in as few as 4 steps.
   - Support for CFG during sampling.

## Notebook Structure ([Colab Version](https://colab.research.google.com/drive/1XjwVbfIT9WxtLEyNwsDHptqImjO4b3Ex#scrollTo=c18dffaf))

1. **Introduction**:
   - Overview of diffusion models and consistency models.
   - Theoretical background and mathematical formulations.

2. **Teacher Model (Stable Diffusion 1.5)**:
   - Load and configure the teacher model for image generation.

3. **Dataset Preparation**:
   - Use a subset of the COCO dataset for training.
   - Preprocess images and create data loaders.

4. **Consistency Training (CT)**:
   - Train a standalone consistency model without a teacher.

5. **Consistency Distillation (CD)**:
   - Distill the teacher model into a student model with CFG.

6. **Multi-Boundary Consistency Distillation**:
   - Train models with multiple boundary points for better performance.

7. **Sampling**:
   - Generate images using the trained models.
   - Compare results across different training methods.

   <img src="https://storage.yandexcloud.net/yandex-research/cvweek-cd-task-images/cd-sampling-idea.jpg" width="600" alt="Sampling Process"/>

8. **Model Upload**:
   - Save and upload the trained models to Hugging Face for sharing.

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Yandex-ConsistencyModels.git
   cd Yandex-ConsistencyModels
   ```

2. Open the notebook `ConsistencyModels.ipynb` in Jupyter or Colab.

3. Follow the step-by-step instructions to train and evaluate consistency models.

4. Use the provided sampling functions to generate images from text prompts.

## Results

- **4-Step Image Generation**: The distilled models can generate high-quality images in just 4 steps, compared to 50 steps for the original Stable Diffusion model.
- **Improved Efficiency**: Multi-boundary consistency distillation achieves better results with deterministic sampling.


## References

- [Consistency Models](https://arxiv.org/pdf/2303.01469)
- [Improved Techniques for Training Consistency Models](https://arxiv.org/pdf/2310.14189)
- [Stable Diffusion](https://huggingface.co/stable-diffusion-v1-5)

## Acknowledgments

This project is based on research and resources provided by Yandex Research within the CV Week intensice course on Diffusion Models. You may find the lectures and tutorials [here](https://shad.yandex.ru/cvweek) (in Russian Language).
