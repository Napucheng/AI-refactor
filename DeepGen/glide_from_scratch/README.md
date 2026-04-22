# GLIDE From Scratch on Windows

This folder contains a compact, Windows-friendly, GLIDE-style text-to-image diffusion training pipeline built from scratch with PyTorch.

It includes:

- Hugging Face asset download helper
- Caption dataset loader
- Token-level text encoder
- Text-conditioned U-Net denoiser
- DDPM/DDIM sampling
- Classifier-free guidance

This is a research-friendly reimplementation of the full workflow, not a parameter-matched reproduction of OpenAI's original large-scale GLIDE release.

## 1. Environment

Recommended on Windows with Python 3.10 or 3.11:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r glide_from_scratch\requirements.txt
```

If you use CUDA, install a matching PyTorch build first from the official PyTorch instructions, then install the remaining packages.

## 2. Download Hugging Face Assets

The training code uses a CLIP tokenizer. Download it once:

```powershell
python -m glide_from_scratch.download_hf_assets --repo-id openai/clip-vit-base-patch32 --local-dir checkpoints\hf\clip-vit-base-patch32
```

You can also download any other repo into a local folder:

```powershell
python -m glide_from_scratch.download_hf_assets --repo-id <repo_id> --local-dir checkpoints\hf\<name>
```

## Quick Start Script

You can also use the one-click Windows launcher:

```powershell
.\start_glide.bat -Mode setup
```

Train:

```powershell
.\start_glide.bat -Mode train -DataRoot your_dataset -Metadata your_dataset\metadata.csv
```

Sample:

```powershell
.\start_glide.bat -Mode sample -Checkpoint checkpoints\glide_tiny\model_latest.pt -Prompt "a glass spaceship in a desert"
```

## 3. Dataset Format

Prepare a folder like:

```text
your_dataset/
  images/
    0001.png
    0002.png
  metadata.csv
```

`metadata.csv` must contain:

```csv
filename,caption
0001.png,a red sports car parked on a rainy street
0002.png,a watercolor painting of a small cat reading a book
```

## 4. Train

Example:

```powershell
python -m glide_from_scratch.train `
  --data-root your_dataset `
  --metadata your_dataset\metadata.csv `
  --tokenizer-path checkpoints\hf\clip-vit-base-patch32 `
  --output-dir checkpoints\glide_tiny `
  --image-size 64 `
  --batch-size 8 `
  --epochs 50 `
  --timesteps 1000 `
  --sample-prompts "a golden retriever wearing sunglasses" "a futuristic city at sunrise"
```

Important notes:

- Start at `64x64` on Windows unless you have a strong GPU.
- The default model is intentionally small enough to debug locally.
- Classifier-free guidance training is enabled through random caption dropout.

## 5. Sample

Generate images from a trained checkpoint:

```powershell
python -m glide_from_scratch.sample `
  --checkpoint checkpoints\glide_tiny\model_latest.pt `
  --tokenizer-path checkpoints\hf\clip-vit-base-patch32 `
  --prompt "a cozy cabin in snowy mountains at dusk" `
  --num-samples 4 `
  --guidance-scale 5.0 `
  --sampler ddim `
  --steps 100 `
  --output-dir outputs\glide_samples
```

## 6. Files

- `download_hf_assets.py`: download tokenizer or model repos from Hugging Face
- `dataset.py`: caption dataset and tokenizer batching
- `models.py`: text encoder and U-Net denoiser
- `diffusion.py`: noise schedule, training loss, DDPM/DDIM samplers
- `train.py`: end-to-end training script
- `sample.py`: prompt-based image generation script

## 7. Practical Expectations

If your goal is paper-faithful GLIDE reproduction, you would still need:

- much larger model sizes
- huge curated text-image data
- longer training
- likely a cascaded upsampler stage

This code is the right place to start if you want to understand and own the whole pipeline on a Windows machine.
