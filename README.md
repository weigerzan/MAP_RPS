# Stage-wise Distortion–Perception Traversal in Zero-shot Inverse Problems with Diffusion Models

This repository provides the implementation of our stage-wise framework for zero-shot inverse problems with diffusion models.

## Acknowledgements

This codebase is largely based on the following repositories:

- [DDRM](https://github.com/bahjat-kawar/ddrm)
- [DDNM](https://github.com/wyhuai/DDNM)
- [DPS](https://github.com/DPS2022/diffusion-posterior-sampling)
- [ProjDiff](https://github.com/weigerzan/ProjDiff)

We sincerely thank the authors for releasing their code.

## Environment Setup

You can set up the environment using the provided `environment.yml`. The environment is mostly consistent with that of [DDRM](https://github.com/bahjat-kawar/ddrm).

```bash
conda env create -f environment.yml
conda activate map_rps
```

For text-to-image experiments based on Stable Diffusion, please additionally install `diffusers` and `transformers`:

```bash
pip install diffusers==0.35.2
pip install transformers
```

For nonlinear deblurring, you may also need to install `bkse` following the instructions in the [DPS repository](https://github.com/DPS2022/diffusion-posterior-sampling).

## Datasets and Pretrained Models

### Datasets

We use 100 test images from FFHQ and 100 test images from MS-COCO.

#### FFHQ

The FFHQ dataset can be downloaded from the [official FFHQ repository](https://github.com/NVlabs/ffhq-dataset).

In our experiments, we randomly sample 100 images from the `00000` folder. We also provide our sampled subset at:

```text
TODO: add FFHQ subset link
```

Please place the FFHQ test images under:

```text
exp/datasets/ffhq/0
```

#### MS-COCO

We provide the MS-COCO test subset used in our experiments at:

```text
TODO: add MS-COCO subset link
```

Please place the MS-COCO test images under:

```text
exp/datasets/coco
```

The expected dataset structure is:

```text
exp
└── datasets
    ├── ffhq
    │   └── 0
    │       ├── 0000.png
    │       ├── 0001.png
    │       └── ...
    └── coco
        ├── metadata.csv
        └── images
            ├── 0000.png
            ├── 0001.png
            └── ...
```

### Pretrained Models

#### FFHQ pixel-space diffusion model

For the FFHQ pixel-space diffusion model, please follow the instructions in the [DPS repository](https://github.com/DPS2022/diffusion-posterior-sampling) to download the pretrained checkpoint. Alternatively, the checkpoint can be downloaded from:

```text
https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh
```

Please link or place the model under:

```text
pretrained/ffhq_pixel
```

#### FFHQ latent diffusion model

For the FFHQ latent diffusion model, please download the pretrained models from the [CompVis latent-diffusion repository](https://github.com/CompVis/latent-diffusion), and place them under:

```text
pretrained/ffhq_ldm
```

#### Stable Diffusion v1.5

For Stable Diffusion v1.5, the model will be automatically downloaded from Hugging Face through `diffusers`.

The expected pretrained model structure is approximately:

```text
pretrained
├── ffhq_pixel
│   └── ffhq_10m.pt
└── ffhq_ldm
    ├── first_stage_models
    │   └── ...
    └── ldm
        └── ...
```

## Basic Usage

Run the following command to perform restoration under a specified degradation setting:

```bash
python main.py \
  --config <PATH_TO_CONFIG> \
  --timesteps <TIMESTEPS> \
  --deg <DEGRADATION_TYPE> \
  --sigma_0 <NOISE_LEVEL> \
  --lr <LR> \
  --lam <LAMBDA> \
  --optimize_iters <NUM_OPTIMIZATION_ITERS> \
  --vae_lr <VAE_LR> \
  --w_prior <PRIOR_WEIGHT> \
  --eta_min <ETA_MIN> \
  --noise_t <NOISE_TIMESTEP> \
  --renoise_t <RENOISE_TIMESTEP> \
  --algo <ALGORITHM> \
  --ps_method <POSTERIOR_SAMPLING_METHOD> \
  --ni
```

### Arguments

| Argument | Description | Choices / Suggested values |
| --- | --- | --- |
| `--config` | Target config file. | `ffhq_ldm/ffhq-ldm-vq-4.yaml`, `ffhq_pixel/ffhq.yaml`, `sd15/sd15.yaml` |
| `--timesteps` | Number of diffusion timesteps used by the pretrained diffusion model. | Usually `1000` |
| `--deg` | Degradation operator for the inverse problem. | `denoise`, `sr4`, `inp`, `cs2`, `deblur_aniso`, `hdr`, `deblur_nonlinear` |
| `--sigma_0` | Standard deviation of the observation noise. | `>= 0`, task-dependent |
| `--lr` | Step size / learning rate for posterior sampling methods. | Tuned task by task |
| `--lam` | Weight for PSLD. | Task-dependent; only used for PSLD |
| `--optimize_iters` | Number of MAP optimization iterations in Stage 1. | Positive integer |
| `--vae_lr` | Initial step size for MAP optimization in Stage 1. | Usually `0.5`–`2.0` |
| `--w_prior` | Weight of the prior term. | `>= 0` |
| `--eta_min` | Minimum learning rate for learning-rate decay. | Usually `1e-5`–`1e-2` |
| `--noise_t` | Timestep used for calculating the prior loss. | `50` for latent-space models; `10` for pixel-space models |
| `--renoise_t` | Timestep used for renoising. | Integer in `[0, timesteps]` |
| `--algo` | Algorithm to run. | `map_rps`, `lmap_rps` |
| `--ps_method` | Posterior sampling method. | e.g., `dps`, `psld` |
| `--ni` | Run without interaction and overwrite the target folder if needed. | Flag |

## Reproducing Our Experiments

We provide all scripts used in our experiments under the `scripts/` directory.

```bash
bash scripts/<SCRIPT_NAME>.sh
```

Please check the corresponding script for the task-specific configuration, degradation type, and hyperparameters.

## Citation

If you find this repository useful, please consider citing our paper:

```bibtex
@inproceedings{zhang2026stagewise,
  title={Stage-wise Distortion--Perception Traversal in Zero-shot Inverse Problems with Diffusion Models},
  author={Zhang, Jiawei and Liu, Ziyuan and Yan, Leon and Xiao, Zhenyu and Gu, Yuantao},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```