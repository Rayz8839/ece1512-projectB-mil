# ECE1512 Project B: JPEG Restoration and DNN-Based Image Compression

**Author:** Syed Rayyan Albeez Ikramullah (MEng Aerospace Engineering, University of Toronto)

From JPEG artifact restoration (a residual CNN) to an RDVQ-style learned codec
(Jiang et al., CVPR 2026) with real bitstreams, evaluated against JPEG on the same images,
plus controlled ablations. Dataset: `data_1um_bicubic` from
[LeonYe/ADPPatches](https://huggingface.co/datasets/LeonYe/ADPPatches) (downloaded automatically by the notebooks).

## Headline results

| Part | Result |
|---|---|
| 2: Restoration | 29,507-parameter residual CNN; +2.17 to +2.34 dB PSNR and +0.046 to +0.068 SSIM over JPEG at Q in {10,15,20,25,30} on 1,576 test images, at identical bitrate |
| 3: RDVQ codec | K=256, D=16, lambda=1e-3, soft assignment for the rate term, hard indices stored; test PSNR 32.43 dB, SSIM 0.9242 (64x64 crops) |
| 4: Evaluation | 100 held-out 256x256 crops, real bitstreams: 1.035 bpp, 32.26 dB, SSIM 0.9276; **+1.97 dB / +0.060 SSIM** over interpolated JPEG at equal scan rate (+2.26 dB / +0.066 with headers) |
| 4: Ablations | rate multiplier lambda, codebook size K, quantization strategy, entropy coder (each with RD plot, reconstructions, limitations) |

## Repository layout

| Path | Contents |
|---|---|
| `report/ECE1512_ProjectB_Report.pdf` | Final report (literature review, Parts 2-4, reproducibility record) |
| `notebooks/ECE1512_ProjectB_Part2_JPEG_Restoration.ipynb` | Part 2: residual-CNN JPEG restoration (executed outputs) |
| `notebooks/ECE1512_ProjectB_Part3_RDVQ_Compression.ipynb` | Part 3: RDVQ reproduction, soft vs hard quantization (executed outputs) |
| `notebooks/ECE1512_ProjectB_Part4_Evaluation_Ablation.ipynb` | Part 4: JPEG comparison and four ablations (executed outputs) |
| `helpers/` | Instructor-provided helper modules (`data.py`, `model.py`, `engine.py`, `codec.py`, `arithmetic.py`, `entropy_context.py`, `comparison.py`, `metrics.py`, `checks.py`, `instructor/warm_start.py`), unchanged. The notebooks also write these files themselves, so they run stand-alone |
| `Part_2_Results/` | Checkpoint, history, test metrics, quality curves, qualitative examples |
| `Part_3_Results/` | Checkpoint, config, history, test metrics, temperature/soft-vs-hard figures, JPEG comparison |
| `Part_4_Results/` | `tables/` (CSV), `figures/` (RD plots, reconstruction examples), `runs/` (checkpoints of all 9 variants), `part4_reproducibility.json`, `limitations_evidence.md` |

## Reproducing

The notebooks were run on Google Colab (Tesla T4, PyTorch 2.14.1, TorchMetrics 1.9.0).

1. Open a notebook in Colab (Runtime > Change runtime type > GPU) and run top to bottom.
   The dataset is downloaded from Hugging Face automatically.
2. Run order: Part 2 (independent), Part 3, then Part 4 (re-trains the Part 3 main model and the variants).
3. Locally: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`, then
   `jupyter lab` and open a notebook from the repository root (so `dnn_compression_notebook/` is created next to it).

Seeds: 42 (Part 2), 1512 (Parts 3-4). Single seed per variant; see the limitations in the report (Section VIII-H).

Note: paths inside the saved `config.json` / `resources.json` files refer to the Colab file system (`/content/...`).

Each part's outputs live in their own top-level folder (`Part_2_Results/`, `Part_3_Results/`, `Part_4_Results/`), so they can be uploaded or cloned separately.

## Reproducibility checklist

Values are taken from report Table XX and the saved result files (`Part_4_Results/part4_reproducibility.json`).

- [x] **Code and data:** this repository; dataset `data_1um_bicubic` (ADPPatches, Hugging Face `LeonYe/ADPPatches`); notebooks for Parts 2-4 run on Google Colab (Tesla T4)
- [x] **Dataset splits:** predefined train / val / test = 12,616 / 1,576 / 1,576 images. Part 2: full train set, 500-image validation subset (seeded), 1,576 test images. Part 3: 1,024 training crops and 128 validation crops (64x64), 256 test crops. Part 4: 100 held-out images (entries 128-227 of the seeded, shuffled test list); the file names are saved in `part4_reproducibility.json`
- [x] **Image sizes:** all images are lossless PNG, 272x272. Part 2 trains on random 256x256 crops and evaluates on the 256x256 center crop; Part 3 uses 64x64 crops; Part 4 evaluates 256x256 center crops. The reference is always the uncompressed original
- [x] **JPEG settings:** Pillow 11.3.0, 4:4:4 chroma (`subsampling=0`), `optimize=False`. Part 2 trains with Q in {10, 20, 30, 40, 50} and tests at {10, 15, 20, 25, 30}; Part 4 uses Q in {1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 95}. JPEG bitrate = 8 x file bytes / (H x W); the scan-only rate excludes headers
- [x] **Random seeds:** 42 (Part 2); 1512 (Parts 3-4); one training seed per variant; cuDNN deterministic mode on, benchmark off
- [x] **Restoration model (Part 2):** residual CNN, five 3x3 convolutions (3 to 32, three 32 to 32, 32 to 3), 29,507 parameters, L1 loss, AdamW (lr 2e-4, weight decay 1e-4), cosine schedule, 20 epochs, batch size 16, best validation PSNR at epoch 19
- [x] **Codebook settings (Parts 3-4):** one level, K = 256 (ablated: 64, 128, 512), D = 16, downsampling factor 2, lambda = 1e-3 (ablated: 1e-4, 1e-2, 1e-1), codebook loss weight 1 with beta = 0.25, 80 epochs, Adam lr 2e-4, batch size 32, straight-through quantization, k-means warm start after 8 autoencoder epochs
- [x] **Rate accounting:** DNN payload bpp = 8 x compressed latent bytes / (H x W); DNN whole-bitstream bpp includes the header; JPEG file bpp includes headers and scan bpp does not. The shared checkpoint (network, codebook, context tables, 1,320 kB) is excluded from per-image rates. The learned-prior estimate is a training proxy and is never reported as a measured size
- [x] **Equal-rate comparison:** linear interpolation of the measured JPEG curve at the DNN's rate, with no extrapolation outside the measured JPEG range
- [x] **Index-decoding checks:** the decoded image equals the model reconstruction (max absolute difference 0.0, criterion below 1e-4) on 100 evaluation images; arithmetic-coder round trips verified on 12 streams (max difference 0.0)
- [x] **Environment:** PyTorch 2.14.1+cu130, TorchMetrics 1.9.0 (used to check the SSIM implementation), Tesla T4 GPU, helper modules unchanged from the course starter code
