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
| `results/part2/` | Checkpoint, history, test metrics, quality curves, qualitative examples |
| `results/part3/` | Checkpoint, config, history, test metrics, temperature/soft-vs-hard figures, JPEG comparison |
| `results/part4/` | `tables/` (CSV), `figures/` (RD plots, reconstruction examples), `runs/` (checkpoints of all 9 variants), `part4_reproducibility.json`, `limitations_evidence.md` |

## Reproducing

The notebooks were run on Google Colab (Tesla T4, PyTorch 2.14.1, TorchMetrics 1.9.0).

1. Open a notebook in Colab (Runtime > Change runtime type > GPU) and run top to bottom.
   The dataset is downloaded from Hugging Face automatically.
2. Run order: Part 2 (independent), Part 3, then Part 4 (re-trains the Part 3 main model and the variants).
3. Locally: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`, then
   `jupyter lab` and open a notebook from the repository root (so `dnn_compression_notebook/` is created next to it).

Seeds: 42 (Part 2), 1512 (Parts 3-4). Single seed per variant; see the limitations in the report (Section VIII-H).

Note: paths inside the saved `config.json` / `resources.json` files refer to the Colab file system (`/content/...`).
