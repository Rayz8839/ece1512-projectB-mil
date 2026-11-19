# Limitations: evidence from this run

- Seeds: one training seed (1512) per variant; differences between variants are single-run measurements without a seed-to-seed confidence interval.
- Training budget: 1024 training crops of 64x64 (from 12616 training images), 80 epochs; evaluation uses 256x256 crops, so training and evaluation crop sizes differ; entropy-coder context tables are fitted on 64x64 index maps.
- Evaluation set: 100 held-out images (center 256x256 crops, not whole images when the native image is larger); main PSNR std across images = 2.06 dB.
- Entropy model: context-free learned prior (estimated 1.831 bpp) versus measured payload 1.035 bpp (+77.0%); coders actually selected: {'arithmetic-up': np.int64(72), 'arithmetic-left': np.int64(18), 'zlib': np.int64(10)}. No conditional (autoregressive/Transformer) entropy model as in the paper.
- Rate range: DNN variants span 0.400-1.185 bpp (payload); measured JPEG scan range is 0.277-5.478 bpp; 9/9 variants lie inside the measured JPEG range (equal-rate comparison exists only there).
- Container overhead: DNN header averages 164 B/image vs JPEG headers 625 B/image (Q50).
- Shared model (network + codebook + context tables) = 1320 kB, excluded from per-image rates; JPEG needs no such shared file.
- Speed: Python arithmetic coder, mean encode 343 ms and decode 259 ms per 256x256 image (CPU coder + GPU network).
- Distortion is MSE plus the codebook term only (no LPIPS/GAN); quality is judged with PSNR and SSIM only.
- Single-scale latent (downsampling 2), codebook 16-256 codewords of dimension 16, about 0.2 M parameters: far smaller than the paper's multi-scale 4096x32 system.