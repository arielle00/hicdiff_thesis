import numpy as np
import math
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import torch


def compute_psnr(targets, predictions):
    # targets = target.cpu().numpy() if isinstance(target, torch.Tensor) else target
    # predictions = predicted.cpu().numpy() if isinstance(predicted, torch.Tensor) else predicted

    return psnr(targets, predictions, data_range=2)

def compute_ssim(targets, predictions):
    # targets = target.cpu().numpy() if isinstance(target, torch.Tensor) else target
    # predictions = predicted.cpu().numpy() if isinstance(predicted, torch.Tensor) else predicted

    return ssim(targets, predictions, data_range=2)

# Define file paths
output_dir = "Outputs_diff/hicedrn_l2_Human1_deno_0.1_cond"
target_file = f"{output_dir}/target.npy"
noisy_file = f"{output_dir}/noisy.npy"
predict_file = f"{output_dir}/predict.npy"

# Load the data
target = np.load(target_file)
noisy = np.load(noisy_file)
predict = np.load(predict_file)

# Print shape to verify data structure
print(f"Target shape: {target.shape}")
print(f"Noisy shape: {noisy.shape}")
print(f"Predict shape: {predict.shape}")

# Ensure data is 4D (batch, channel, height, width)
if target.ndim != 4 or predict.ndim != 4 or noisy.ndim != 4:
    raise ValueError("Expected 4D arrays (batch, channel, height, width)")

# Compute SSIM, PSNR, and SNR for each sample
ssim_noisy_target = []
psnr_noisy_target = []

ssim_predict_target = []
psnr_predict_target = []

for i in range(target.shape[0]):  # Loop through each sample
    # Extract the first channel (assuming grayscale)
    target_img = target[i, 0]
    noisy_img = noisy[i, 0]
    predict_img = predict[i, 0]

    # Compute SNR (Noisy vs Target)
    # snr_nt = compute_snr(target_img, noisy_img)
    # snr_noisy_target.append(snr_nt)

    # Compute SSIM (Noisy vs Target)
    ssim_nt = compute_ssim(target_img, noisy_img)
    ssim_noisy_target.append(ssim_nt)

    # Compute PSNR (Noisy vs Target)
    psnr_nt = compute_psnr(target_img, noisy_img)
    psnr_noisy_target.append(psnr_nt)

    # Compute SSIM (Predicted vs Target)
    ssim_pt = compute_ssim(target_img, predict_img)
    ssim_predict_target.append(ssim_pt)

    # Compute PSNR (Predicted vs Target)
    psnr_pt = compute_psnr(target_img, predict_img)
    psnr_predict_target.append(psnr_pt)

    print(f"Sample {i}: Noisy vs Target (SSIM={ssim_nt:.4f}, PSNR={psnr_nt:.2f} dB, | "
          f"Predicted vs Target (SSIM={ssim_pt:.4f}, PSNR={psnr_pt:.2f} dB")

# Compute averages
avg_ssim_nt = np.mean(ssim_noisy_target)
avg_psnr_nt = np.mean(psnr_noisy_target)

avg_ssim_pt = np.mean(ssim_predict_target)
avg_psnr_pt = np.mean(psnr_predict_target)

# Print results
print(f"\nAverage SSIM (Noisy vs Target): {avg_ssim_nt:.4f}")
print(f"Average PSNR  (Noisy vs Target): {avg_psnr_nt:.2f} dB")

print(f"\nAverage SSIM (Predicted vs Target): {avg_ssim_pt:.4f}")
print(f"Average PSNR  (Predicted vs Target): {avg_psnr_pt:.2f} dB")

print(f"Target min: {target.min()}, max: {target.max()}")
print(f"Predicted min: {predict.min()}, max: {predict.max()}")
print(f"MSE: {np.mean((target - predict) ** 2)}")
print(f"Data Range: {target.max() - target.min()}")

# Save results to a file
result_file = f"{output_dir}/ssim_psnr_snr_results.txt"
with open(result_file, "w") as f:
    f.write(f"Average SSIM (Noisy vs Target): {avg_ssim_nt:.4f}\n")
    f.write(f"Average PSNR  (Noisy vs Target): {avg_psnr_nt:.2f} dB\n")
    f.write(f"Average SSIM (Predicted vs Target): {avg_ssim_pt:.4f}\n")
    f.write(f"Average PSNR  (Predicted vs Target): {avg_psnr_pt:.2f} dB\n")

print(f"\nResults saved to: {result_file}")
