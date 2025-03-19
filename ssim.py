# import numpy as np
# from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

# def normalize_to_target_range(targets, predictions):
#     """ Normalize predictions to match the range of the targets (-1 to 1) """
#     min_t, max_t = targets.min(), targets.max()
#     min_p, max_p = predictions.min(), predictions.max()
    
#     # Avoid division by zero
#     if max_p - min_p == 0:
#         return predictions
    
#     # Rescale predictions to match target range
#     predictions = (predictions - min_p) / (max_p - min_p)  # Scale to [0,1]
#     predictions = predictions * (max_t - min_t) + min_t  # Scale to [-1,1] (same as targets)
    
#     return predictions

# def compute_psnr(targets, predictions):
#     predictions = normalize_to_target_range(targets, predictions)  # Ensure it's normalized
#     return psnr(targets, predictions, data_range=2)

# def compute_ssim(targets, predictions):
#     predictions = normalize_to_target_range(targets, predictions)  # Ensure it's normalized
#     return ssim(targets, predictions, data_range=2)

# # Define file paths
# output_dir = "Outputs_diff/hicedrn_l2_Human1_deno_0.1_cond"
# target_file = f"{output_dir}/target.npy"
# noisy_file = f"{output_dir}/noisy.npy"
# predict_file = f"{output_dir}/predict.npy"

# # Load the data
# target = np.load(target_file)
# noisy = np.load(noisy_file)
# predict = np.load(predict_file)

# # 🔹 **Normalize predictions BEFORE computing metrics**
# predict = normalize_to_target_range(target, predict)

# # Print shape to verify data structure
# print(f"Target shape: {target.shape}")
# print(f"Noisy shape: {noisy.shape}")
# print(f"Predict shape: {predict.shape}")

# # Ensure data is 4D (batch, channel, height, width)
# if target.ndim != 4 or predict.ndim != 4 or noisy.ndim != 4:
#     raise ValueError("Expected 4D arrays (batch, channel, height, width)")

# # Compute SSIM, PSNR, and SNR for each sample
# ssim_noisy_target = []
# psnr_noisy_target = []
# ssim_predict_target = []
# psnr_predict_target = []

# for i in range(target.shape[0]):  # Loop through each sample
#     target_img = target[i, 0]
#     noisy_img = noisy[i, 0]
#     predict_img = predict[i, 0]

#     # Compute SSIM and PSNR
#     ssim_noisy_target.append(compute_ssim(target_img, noisy_img))
#     psnr_noisy_target.append(compute_psnr(target_img, noisy_img))
#     ssim_predict_target.append(compute_ssim(target_img, predict_img))
#     psnr_predict_target.append(compute_psnr(target_img, predict_img))

#     print(f"Sample {i}: Noisy vs Target (SSIM={ssim_noisy_target[-1]:.4f}, PSNR={psnr_noisy_target[-1]:.2f} dB) | "
#           f"Predicted vs Target (SSIM={ssim_predict_target[-1]:.4f}, PSNR={psnr_predict_target[-1]:.2f} dB)")

# # Compute averages
# avg_ssim_nt = np.mean(ssim_noisy_target)
# avg_psnr_nt = np.mean(psnr_noisy_target)
# avg_ssim_pt = np.mean(ssim_predict_target)
# avg_psnr_pt = np.mean(psnr_predict_target)

# # Print results
# print(f"\nAverage SSIM (Noisy vs Target): {avg_ssim_nt:.4f}")
# print(f"Average PSNR  (Noisy vs Target): {avg_psnr_nt:.2f} dB")
# print(f"\nAverage SSIM (Predicted vs Target): {avg_ssim_pt:.4f}")
# print(f"Average PSNR  (Predicted vs Target): {avg_psnr_pt:.2f} dB")

# print(f"\n🔹 Normalized Predictions Range:")
# print(f"Target min: {target.min()}, max: {target.max()}")
# print(f"Predicted min: {predict.min()}, max: {predict.max()}")  
# print(f"MSE: {np.mean((target - predict) ** 2)}")

# # Save results to a file
# result_file = f"{output_dir}/ssim_psnr_snr_results.txt"
# with open(result_file, "w") as f:
#     f.write(f"Average SSIM (Noisy vs Target): {avg_ssim_nt:.4f}\n")
#     f.write(f"Average PSNR  (Noisy vs Target): {avg_psnr_nt:.2f} dB\n")
#     f.write(f"Average SSIM (Predicted vs Target): {avg_ssim_pt:.4f}\n")
#     f.write(f"Average PSNR  (Predicted vs Target): {avg_psnr_pt:.2f} dB\n")

# print(f"\nResults saved to: {result_file}")


import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

def normalize_to_target_range(targets, predictions):
    """ Normalize predictions to match the range of the targets (-1 to 1) """
    min_t, max_t = targets.min(), targets.max()
    min_p, max_p = predictions.min(), predictions.max()

    if max_p == min_p:
        print("⚠️ Warning: Predictions have no variation! Skipping normalization.")
        return predictions

    # Rescale predictions to match target range
    predictions = (predictions - min_p) / (max_p - min_p)  # Scale to [0,1]
    predictions = predictions * (max_t - min_t) + min_t  # Scale to [-1,1]
    
    return predictions
    
def compute_psnr(targets, predictions):
    predictions = normalize_to_target_range(targets, predictions)
    mse = np.mean((targets - predictions) ** 2)

    if mse == 0:
        print("⚠️ Warning: MSE is 0! Targets and Predictions might be identical.")
        return float('inf')  # Prevents PSNR calculation errors
    
    return psnr(targets, predictions, data_range=2)

def compute_ssim(targets, predictions):
    predictions = normalize_to_target_range(targets, predictions)
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

# Compute SSIM, PSNR, and MSE for each sample
ssim_noisy_target, psnr_noisy_target = [], []
ssim_predict_target, psnr_predict_target, mse_values = [], [], []

for i in range(target.shape[0]):  # Loop through each sample
    target_img = target[i, 0]
    noisy_img = noisy[i, 0]
    predict_img = predict[i, 0]

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

    # Compute MSE
    mse = np.mean((target_img - predict_img) ** 2)
    mse_values.append(mse)

    print(f"Sample {i}: MSE={mse:.6f} | "
          f"Noisy vs Target (SSIM={ssim_nt:.4f}, PSNR={psnr_nt:.2f} dB) | "
          f"Predicted vs Target (SSIM={ssim_pt:.4f}, PSNR={psnr_pt:.2f} dB)")

# Compute averages
avg_ssim_nt = np.mean(ssim_noisy_target)
avg_psnr_nt = np.mean(psnr_noisy_target)
avg_ssim_pt = np.mean(ssim_predict_target)
avg_psnr_pt = np.mean(psnr_predict_target)
avg_mse = np.mean(mse_values)

# Print results
print(f"\nAverage SSIM (Noisy vs Target): {avg_ssim_nt:.4f}")
print(f"Average PSNR  (Noisy vs Target): {avg_psnr_nt:.2f} dB")

print(f"\nAverage SSIM (Predicted vs Target): {avg_ssim_pt:.4f}")
print(f"Average PSNR  (Predicted vs Target): {avg_psnr_pt:.2f} dB")

print(f"\nTarget min: {target.min()}, max: {target.max()}")
print(f"Predicted min: {predict.min()}, max: {predict.max()}")
print(f"MSE: {avg_mse:.6f}")

# Save results to a file
result_file = f"{output_dir}/ssim_psnr_snr_results.txt"
with open(result_file, "w") as f:
    f.write(f"Average SSIM (Noisy vs Target): {avg_ssim_nt:.4f}\n")
    f.write(f"Average PSNR  (Noisy vs Target): {avg_psnr_nt:.2f} dB\n")
    f.write(f"Average SSIM (Predicted vs Target): {avg_ssim_pt:.4f}\n")
    f.write(f"Average PSNR  (Predicted vs Target): {avg_psnr_pt:.2f} dB\n")
    f.write(f"Average MSE: {avg_mse:.6f}\n")

print(f"\nResults saved to: {result_file}")
