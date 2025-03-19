import numpy as np
import matplotlib.pyplot as plt

# Load the files
target = np.load("Outputs_diff/hicedrn_l2_Human1_deno_0.1_cond/target.npy")
noisy = np.load("Outputs_diff/hicedrn_l2_Human1_deno_0.1_cond/noisy.npy")
predict = np.load("Outputs_diff/hicedrn_l2_Human1_deno_0.1_cond/predict.npy")

# Select a sample to display (e.g., the first sample)
idx = 0  # Change this to see different samples

# Create a figure
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot each matrix
axs[0].imshow(target[idx, 0, :, :], cmap="hot")
axs[0].set_title("Ground Truth (Target)")
axs[0].axis("off")

axs[1].imshow(noisy[idx, 0, :, :], cmap="hot")
axs[1].set_title("Noisy Input")
axs[1].axis("off")

axs[2].imshow(predict[idx, 0, :, :], cmap="hot")
axs[2].set_title("Predicted (Denoised)")
axs[2].axis("off")

# axs[1].imshow(target[idx,0,:,:] - predict[idx, 0, :, :], cmap="hot")
# axs[1].set_title("Target - Predicted")
# axs[1].axis("off")

# axs[2].imshow(target[idx,0,:,:] - noisy[idx, 0, :, :], cmap="hot")
# axs[2].set_title("Target - Noisy")
# axs[2].axis("off")



# Show the plots
plt.tight_layout()
plt.savefig("plot.png", dpi=300)
plt.show()
