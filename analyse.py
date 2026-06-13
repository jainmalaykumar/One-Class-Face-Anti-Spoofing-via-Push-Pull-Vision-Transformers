import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
import cv2
from model import SCMNet

# --- CONFIGURATION ---
best_model_path = "checkpoints_v7/oulu/SCMNet/model_count_057.pth" # UPDATE THIS 5, 18, 47, 33, 8, 54, 10
num_samples_tsne = 300  # Number of samples per class for t-SNE
num_samples_heat = 4    # Number of samples to plot heatmaps for

# --- DEVICE SETUP ---
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def NormalizeData_torch(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

print("Loading Data...")
test_live_data = np.load("datasets/oulu_npu/test_live.npy")
test_spoof_data = np.load("datasets/oulu_npu/test_spoof.npy")

# Subsample for visualization to keep it fast and readable
live_subset = test_live_data[:num_samples_tsne]
spoof_subset = test_spoof_data[:num_samples_tsne]

test_images = np.vstack([live_subset, spoof_subset])
labels = np.hstack([np.ones(len(live_subset)), np.zeros(len(spoof_subset))])

test_images_tensor = torch.tensor(test_images).permute(0, 3, 1, 2).float()
test_images_tensor = NormalizeData_torch(test_images_tensor).to(device)

print("Loading Model...")
model = SCMNet().to(device)
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

features_list = []
scms_list = []

print("Extracting Latent Spaces and SCMs...")
with torch.no_grad():
    # Process in chunks to avoid memory overload
    chunk_size = 30
    for i in range(0, len(test_images_tensor), chunk_size):
        batch = test_images_tensor[i:i+chunk_size]
        
        # 1. Get raw features from the Swin Extractor and apply GAP for t-SNE
        raw_features = F.normalize(model.F(batch))
        gap_features = raw_features.mean(dim=[2, 3]) # Shape: [Batch, 128]
        features_list.append(gap_features.cpu().numpy())
        
        # 2. Get the Spoof Cue Maps (Heatmaps)
        spoof_cues = model(batch) # Shape: [Batch, 1, 32, 32]
        scms_list.append(spoof_cues.cpu().numpy())

all_features = np.vstack(features_list)
all_scms = np.vstack(scms_list)

# ==========================================
# PLOT 1: LATENT SPACE (t-SNE)
# ==========================================
print("Computing t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_results = tsne.fit_transform(all_features)

plt.figure(figsize=(10, 8))
plt.scatter(tsne_results[labels == 1, 0], tsne_results[labels == 1, 1], 
            c='green', label='Live (Compact Target)', alpha=0.6, edgecolors='w')
plt.scatter(tsne_results[labels == 0, 0], tsne_results[labels == 0, 1], 
            c='red', label='Spoof (OOD)', alpha=0.6, edgecolors='w')
plt.title(f"t-SNE of Swin-Transformer Latent Space\n(Ideally, Green is a tight cluster, Red is scattered)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("latent_space_tsne8.png", dpi=300)
plt.show()

# ==========================================
# PLOT 2: SCM HEATMAP VISUALIZATION
# ==========================================
print("Generating Heatmaps...")

def overlay_heatmap(img_tensor, scm_tensor):
    # Convert image back to HWC numpy for plotting [0, 1]
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    
    # SCM is [1, 32, 32], squeeze it to [32, 32]
    scm_np = scm_tensor[0]
    
    # Normalize SCM to [0, 255] for the colormap
    # We clip negative values (if any) to 0
    scm_np = np.clip(scm_np, 0, None) 
    if np.max(scm_np) > 0:
        scm_norm = (scm_np / np.max(scm_np) * 255).astype(np.uint8)
    else:
        scm_norm = np.zeros_like(scm_np).astype(np.uint8)
        
    # Resize SCM to match image dimensions (256x256)
    scm_resized = cv2.resize(scm_norm, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # Apply JET colormap (Blue = Low/Live, Red = High/Spoof)
    heatmap = cv2.applyColorMap(scm_resized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # Blend image and heatmap
    alpha = 0.5
    blended = (1.0 - alpha) * img_np + alpha * heatmap
    return img_np, heatmap, np.clip(blended, 0, 1)

# Select a few live and spoof samples
live_idx = np.where(labels == 1)[0][:num_samples_heat]
spoof_idx = np.where(labels == 0)[0][:num_samples_heat]
selected_indices = np.concatenate([live_idx, spoof_idx])

fig, axes = plt.subplots(nrows=len(selected_indices), ncols=3, figsize=(9, 3 * len(selected_indices)))
axes[0, 0].set_title("Original Image")
axes[0, 1].set_title("Spoof Cue Map (SCM)")
axes[0, 2].set_title("Overlay")

for i, idx in enumerate(selected_indices):
    img = test_images_tensor[idx]
    scm = all_scms[idx]
    
    orig, heat, blended = overlay_heatmap(img, scm)
    
    axes[i, 0].imshow(orig)
    axes[i, 0].axis('off')
    
    # Use vmin/vmax so colormap is consistent
    im = axes[i, 1].imshow(scm[0], cmap='jet')
    axes[i, 1].axis('off')
    
    axes[i, 2].imshow(blended)
    axes[i, 2].axis('off')
    
    label_str = "LIVE" if labels[idx] == 1 else "SPOOF"
    axes[i, 0].text(10, 30, label_str, color='white' if labels[idx]==0 else 'black', 
                    fontsize=14, fontweight='bold', bbox=dict(facecolor='red' if labels[idx]==0 else 'green', alpha=0.7))

plt.tight_layout()
plt.savefig("scm_heatmaps8.png", dpi=300)
plt.show()
print("Analysis complete. Check 'latent_space_tsne.png' and 'scm_heatmaps.png'.")