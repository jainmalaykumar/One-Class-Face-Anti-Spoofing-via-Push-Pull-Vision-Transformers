import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, roc_curve

# Import your Phase V architecture
from model import SCMNet

# ==========================================
# CONFIGURATION
# ==========================================
# Make sure this points to the exact epoch that gave you the 0.8322 AUC
BEST_MODEL_PATH = "checkpoints_v7/casia/SCMNet/model_count_010.pth" 
TEST_LIVE_PATH = "datasets/casia_mfsd/test_live.npy"
TEST_SPOOF_PATH = "datasets/casia_mfsd/test_spoof.npy"
BATCH_SIZE = 30

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def NormalizeData_torch(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR + (1 - FPR)
    Youden_index = np.argmax(y)
    optimal_threshold = threshold[Youden_index]
    return optimal_threshold

# ==========================================
# DEVICE SETUP
# ==========================================
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")

# ==========================================
# 1. LOAD DATASET (Exactly like test.py)
# ==========================================
print("Loading Test Dataset...")
test_live_data = np.load(TEST_LIVE_PATH)
test_spoof_data = np.load(TEST_SPOOF_PATH)

labels0 = np.zeros(test_live_data.shape[0])
labels1 = np.ones(test_spoof_data.shape[0])

test_images = np.vstack([test_live_data, test_spoof_data]) 
test_labels = np.hstack([labels0, labels1]) 

test_images_tensor = (torch.tensor(test_images).permute(0, 3, 1, 2).float())
test_images_tensor = NormalizeData_torch(test_images_tensor)
test_labels_tensor = torch.tensor(test_labels).float()

test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 2. LOAD MODEL
# ==========================================
print(f"Loading Model Weights from: {BEST_MODEL_PATH}")
model = SCMNet().to(device)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model.eval()

# ==========================================
# 3. EXTRACT SCORES (Exact match to test.py)
# ==========================================
all_scores = []
new_labels = []

print("Running Inference...")
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        
        # RAW OUTPUT. NO SIGMOID.
        spoof_cue = model(NormalizeData_torch(images))

        # sum the maps of every pixel and normalize it
        scores_cues = torch.sum(spoof_cue, dim=(2, 3)) / (spoof_cue.shape[2] * spoof_cue.shape[3])
        scores_cues = torch.squeeze(scores_cues, 1)

        for k in range(0, spoof_cue.size(0)):
            all_scores.append(1.0 * scores_cues[k].cpu().numpy())
            new_labels.append(labels[k].cpu().numpy())

all_scores = np.array(all_scores)
new_labels = np.array(new_labels)

# ==========================================
# 4. CALCULATE METRICS (Exact match to test.py)
# ==========================================
fpr, tpr, thresholds = roc_curve(new_labels, all_scores, pos_label=1)
threshold_cs = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)

TP = TN = FP = FN = 0.0000001
for j in range(len(all_scores)):
    score = all_scores[j]
    if (score >= threshold_cs and new_labels[j] == 1):
        TP += 1
    elif (score < threshold_cs and new_labels[j] == 0):
        TN += 1
    elif (score >= threshold_cs and new_labels[j] == 0):
        FP += 1
    elif (score < threshold_cs and new_labels[j] == 1):
        FN += 1

# Extract the identical variables used in test.py
APCER = FP / (TN + FP)
NPCER = FN / (FN + TP)
ACER = (APCER + NPCER) / 2
AUC = roc_auc_score(new_labels, all_scores)

print(f"\nVerified test.py Output -> ACER: {ACER:.4f} | AUC: {AUC:.4f} | APCER: {APCER:.4f} | BPCER: {NPCER:.4f}")
print(f"Optimal Threshold: {threshold_cs:.6f}")

# ==========================================
# 5. PLOT 1: ROC CURVE
# ==========================================
print("Generating ROC Curve...")
plt.figure(figsize=(8, 6))

# Plot the curve
plt.plot(fpr, tpr, color='blue', lw=2, label=f'OC-SCMNet (AUC = {AUC:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

# Mark the exact metric point mapped from your script
plt.scatter(APCER, (1 - NPCER), color='red', s=100, zorder=5, 
            label=f'Optimal Operation Point\n(APCER: {APCER:.4f}, BPCER: {NPCER:.4f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title(f'ROC Curve (ACER = {ACER:.4f})', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('roc_curve_paper1.png', dpi=300)
plt.savefig('roc_curve_paper1.pdf', format='pdf', dpi=300) 
plt.show()

# ==========================================
# 6. PLOT 2: SCORE DISTRIBUTION
# ==========================================
print("Generating Score Distribution Plot...")
plt.figure(figsize=(10, 6))

live_scores = all_scores[new_labels == 0]
spoof_scores = all_scores[new_labels == 1]

# Dynamic zooming to ignore wild outliers and make the graph clear
min_val = np.percentile(all_scores, 1)
max_val = np.percentile(all_scores, 99)

sns.kdeplot(live_scores, color='green', fill=True, label='Live Faces (Target)', alpha=0.5, linewidth=2, clip=(min_val, max_val))
sns.kdeplot(spoof_scores, color='red', fill=True, label='Spoof Attacks (OOD)', alpha=0.5, linewidth=2, clip=(min_val, max_val))

# Draw the exact threshold line found by test.py
plt.axvline(threshold_cs, color='black', linestyle='--', linewidth=2, label=f'Optimal Threshold: {threshold_cs:.6f}')

# Zoom the X-axis so the overlap is highly visible
plt.xlim(min_val, max_val)

plt.xlabel('Raw Global Average SCM Score', fontsize=12, fontweight='bold')
plt.ylabel('Density', fontsize=12, fontweight='bold')
plt.title('Latent SCM Score Distribution', fontsize=14, fontweight='bold')
plt.legend(loc="upper right", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('score_distribution_paper1.png', dpi=300)
plt.savefig('score_distribution_paper1.pdf', format='pdf', dpi=300)
plt.show()