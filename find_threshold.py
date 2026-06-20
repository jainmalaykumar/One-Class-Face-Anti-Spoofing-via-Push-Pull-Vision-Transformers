import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, roc_curve

# Import your Phase V architecture
from model import SCMNet

# ==========================================
# CONFIGURATION (UPDATE THESE PATHS)
# ==========================================
BEST_MODEL_PATH = "checkpoints_v7/casia/SCMNet/model_count_010.pth" # e.g., your Epoch 10 weights
TEST_LIVE_PATH = "datasets/casia_mfsd/test_live.npy"
TEST_SPOOF_PATH = "datasets/casia_mfsd/test_spoof.npy"
BATCH_SIZE = 30

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def NormalizeData_torch(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    """
    Uses Youden's J statistic to find the optimal threshold.
    Maximizes the distance from random guessing.
    """
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
# 1. LOAD DATASET
# ==========================================
print("Loading Test Dataset...")
test_live_data = np.load(TEST_LIVE_PATH)
test_spoof_data = np.load(TEST_SPOOF_PATH)

print(f"Live Samples: {test_live_data.shape[0]}, Spoof Samples: {test_spoof_data.shape[0]}")

# Labels: Live = 0 (Normal), Spoof = 1 (Anomaly)
labels0 = np.zeros(test_live_data.shape[0]) 
labels1 = np.ones(test_spoof_data.shape[0]) 

test_images = np.vstack([test_live_data, test_spoof_data]) 
test_labels = np.hstack([labels0, labels1]) 

# Convert to Tensors: NCHW format
test_images_tensor = torch.tensor(test_images).permute(0, 3, 1, 2).float()  
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
# 3. EXTRACT SCORES (INFERENCE)
# ==========================================
all_scores = []
new_labels = []

print("Running Inference on Dataset to extract SCM scores...")
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        
        # Get the Spoof Cue Map [B, 1, 32, 32]
        spoof_cue = model(NormalizeData_torch(images))

        # Calculate Global Average Pixel Intensity
        # sum the maps of every pixel and normalize it with product of height and width
        scores_cues = torch.sum(spoof_cue, dim=(2, 3)) / (spoof_cue.shape[2] * spoof_cue.shape[3])
        scores_cues = torch.squeeze(scores_cues, 1)

        for k in range(0, spoof_cue.size(0)):
            all_scores.append(1.0 * scores_cues[k].cpu().numpy())
            new_labels.append(labels[k].cpu().numpy())

# ==========================================
# 4. CALCULATE OPTIMAL DEPLOYMENT THRESHOLD
# ==========================================
print("Calculating ROC Curve and Youden's Index...")
fpr, tpr, thresholds = roc_curve(new_labels, all_scores, pos_label=1)
optimal_threshold = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)

# ==========================================
# 5. SANITY CHECK (FINAL METRICS)
# ==========================================
# Calculate how this specific threshold actually performs
TP = TN = FP = FN = 0.0000001

for j in range(len(all_scores)):
    score = all_scores[j]
    if (score >= optimal_threshold and new_labels[j] == 1):
        TP += 1
    elif (score < optimal_threshold and new_labels[j] == 0):
        TN += 1
    elif (score >= optimal_threshold and new_labels[j] == 0):
        FP += 1
    elif (score < optimal_threshold and new_labels[j] == 1):
        FN += 1

APCER = FP / (TN + FP)
NPCER = FN / (FN + TP)
ACER = (APCER + NPCER) / 2
AUC = roc_auc_score(new_labels, all_scores)

# ==========================================
# FINAL OUTPUT
# ==========================================
print("\n" + "="*50)
print("             DEPLOYMENT RESULTS")
print("="*50)
print(f"AUC   (Area Under Curve): {AUC:.4f}")
print(f"ACER  (Avg Error Rate)  : {ACER:.4f}")
print(f"APCER (False Positives) : {APCER:.4f}")
print(f"BPCER (False Negatives) : {NPCER:.4f}")
print("-" * 50)
print(f"OPTIMAL DEPLOYMENT THRESHOLD: >>> {optimal_threshold:.6f} <<<")
print("="*50)
print("\nUsage for Real-World API:")
print(f"if current_image_score >= {optimal_threshold:.6f}:")
print("    return 'SPOOF'")
print("else:")
print("    return 'LIVE'")