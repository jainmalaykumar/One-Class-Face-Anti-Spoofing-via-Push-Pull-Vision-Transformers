import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import numpy as np
import random

def softplus(x):
    return torch.nn.functional.softplus(x, beta=100)

 
class block(nn.Module):

    def __init__(self,begin):
        super(block,self).__init__()

        if begin==True:
            self.cnn1=nn.Conv2d(in_channels=64,out_channels=128, kernel_size=3,stride=1,padding=1)
        else:
            self.cnn1=nn.Conv2d(in_channels=128,out_channels=128, kernel_size=3,stride=1,padding=1)
        
        nn.init.xavier_normal_(self.cnn1.weight)
        self.bn1=nn.BatchNorm2d(128,track_running_stats=False)
        # self.non_linearity1=nn.CELU(alpha=1.0, inplace=False)
        self.non_linearity1 = nn.ReLU(inplace=False)
        self.cnn2=nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3,stride=1,padding=1)
        nn.init.xavier_normal_(self.cnn2.weight)
        self.bn2=nn.BatchNorm2d(196,track_running_stats=False)
        # self.non_linearity2=nn.CELU(alpha=1.0, inplace=False)
        self.non_linearity2 = nn.ReLU(inplace=False)
        
        self.cnn3=nn.Conv2d(in_channels=196, out_channels=128, kernel_size=3,stride=1,padding=1)
        nn.init.xavier_normal_(self.cnn3.weight)
        self.bn3=nn.BatchNorm2d(128,track_running_stats=False)
        # self.non_linearity3=nn.CELU(alpha=1.0, inplace=False)
        self.non_linearity3 = nn.ReLU(inplace=False)

        
        self.pool=nn.MaxPool2d(kernel_size=2)

    def forward(self,x):
        
        x=self.cnn1(x)
        x=self.bn1(x)
        x=self.non_linearity1(x)
        x=self.cnn2(x)
        x=self.bn2(x)
        x=self.non_linearity2(x)
        x=self.cnn3(x)
        x=self.bn3(x)
        x=self.non_linearity3(x)
        x=self.pool(x)
        return x


import torchvision.models as models

# --- REPLACE Simple_FeatureExtractor WITH THIS ---

class Swin_FeatureExtractor(nn.Module):
    def __init__(self):
        super(Swin_FeatureExtractor, self).__init__()
        # Load pre-trained Swin-T
        swin = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        
        # We only need the feature extraction blocks, not the final classification head
        self.features = swin.features
        
        # Swin-T outputs [B, H, W, C]. For 256x256 input, this is [B, 8, 8, 768].
        # The SCM estimator (self.E) expects [B, 128, 32, 32].
        
        # --- THE REDESIGNED ADAPTER ---
        # We use learnable Transposed Convolutions to avoid positional artifacting.
        self.adapter = nn.Sequential(
            # 1. Channel Reduction: 768 -> 256
            nn.Conv2d(768, 256, kernel_size=1, bias=False), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 2. First Learnable Upsample: 8x8 -> 16x16
            # Output size = (input - 1) * stride - 2 * padding + kernel_size
            # (8 - 1)*2 - 2*1 + 4 = 14 - 2 + 4 = 16
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 3. Second Learnable Upsample & Channel Projection: 16x16 -> 32x32
            # (16 - 1)*2 - 2*1 + 4 = 30 - 2 + 4 = 32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            
            # 4. Final Refinement (Smooths out any residual upsampling grid artifacts)
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Pass through Swin Transformer
        x = self.features(x)
        
        # Permute from [Batch, Height, Width, Channels] to [Batch, Channels, Height, Width]
        # and force contiguous memory reallocation
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # Pass through the new learnable adapter to map cleanly to [B, 128, 32, 32]
        x = self.adapter(x)
        return x
    
# --- UPDATE SCMNet TO USE THE NEW EXTRACTOR ---

class SCMNet(nn.Module):  
    def __init__(self, output=2):
        super(SCMNet, self).__init__() 

        # Change this line to use the new Swin backbone
        self.F = Swin_FeatureExtractor()
        
        self.E = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        )
        
        # ... (keep self.G and the rest of SCMNet exactly the same) ...

        self.G = self.net = nn.Sequential(
            nn.Conv2d(129, 256, kernel_size=4, stride=2, padding=1), 
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # Upsample to 32x32
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # Upsample to 64x64
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), 
            )
        
    def forward(self, x, ms=None, update="learn_FE"):
        if self.training:
            
            # --- THE FIX: CORRECT PYTORCH FREEZING ---
            if update == "learn_FE":
                self.F.requires_grad_(True)
                self.F.features.requires_grad_(False) # Keep Swin backbone permanently frozen!
                self.G.requires_grad_(False)
                self.E.requires_grad_(True) 
            elif update == "learn_Gtr":
                self.F.requires_grad_(False)
                self.G.requires_grad_(True)
                self.E.requires_grad_(False) 
            # -----------------------------------------
            
            live_feature = F.normalize(self.F(x)) 
            
            # (Keep the rest of the forward function the exact same)
            noise = torch.randn(live_feature.shape).to(x.device)
            noise_ms_p = torch.cat([noise, ms], dim=1)
            
            # Force the spoof features onto the same unit hypersphere!
            partial_spoof_z = F.normalize(self.G(noise_ms_p))


            live_map = self.E(live_feature) 
            m_p = self.E(partial_spoof_z)

            return live_feature, partial_spoof_z, live_map, m_p

        else:
            
            feature = F.normalize(self.F(x))# torch.Size([2, 128, 32, 32])
            spoof_cue = self.E(feature) # torch.Size([2, 1, 32, 32])
            return spoof_cue
        
if __name__ == "__main__":
    net = SCMNet().mps()
    net(torch.randn(5, 3, 256, 256).mps())