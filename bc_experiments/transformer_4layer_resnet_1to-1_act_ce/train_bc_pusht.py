#!/usr/bin/env python3
# train_bc_pusht_norm_chunk.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import ToTensor, Normalize, Compose
from tqdm import tqdm
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from bc_model import ResNetStateFusionTrans

# ---- 超参 ----
ROOT_DIR    = "./lerobot_pusht"
BATCH_SIZE  = 2048
EPOCHS      = 15
LR          = 1e-4
CHUNK_SIZE  = 8
XY_MAX      = 512.0

# ---- 归一化函数 ----
def normalize(xy):
    return (torch.as_tensor(xy, dtype=torch.float32) / XY_MAX) * 2.0 - 1.0

# ---- 数据准备 ----
transforms_img = Compose([
    ToTensor(),                              # → [0,1]
    Normalize((0.485,0.456,0.406),           # ImageNet norm
              (0.229,0.224,0.225))
])

ds_raw = LeRobotDataset(
    repo_id="lerobot/pusht",
    root=ROOT_DIR,
    download_videos=True,
    video_backend="pyav",
    image_transforms=transforms_img
)

class PushTChunkDataset(Dataset):
    def __init__(self, ds_raw, chunk_size):
        self.ds_raw     = ds_raw
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.ds_raw) - self.chunk_size + 1

    def __getitem__(self, idx):
        item = self.ds_raw[idx]
        img   = item["observation.image"]                 # Tensor(C,H,W)
        state = normalize(item["observation.state"])      # Tensor(2,)

        # 连续取 chunk_size 步 action
        actions = []
        for j in range(self.chunk_size):
            a_j = normalize(self.ds_raw[idx + j]["action"])
            actions.append(a_j)
        actions = torch.stack(actions, dim=0)  # (chunk_size, 2)

        return img, state, actions

dataset = PushTChunkDataset(ds_raw, CHUNK_SIZE)
loader  = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=16,
    drop_last=True,
    pin_memory=True
)

# ---- 模型、优化器 ----
device   = "cuda" if torch.cuda.is_available() else "cpu"
model    = ResNetStateFusionTrans(chunk_size=CHUNK_SIZE).to(device)
optimizer= torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
criterion= nn.MSELoss()

# ---- 训练循环 ----
for ep in range(1, EPOCHS+1):
    model.train()
    total_loss = 0.0
    for imgs, states, act_chunks in tqdm(loader, desc=f"Ep{ep}/{EPOCHS}", ncols=100):
        imgs, states, act_chunks = imgs.to(device), states.to(device), act_chunks.to(device)
        preds = model(imgs, states)  # (B, chunk_size, 2)
        loss  = criterion(preds, act_chunks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)

    avg = total_loss / len(dataset)
    print(f"[Epoch {ep:03d}] avg_MSE_chunk={avg:.4f}")

# 保存
torch.save(model.state_dict(), "bc_resnet_trans_chunk.pt")
print("✓ saved → bc_resnet_trans_chunk.pt")
