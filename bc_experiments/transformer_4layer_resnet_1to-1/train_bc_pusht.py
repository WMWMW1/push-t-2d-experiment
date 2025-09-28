#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_bc_pusht_norm_ds.py —— ResNet-18 + 4-Layer Transformer
数据改为直接使用 LeRobotDataset（含视频解码），其余设置与
train_bc_pusht_norm.py 保持一致：
  • 动作 / state 归一化到 [-1,1]
  • batch_size = 64
  • 损失函数 = MSELoss
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import ToTensor, Normalize, Compose
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

from bc_model import ResNetStateFusionTrans   # ← 你的模型实现

# ───────────── 1) 数据集 ─────────────
ROOT_DIR = "./lerobot_pusht"   # 已下载目录，也可随意指定
TRANSFORMS = Compose([
    ToTensor(),                                    # (H,W,C)→(C,H,W), 范围 [0,1]
    Normalize((0.485, 0.456, 0.406),               # ImageNet 归一化
              (0.229, 0.224, 0.225))
])

ds_raw = LeRobotDataset(
    repo_id="lerobot/pusht",
    root=ROOT_DIR,
    download_videos=True,        # 需要视频才能 decode image
    video_backend="pyav",
    image_transforms=TRANSFORMS, # 直接在内部做 ToTensor+Norm
)

# ───────────── 2) 归一化助手 ─────────────
XY_MAX = 512.0
def normalize(xy):           
    return (torch.as_tensor(xy, dtype=torch.float32) / XY_MAX) * 2.0 - 1.0

# ───────────── 3) Dataset 包装器 ─────────────
class PushTDataset(torch.utils.data.Dataset):
    def __init__(self, ds_raw):
        self.ds = ds_raw
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        item = self.ds[idx]
        img     = item["observation.image"]        # 已是 Tensor(C,H,W)
        state   = normalize(item["observation.state"])
        action  = normalize(item["action"])
        return img, state, action

dataset = PushTDataset(ds_raw)

# ───────────── 4) DataLoader ─────────────
BATCH_SIZE = 64
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    drop_last=True,
    pin_memory=True,
)

# ───────────── 5) 模型 ─────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNetStateFusionTrans().to(device)

# ───────────── 6) 优化器 & 损失 ─────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.MSELoss()

# ───────────── 7) 训练循环 ─────────────
EPOCHS = 10
for ep in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for imgs, states, acts in tqdm(loader, desc=f"Ep {ep}/{EPOCHS}", ncols=100):
        imgs, states, acts = imgs.to(device), states.to(device), acts.to(device)

        preds = model(imgs, states)
        loss  = criterion(preds, acts)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    avg = running_loss / len(dataset)
    print(f"[Epoch {ep:03d}]  avg_MSE={avg:.4f}")

# ───────────── 8) 保存权重 ─────────────
torch.save(model.state_dict(), "bc_resnet_trans_norm.pt")
print("✓ saved → bc_resnet_trans_norm.pt")
