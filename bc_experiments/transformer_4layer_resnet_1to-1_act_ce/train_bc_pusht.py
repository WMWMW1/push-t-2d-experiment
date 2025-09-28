#!/usr/bin/env python3
# train_bc_pusht_class_chunk.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import ToTensor, Normalize, Compose
from tqdm import tqdm
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from bc_model import ResNetStateFusionTrans, NUM_BINS

# ───────────── 超参 ─────────────
ROOT_DIR   = "./lerobot_pusht"
BATCH_SIZE = 256          # 4090 上跑 2048 有点大，改成 256
EPOCHS     = 30
LR         = 5e-4         # 分类任务常用更大学习率
CHUNK_SIZE = 8
XY_MAX     = 512.0        # 坐标上限

# ───────────── 数据准备 ─────────────
img_tf = Compose([
    ToTensor(),
    Normalize((0.485,0.456,0.406),
              (0.229,0.224,0.225))
])

ds_raw = LeRobotDataset(
    repo_id="lerobot/pusht",
    root=ROOT_DIR,
    download_videos=True,
    video_backend="pyav",
    image_transforms=img_tf
)

def norm_state(xy: torch.Tensor) -> torch.Tensor:
    # state 仍归一到 [-1,1]
    return (xy / XY_MAX) * 2.0 - 1.0

class PushTChunkCls(Dataset):
    def __init__(self, raw: LeRobotDataset, k: int):
        self.raw = raw
        self.k   = k

    def __len__(self):
        return len(self.raw) - self.k + 1

    def __getitem__(self, i: int):
        item  = self.raw[i]
        img   = item["observation.image"]                 # Tensor(C,H,W)
        state = norm_state(item["observation.state"])     # Tensor(2,)

        # 连续取 k 步动作，并转成类别 long tensor
        acts = []
        for j in range(self.k):
            a = self.raw[i + j]["action"]                 # Tensor(2,) float
            # clamp 到 [0,XY_MAX-1] 再转 int
            a_clamped = torch.clamp(a, 0.0, XY_MAX - 1e-3)
            a_idx     = a_clamped.to(torch.long)          # Tensor(2,) long
            acts.append(a_idx)
        actions = torch.stack(acts, dim=0)                # (k,2) long

        return img, state, actions

dataset = PushTChunkCls(ds_raw, CHUNK_SIZE)
loader  = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    drop_last=True,
    pin_memory=True
)

# ───────────── 模型 & 损失 & 优化 ─────────────
device    = "cuda" if torch.cuda.is_available() else "cpu"
model     = ResNetStateFusionTrans(chunk_size=CHUNK_SIZE).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

scaler = torch.cuda.amp.GradScaler()  # mixed precision

# ───────────── 训练循环 ─────────────
for ep in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for imgs, states, targets in tqdm(loader, desc=f"Ep{ep}/{EPOCHS}", ncols=100):
        imgs, states, targets = imgs.to(device), states.to(device), targets.to(device)
        # targets: (B, k, 2)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(imgs, states)          # (B, k, 2, 512)
            # 展平到 (B*k*2, 512) vs (B*k*2,)
            loss   = criterion(
                logits.view(-1, NUM_BINS),
                targets.view(-1)
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)

    avg = total_loss / len(dataset)
    print(f"[Epoch {ep:02d}]  CE loss = {avg:.4f}")

# 保存模型
torch.save(model.state_dict(), "bc_resnet_trans_chunk_cls.pt")
print("✓ saved → bc_resnet_trans_chunk_cls.pt")
