#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resume_train_bc_pusht_class_chunk.py — 从已有分类模型断点续训，用 SGD + 更低 LR 精细打磨
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import ToTensor, Normalize, Compose
from tqdm import tqdm
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from bc_model import ResNetStateFusionTrans, NUM_BINS

# ───────────── 超参 ─────────────
ROOT_DIR     = "./lerobot_pusht"
BATCH_SIZE   = 64
EPOCHS_FINE  = 30     # 继续训练的 epoch 数
LR_FINE      = 1e-4   # 更低学习率
MOMENTUM     = 0.9
CHUNK_SIZE   = 8
XY_MAX       = 512.0

CHECKPOINT   = "bc_resnet_trans_chunk_cls.pt"      # 原始训练保存的模型
OUTPUT_MODEL = "bc_resnet_trans_chunk_cls_ft.pt"   # 续训后保存的模型

# ───────────── 数据准备 ─────────────
img_tf = Compose([
    ToTensor(),
    Normalize((0.485,0.456,0.406),
              (0.229,0.224,0.225))
])

ds_raw = LeRobotDataset(
    repo_id="lerobot/pusht",
    root=ROOT_DIR,
    download_videos=False,       # 已有数据，关掉下载
    image_transforms=img_tf,
    video_backend="pyav"
)

def norm_state(xy: torch.Tensor) -> torch.Tensor:
    return (xy / XY_MAX) * 2.0 - 1.0

class PushTChunkCls(Dataset):
    def __init__(self, raw: LeRobotDataset, k: int):
        self.raw = raw
        self.k   = k

    def __len__(self):
        return len(self.raw) - self.k + 1

    def __getitem__(self, i: int):
        item  = self.raw[i]
        img   = item["observation.image"]
        state = norm_state(item["observation.state"])
        acts  = []
        for j in range(self.k):
            a = self.raw[i + j]["action"]
            a_clamped = torch.clamp(a, 0.0, XY_MAX - 1e-3)
            acts.append(a_clamped.to(torch.long))
        actions = torch.stack(acts, dim=0)  # (k,2) long
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

# ───────────── 模型 & 优化器 ─────────────
device    = "cuda" if torch.cuda.is_available() else "cpu"
model     = ResNetStateFusionTrans(chunk_size=CHUNK_SIZE).to(device)

# 加载已有模型权重
state = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(state)

# 用 SGD 细调
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LR_FINE,
    momentum=MOMENTUM,
    weight_decay=1e-4
)
criterion = nn.CrossEntropyLoss()

# ───────────── 续训循环 ─────────────
for ep in range(1, EPOCHS_FINE+1):
    model.train()
    total_loss = 0.0

    for imgs, states, targets in tqdm(loader, desc=f"Fine-tune Ep{ep}/{EPOCHS_FINE}", ncols=100):
        imgs, states, targets = imgs.to(device), states.to(device), targets.to(device)

        optimizer.zero_grad()
        logits = model(imgs, states)               # (B, k, 2, 512)
        loss   = criterion(
            logits.view(-1, NUM_BINS),
            targets.view(-1)
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    avg = total_loss / len(dataset)
    print(f"[Fine Ep{ep:02d}] CE loss = {avg:.4f}")

# 保存续训后的模型
torch.save(model.state_dict(), OUTPUT_MODEL)
print(f"✓ saved fine-tuned model → {OUTPUT_MODEL}")
