#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_bc_pusht_norm.py —— ResNet-18 + 4-Layer Transformer
仅改 3 处：动作/state 归一化、batch_size=64、MSELoss
"""
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from bc_model import ResNetStateFusionTrans

# ---------- 1) 载入数据 ----------
ds = load_dataset("lerobot/pusht_image", split="train")
imgs    = np.stack(ds["observation.image"])
states  = np.stack(ds["observation.state"]).astype("f")    # (N,2)
actions = np.stack(ds["action"]).astype("f")               # (N,2)

# ---------- 2) 归一化助手 ----------
XY_MAX = 96.0   # Push-T 平面尺寸
def normalize(xy):      # [0,96] -> [-1,1]
    return (xy / XY_MAX) * 2.0 - 1.0
def denormalize(xy_n):  # [-1,1] -> [0,96]
    return (xy_n + 1.0) / 2.0 * XY_MAX

states_n  = normalize(states)
actions_n = normalize(actions)

# ---------- 3) TensorDataset ----------
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),
                         (0.229,0.224,0.225)),
])
img_t    = torch.stack([to_tensor(im) for im in imgs])
state_t  = torch.from_numpy(states_n)
action_t = torch.from_numpy(actions_n)
dataset  = TensorDataset(img_t, state_t, action_t)

# ---------- 4) DataLoader ----------
BATCH_SIZE = 64                    # ★ 改动 2
loader = DataLoader(dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=4,
                    drop_last=True)

# ---------- 5) 模型 ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNetStateFusionTrans().to(device)

# 保持你原先“全训练”策略：不冻结任何层
for p in model.parameters():
    p.requires_grad = True

# ---------- 6) 优化器 & 损失 ----------
optim = torch.optim.AdamW(model.parameters(),
                          lr=1e-4, weight_decay=1e-4)
criterion = nn.MSELoss()            # ★ 改动 3

# ---------- 7) 训练 ----------
EPOCHS = 100
for ep in range(1, EPOCHS + 1):
    model.train();  total = 0.0
    for img, st, act in tqdm(loader, desc=f"Ep{ep}/{EPOCHS}"):
        img, st, act = img.to(device), st.to(device), act.to(device)
        pred = model(img, st)
        loss = criterion(pred, act)
        optim.zero_grad(); loss.backward(); optim.step()
        total += loss.item() * img.size(0)
    print(f"[Epoch {ep:03d}]  avg_MSE={total/len(dataset):.4f}")

torch.save(model.state_dict(), "bc_resnet_trans_norm.pt")
print("✓ saved → bc_resnet_trans_norm.pt")
