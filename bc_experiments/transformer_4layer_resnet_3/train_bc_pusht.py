#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_bc_pusht_overfit.py  —— 全量 overfit 训练：ResNet + Transformer
（输入图像 + 状态，输出动作；100 epoch；大 batch；不做拆分；无 LR 调度）
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from bc_model import ResNetStateFusionTrans

# ----- 1) 读取全量数据集（不拆分） -----
ds = load_dataset("lerobot/pusht_image", split="train")
imgs    = np.stack(ds["observation.image"])
states  = np.stack(ds["observation.state"]).astype("f")
actions = np.stack(ds["action"]).astype("f")

# 图像预处理
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])

img_t    = torch.stack([to_tensor(im) for im in imgs])
state_t  = torch.from_numpy(states)
action_t = torch.from_numpy(actions)
dataset  = TensorDataset(img_t, state_t, action_t)

# 用更大的 batch size 全量加载
BATCH_SIZE = 512
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# ----- 2) 模型 & 全量微调 -----
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNetStateFusionTrans().to(device)
for p in model.parameters():
    p.requires_grad = True

# ----- 3) 优化器 + 损失 -----
optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.MSELoss()

# ----- 4) Overfit 训练 100 epoch -----
EPOCHS = 100
for ep in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for img, st, act in tqdm(loader, desc=f"Ep{ep}/{EPOCHS}"):
        img, st, act = img.to(device), st.to(device), act.to(device)
        pred = model(img, st)
        loss = criterion(pred, act)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item() * img.size(0)

    avg_loss = total_loss / len(dataset)
    print(f"Epoch {ep:03d}/{EPOCHS}  Avg L1 Loss: {avg_loss:.4f}")

# ----- 5) 保存模型 -----
torch.save(model.state_dict(), "bc_resnet_trans_pusht_overfit.pt")
print("✓ Saved overfit model to bc_resnet_trans_pusht_overfit.pt")
