#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_bc_pusht_full.py  —— 全量训练：ResNet + Transformer
（输入图像 + 状态，输出动作）
"""
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms, models
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from bc_model import ResNetStateFusionTrans

# ----- 1) 数据 -----
ds = load_dataset("lerobot/pusht_image", split="train")
imgs    = np.stack(ds["observation.image"])
states  = np.stack(ds["observation.state"]).astype("f")
actions = np.stack(ds["action"]).astype("f")

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])

img_t    = torch.stack([to_tensor(im) for im in imgs])
state_t  = torch.from_numpy(states)
action_t = torch.from_numpy(actions)
dataset  = TensorDataset(img_t, state_t, action_t)

train_set, val_set = random_split(dataset, [0.9,0.1],
                                  generator=torch.Generator().manual_seed(0))
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_set,   batch_size=128, num_workers=4)

# ----- 2) 模型 -----
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNetStateFusionTrans().to(device)

# 取消 backbone 冻结：确保所有参数都要梯度
for p in model.parameters():
    p.requires_grad = True

# ----- 3) 优化器 & 损失 -----
optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
# 降低学习率、weight decay，使预训练网络更稳定地微调
criterion = nn.L1Loss()

# ----- 4) 训练 -----
EPOCHS = 50
for ep in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for img, st, act in tqdm(train_loader, desc=f"Train Ep{ep}/{EPOCHS}"):
        img, st, act = img.to(device), st.to(device), act.to(device)
        pred = model(img, st)
        loss = criterion(pred, act)
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss += loss.item() * img.size(0)

    train_loss /= len(train_set)

    # 验证
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for img, st, act in val_loader:
            img, st, act = img.to(device), st.to(device), act.to(device)
            val_loss += criterion(model(img, st), act).item() * img.size(0)
    val_loss /= len(val_set)

    print(f"Epoch {ep}: train L1={train_loss:.4f}, val L1={val_loss:.4f}")

# ----- 5) 保存模型 -----
torch.save(model.state_dict(), "bc_resnet_trans_pusht.pt")
print("✓ Saved full-finetuned model to bc_resnet_trans_pusht.pt")
