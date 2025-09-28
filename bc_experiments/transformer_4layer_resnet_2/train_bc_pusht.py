#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_bc_pusht_overfit.py  —— 全量 overfit 训练：ResNet + Transformer
（输入图像 + 状态，输出动作；100 epoch；大 batch；不做拆分；无 LR 调度 → 加入 StepLR；支持断点续训）
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from bc_model import ResNetStateFusionTrans

# ---------------- 配置 ----------------
CHECKPOINT_PATH = "checkpoint.pth"   # 断点文件
FINAL_MODEL    = "bc_resnet_trans_pusht_overfit.pt"
BATCH_SIZE     = 512
LR_INIT        = 1e-3
WEIGHT_DECAY   = 1e-4
EPOCHS         = 150
STEP_SIZE      = 10    # 每隔多少 epoch 衰减一次
GAMMA          = 0.05  # 衰减倍率
NUM_WORKERS    = 4

# 1. 读取全量数据集（不拆分）
ds       = load_dataset("lerobot/pusht_image", split="train")
imgs     = np.stack(ds["observation.image"])
states   = np.stack(ds["observation.state"]).astype("f")
actions  = np.stack(ds["action"]).astype("f")

# 图像预处理
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
])
img_t     = torch.stack([to_tensor(im) for im in imgs])
state_t   = torch.from_numpy(states)
action_t  = torch.from_numpy(actions)
dataset   = TensorDataset(img_t, state_t, action_t)
loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# 2. 初始化模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNetStateFusionTrans().to(device)
for p in model.parameters():
    p.requires_grad = True

# 3. 优化器、损失函数、调度器
optim     = torch.optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=STEP_SIZE, gamma=GAMMA)
criterion = nn.L1Loss()

# 4. 断点续训：检查 checkpoint.pth
start_epoch = 1
if os.path.isfile(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optim.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    print(f"→ 从 checkpoint 恢复：已完成 {ckpt['epoch']} / {EPOCHS}，下次从 epoch {start_epoch} 开始训练")

# 5. 训练循环
for ep in range(start_epoch, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for img, st, act in tqdm(loader, desc=f"Ep{ep:03d}/{EPOCHS}"):
        img, st, act = img.to(device), st.to(device), act.to(device)

        pred  = model(img, st)
        loss  = criterion(pred, act)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item() * img.size(0)

    # 更新学习率
    scheduler.step()

    # 计算并打印平均 loss
    avg_loss = total_loss / len(dataset)
    current_lr = optim.param_groups[0]['lr']
    print(f"Epoch {ep:03d}/{EPOCHS}  Avg L1 Loss: {avg_loss:.4f}  LR: {current_lr:.2e}")

    # 保存 checkpoint
    torch.save({
        'epoch':                ep,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, CHECKPOINT_PATH)

# 6. 训练完毕，另存最终模型
torch.save(model.state_dict(), FINAL_MODEL)
print(f"✓ 训练完毕，断点保存在 {CHECKPOINT_PATH}，最终模型保存在 {FINAL_MODEL}")
