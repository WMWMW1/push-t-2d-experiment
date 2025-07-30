#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_bc_pusht_norm.py —— 载入归一化版本模型并对输入/输出做对应的归一化/反归一化
"""
import gymnasium as gym
import gym_pusht
import torch
import imageio.v3 as iio
import numpy as np
from gymnasium.wrappers import TimeLimit
from torchvision import transforms
from pathlib import Path

from bc_model import ResNetStateFusionTrans

# ---------- 1) 常量 & 归/反归一化函数 ----------
XY_MAX = 512.0

def normalize(xy):
    """
    把环境原始坐标 [0, XY_MAX] → [-1, 1]
    """
    return (xy / XY_MAX) * 2.0 - 1.0

def denormalize(xy_n):
    """
    把模型输出坐标 [-1, 1] → [0, XY_MAX]
    """
    return (xy_n + 1.0) / 2.0 * XY_MAX

# ---------- 2) 模型加载 ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
policy = ResNetStateFusionTrans().to(device)
policy.load_state_dict(
    torch.load("bc_resnet_trans_norm.pt", map_location=device)
)
policy.eval()

# 图像预处理：与训练阶段保持完全一致
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])

# ---------- 3) 环境配置 ----------
raw = gym.make(
    "gym_pusht/PushT-v0",
    obs_type="pixels_agent_pos",
    render_mode="rgb_array"
).unwrapped
env = TimeLimit(raw, max_episode_steps=2000)
env.success_threshold = 0.70

SAVE_DIR = Path("eval_videos_norm")
SAVE_DIR.mkdir(exist_ok=True)
fps = env.metadata.get("render_fps", 10)

# ---------- 4) 评估循环 ----------
N_EP = 10
success = 0

for ep in range(N_EP):
    obs, _ = env.reset()
    frames = []
    done = False
    trunc = False
    step = 0

    while not (done or trunc):
        step += 1

        # 4.1 处理图像
        img = to_tensor(obs["pixels"]).unsqueeze(0).to(device)

        # 4.2 处理 state：归一化到 [-1,1]
        st_raw = torch.tensor(
            obs["agent_pos"],
            dtype=torch.float32,
            device=device
        )
        st_n = normalize(st_raw)
        st = st_n.unsqueeze(0)  # (1,2)

        # 4.3 模型推理
        with torch.no_grad():
            act_n = policy(img, st).cpu().numpy()[0]  # 预测输出 ∈ [-1,1]（若加了 tanh，必然如此）

        # 4.4 反归一化到环境坐标
        act = denormalize(act_n).astype(np.float32)  # ∈ [0,512]

        # 4.5 环境交互
        obs, _, done, trunc, info = env.step(act)
        frames.append(env.render())

    # 保存视频 & 统计成功
    iio.imwrite(SAVE_DIR / f"ep_{ep:02d}.mp4", frames, fps=fps)
    success += int(info.get("is_success", False))
    print(f"ep {ep:02d}: steps={step}, success={info.get('is_success')}")

print(f"\nSuccess rate: {success/N_EP:.2%}")
env.close()
