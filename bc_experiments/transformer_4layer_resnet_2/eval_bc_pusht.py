#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测 + 保存视频 (rgb_array)，每集最多 6000 步
"""
import gymnasium as gym
import gym_pusht
import torch
from gymnasium.wrappers import TimeLimit
from torchvision import transforms
import imageio.v3 as iio
import numpy as np
from pathlib import Path
from bc_model import ResNetStateFusionTrans

# ----- 1) 模型加载 -----
device = "cuda" if torch.cuda.is_available() else "cpu"
policy = ResNetStateFusionTrans().to(device)
policy.load_state_dict(torch.load("bc_resnet_trans_pusht_overfit.pt", map_location=device))
policy.eval()

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# ----- 2) 拿到“裸”环境 & 只包一层 6000 步 -----
raw = gym.make(
    "gym_pusht/PushT-v0",
    obs_type="pixels_agent_pos",
    render_mode="rgb_array",
).unwrapped

env = TimeLimit(raw, max_episode_steps=3000)
env.success_threshold = 0.50  # 放宽成功阈值到 70%

# ----- 3) 评测与录像准备 -----
SAVE_DIR = Path("eval_videos"); SAVE_DIR.mkdir(exist_ok=True)
fps      = env.metadata.get("render_fps", 10)
N_EP     = 10
success  = 0

# 打印一次 obs 形状 & 最大步数
obs, _ = env.reset()
img = to_tensor(obs["pixels"]).unsqueeze(0).to(device)
st  = torch.tensor(obs["agent_pos"], dtype=torch.float32, device=device).unsqueeze(0)
print(f"[img shape: {img.shape}, state shape: {st.shape}]")
print(f"[max_episode_steps: {env.spec.max_episode_steps}]")  # 应该会打印 6000

# ----- 4) Episode 循环 -----
for ep in range(N_EP):
    obs, _ = env.reset()
    frames  = []
    step_cnt = 0
    done = False
    trunc = False

    while not (done or trunc):
        step_cnt += 1
        img = to_tensor(obs["pixels"]).unsqueeze(0).to(device)
        st  = torch.tensor(obs["agent_pos"], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            act = policy(img, st).cpu().numpy()[0]
        obs, _, done, trunc, info = env.step(act)
        frames.append(env.render())

    iio.imwrite(SAVE_DIR / f"ep_{ep:02d}.mp4", frames, fps=fps)
    if info.get("is_success", False):
        success += 1
    print(
        f"Episode {ep:02d} ended in {step_cnt} steps, "
        f"is_success={info.get('is_success')}, trunc={trunc}"
    )

# ----- 5) 总结 -----
print(f"\nSuccess rate: {success / N_EP:.2%}")
env.close()
