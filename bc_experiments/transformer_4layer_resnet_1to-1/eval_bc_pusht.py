#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_bc_pusht_norm.py —— 载入归一化版本模型并反归一化动作
"""
import gymnasium as gym, gym_pusht
import torch, imageio.v3 as iio, numpy as np
from gymnasium.wrappers import TimeLimit
from torchvision import transforms
from pathlib import Path
from bc_model import ResNetStateFusionTrans

# ---------- 1) 常量 & 反归一化 ----------
XY_MAX = 96.0
def denormalize(xy_n):    # [-1,1] -> [0,96]
    return (xy_n + 1.0) / 2.0 * XY_MAX

# ---------- 2) 模型 ----------
device  = "cuda" if torch.cuda.is_available() else "cpu"
policy  = ResNetStateFusionTrans().to(device)
policy.load_state_dict(torch.load("bc_resnet_trans_norm.pt",
                                  map_location=device))
policy.eval()

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),
                         (0.229,0.224,0.225)),
])

# ---------- 3) 环境 ----------
raw  = gym.make("gym_pusht/PushT-v0",
                obs_type="pixels_agent_pos",
                render_mode="rgb_array").unwrapped
env  = TimeLimit(raw, max_episode_steps=6000)
env.success_threshold = 0.70

SAVE_DIR = Path("eval_videos_norm"); SAVE_DIR.mkdir(exist_ok=True)
fps = env.metadata.get("render_fps", 10)
N_EP = 10; success = 0

for ep in range(N_EP):
    obs,_ = env.reset(); frames=[]
    done=trunc=False; step=0
    while not (done or trunc):
        step += 1
        img = to_tensor(obs["pixels"]).unsqueeze(0).to(device)
        st  = torch.tensor(obs["agent_pos"], dtype=torch.float32,
                           device=device).unsqueeze(0)
        with torch.no_grad():
            act_n = policy(img, st).cpu().numpy()[0]      # [-1,1]
        act = denormalize(act_n).astype(np.float32)       # [0,96]
        obs,_,done,trunc,info = env.step(act)
        frames.append(env.render())

    iio.imwrite(SAVE_DIR/f"ep_{ep:02d}.mp4", frames, fps=fps)
    success += int(info.get("is_success", False))
    print(f"ep {ep:02d}: steps={step}, success={info.get('is_success')}")

print(f"\nSuccess rate: {success/N_EP:.2%}")
env.close()
