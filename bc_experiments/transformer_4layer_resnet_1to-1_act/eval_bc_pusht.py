#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_bc_pusht_norm_chunk_parallel.py — 并行评测 action-chunking 策略，
按 10Hz（0.1s）控制频率执行动作，评测前清空输出文件夹
"""
import os
import shutil
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import gymnasium as gym
import gym_pusht
import torch
import imageio.v3 as iio
import numpy as np
from gymnasium.wrappers import TimeLimit
from torchvision import transforms

from bc_model import ResNetStateFusionTrans

# ---- 常量 & 函数 ----
CHUNK_SIZE   = 8
XY_MAX       = 512.0
CTRL_PERIOD  = 0.1   # 10 Hz

# 并行评测轮数
N_EP         = 14
SAVE_DIR     = Path("eval_videos_chunk")
MODEL_PATH   = "bc_resnet_trans_chunk.pt"

def normalize(xy):
    return (xy / XY_MAX) * 2.0 - 1.0

def denormalize(xy_n):
    return (xy_n + 1.0) / 2.0 * XY_MAX

def evaluate_episode(ep: int):
    """
    在单进程中跑 1 个 episode，并保存视频到 SAVE_DIR/ep_{ep:02d}.mp4
    返回 (ep, success_flag)
    """
    # 1) 模型加载
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = ResNetStateFusionTrans(chunk_size=CHUNK_SIZE).to(device)
    policy.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    policy.eval()

    # 2) 图像预处理
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])

    # 3) 环境
    raw = gym.make("gym_pusht/PushT-v0",
                   obs_type="pixels_agent_pos",
                   render_mode="rgb_array").unwrapped
    env = TimeLimit(raw, max_episode_steps=1500)
    env.success_threshold = 0.70

    # 4) 评测循环
    obs, _ = env.reset()
    frames = []
    done = trunc = False
    step = 0

    while not (done or trunc):
        # 4.1 取 obs 图像 & state
        img = to_tensor(obs["pixels"]).unsqueeze(0).to(device)
        st_raw = torch.tensor(obs["agent_pos"],
                              dtype=torch.float32,
                              device=device)
        st = normalize(st_raw).unsqueeze(0)

        # 4.2 模型预测 chunk_size 步动作
        with torch.no_grad():
            act_seq_n = policy(img, st).cpu().numpy()[0]  # (CHUNK_SIZE, 2)

        # 4.3 依次执行 chunk 中的每一步
        for act_n in act_seq_n:
            step += 1
            action = denormalize(act_n).astype(np.float32)  # (2,)
            obs, _, done, trunc, info = env.step(action)
            frames.append(env.render())
            # 按 10Hz 节奏补足时长

            if done or trunc:
                break

    # 5) 保存视频
    video_path = SAVE_DIR / f"ep_{ep:02d}.mp4"
    iio.imwrite(video_path, frames, fps=int(1/CTRL_PERIOD))

    env.close()
    return ep, int(info.get("is_success", False))


if __name__ == "__main__":
    # 评测前清空输出文件夹
    if SAVE_DIR.exists():
        shutil.rmtree(SAVE_DIR)
    SAVE_DIR.mkdir(exist_ok=True)

    # 并行执行 N_EP episodes
    success = 0
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(evaluate_episode, ep): ep for ep in range(N_EP)}
        for future in as_completed(futures):
            ep, succ = future.result()
            success += succ
            print(f"ep {ep:02d}: success={bool(succ)}")

    print(f"\nOverall Success rate: {success}/{N_EP} = {success/N_EP:.2%}")
