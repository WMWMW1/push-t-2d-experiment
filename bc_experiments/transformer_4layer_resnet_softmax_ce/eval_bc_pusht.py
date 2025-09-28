#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测 + 保存视频 (rgb_array)，每集最多 1000 步
"""
import gymnasium as gym, gym_pusht, torch
from torchvision import transforms
import imageio.v3 as iio, numpy as np
from pathlib import Path
from bc_model import ResNetStateFusionTrans

# ----- 1) 模型 -----
device = "cuda" if torch.cuda.is_available() else "cpu"
policy = ResNetStateFusionTrans().to(device)
policy.load_state_dict(torch.load("bc_resnet_trans_pusht.pt", map_location=device))
policy.eval()

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# ----- 2) 环境（使用 max_episode_steps 覆盖默认 TimeLimit） -----
env = gym.make(
    "gym_pusht/PushT-v0",
    obs_type="pixels_agent_pos",
    render_mode="rgb_array",
    max_episode_steps=1000
)

# ----- 3) 评测 & 录像 -----
SAVE_DIR = Path("eval_videos"); SAVE_DIR.mkdir(exist_ok=True)
fps = env.metadata.get("render_fps", 10)
N_EP = 10
success = 0

# ----- 打印一次形状 & max steps -----
obs, _ = env.reset()
img = to_tensor(obs["pixels"]).unsqueeze(0).to(device)
st  = torch.tensor(obs["agent_pos"], dtype=torch.float32, device=device).unsqueeze(0)
print(f"[img shape: {img.shape}, state shape: {st.shape}]")
print(f"[max_episode_steps: {env.spec.max_episode_steps}]")
step_cnt=0

# ----- Episode 循环 -----
for ep in range(N_EP):
    obs, _ = env.reset()
    frames = []
    done = trunc = False

    while not (done or trunc):
        step_cnt += 1

        img = to_tensor(obs["pixels"]).unsqueeze(0).to(device)
        st  = torch.tensor(obs["agent_pos"], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            act = policy(img, st).cpu().numpy()[0]
            print(act)
        obs, _, done, trunc, info = env.step(act)
        frames.append(env.render())

    iio.imwrite(SAVE_DIR / f"ep_{ep:02d}.mp4", frames, fps=fps)
    print(f"Episode {ep:02d} ended after {step_cnt} steps, "
      f"is_success={info['is_success']}, trunc={trunc}")


print(f"\nSuccess rate: {success / N_EP:.2%}")
env.close()
