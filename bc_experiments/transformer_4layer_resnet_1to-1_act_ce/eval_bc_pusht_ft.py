#!/usr/bin/env python3
# eval_bc_pusht_ft.py — 并行评测 fine-tuned 分类模型
import shutil, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import gymnasium as gym, gym_pusht
import torch, imageio.v3 as iio
from gymnasium.wrappers import TimeLimit
from torchvision import transforms

from bc_model import ResNetStateFusionTrans, NUM_BINS

# ──────────── 配置 ────────────
CHUNK      = 8
XY_MAX     = 512.0
N_EP       = 50
MODEL_FT   = "bc_resnet_trans_chunk_cls_ft.pt"   # Fine-tuned 权重
SAVE_FT    = Path("eval_videos_class_ft")        # 输出目录

# 图像预处理
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),
                         (0.229,0.224,0.225))
])

def norm_state_tensor(xy_np, device):
    """
    把 numpy [x,y] 转为 tensor 并归一化到 [-1,1]
    """
    st = torch.tensor(xy_np, dtype=torch.float32, device=device)
    return (st / XY_MAX) * 2.0 - 1.0

def run_episode(ep: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 加载 fine-tuned 模型
    net = ResNetStateFusionTrans(chunk_size=CHUNK).to(device)
    net.load_state_dict(torch.load(MODEL_FT, map_location=device))
    net.eval()

    # 创建环境
    raw = gym.make("gym_pusht/PushT-v0",
                   obs_type="pixels_agent_pos",
                   render_mode="rgb_array").unwrapped
    env = TimeLimit(raw, max_episode_steps=1500)
    obs, _ = env.reset()

    frames = []
    done = trunc = False

    while not (done or trunc):
        img = to_tensor(obs["pixels"]).unsqueeze(0).to(device)
        st  = norm_state_tensor(obs["agent_pos"], device).unsqueeze(0)

        with torch.no_grad():
            logits = net(img, st)           # (1, CHUNK, 2, 512)
            acts_i = logits.argmax(-1)[0]   # (CHUNK,2)

        for a in acts_i.cpu().numpy():
            obs, _, done, trunc, _ = env.step(a.astype(float))
            frames.append(env.render())
            if done or trunc:
                break

    # 保存视频
    SAVE_FT.mkdir(parents=True, exist_ok=True)
    out_path = SAVE_FT / f"ep_{ep:02d}.mp4"
    iio.imwrite(out_path, frames, fps=10)

    env.close()
    return int(done)

if __name__ == "__main__":
    # 清空旧结果
    shutil.rmtree(SAVE_FT, ignore_errors=True)
    SAVE_FT.mkdir(parents=True, exist_ok=True)

    success = 0
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_episode, i) for i in range(N_EP)]
        for f in as_completed(futures):
            success += f.result()

    print(f"Fine-tuned Success: {success}/{N_EP} = {success/N_EP:.2%}")
