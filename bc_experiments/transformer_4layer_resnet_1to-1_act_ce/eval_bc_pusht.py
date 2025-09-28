#!/usr/bin/env python3
# eval_bc_pusht.py — 并行评测（分类 head），修复 state 归一化错误
import shutil, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import gymnasium as gym, gym_pusht
import torch, imageio.v3 as iio
from gymnasium.wrappers import TimeLimit
from torchvision import transforms

from bc_model import ResNetStateFusionTrans, NUM_BINS

CHUNK    = 8
XY_MAX   = 512.0
N_EP     = 30
SAVE     = Path("eval_videos_class")
MODEL    = "bc_resnet_trans_chunk_cls.pt"

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),
                         (0.229,0.224,0.225))
])

def norm_state_tensor(xy_np, device):
    """
    将 numpy.ndarray 的 [x,y] 先转为 Tensor，再归一化到 [-1,1]
    """
    st = torch.tensor(xy_np, dtype=torch.float32, device=device)
    return (st / XY_MAX) * 2.0 - 1.0

def run(ep):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ResNetStateFusionTrans(chunk_size=CHUNK).to(device)
    net.load_state_dict(torch.load(MODEL, map_location=device))
    net.eval()

    env = TimeLimit(
        gym.make("gym_pusht/PushT-v0",
                 obs_type="pixels_agent_pos",
                 render_mode="rgb_array").unwrapped,
        max_episode_steps=1500
    )
    obs, _ = env.reset()
    frames = []
    done = trunc = False

    while not (done or trunc):
        # 1) 图像
        img = to_tensor(obs["pixels"]).unsqueeze(0).to(device)
        # 2) state: numpy -> tensor -> normalize -> (1,2)
        st = norm_state_tensor(obs["agent_pos"], device).unsqueeze(0)

        # 3) forward
        with torch.no_grad():
            logits = net(img, st)         # (1, k, 2, 512)
            acts_i = logits.argmax(-1)[0] # (k,2) long

        # 4) 执行动作 chunk
        for a in acts_i.cpu().numpy():
            obs, _, done, trunc, _ = env.step(a.astype(float))
            frames.append(env.render())
            if done or trunc:
                break

    # 存视频
    SAVE.mkdir(exist_ok=True)
    iio.imwrite(SAVE / f"ep_{ep:02d}.mp4", frames, fps=10)
    env.close()
    return int(done)

if __name__ == "__main__":
    # 清空旧结果
    shutil.rmtree(SAVE, ignore_errors=True)

    succ = 0
    with ProcessPoolExecutor() as pool:
        futures = [pool.submit(run, i) for i in range(N_EP)]
        for f in as_completed(futures):
            succ += f.result()
    print(f"Success {succ}/{N_EP} = {succ/N_EP:.2%}")
