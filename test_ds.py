#!/usr/bin/env python3
import matplotlib.pyplot as plt
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torchvision.transforms.v2 import ToTensor   # 可选

# ① 载入数据；root 指向你刚下载的 lerobot_pusht 目录
ds = LeRobotDataset(
    repo_id="lerobot/pusht",           # 仍然是 state-only 版
    root="./lerobot_pusht",            # 或者让它自己下载
    download_videos=True,              # 带下 mp4，才能解码
    image_transforms=ToTensor(),       # 随便放个 transform
    video_backend="pyav",              # 没装 torch-codec 就用 pyav
)

print(ds)                              # 看一下数据集信息

# ② 拿第一条样本
sample = ds[0]
rgb = sample["observation.image"].permute(1, 2, 0).numpy()  # (96,96,3)
state = sample["observation.state"].numpy()                 # (x, y)

# ③ 显示
plt.imshow(rgb)
plt.title(f"state = {state}")
plt.axis("off")
plt.show()
