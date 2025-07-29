#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
push_t_mouse_drag.py  ——  PushT 人机交互（鼠标拖动）+ 自定义 max_episode_steps
                   —— 成功判定覆盖率放宽到 70%
"""

import pygame
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import gym_pusht

# ——— 1. 创建环境并把步数上限改大 + 放宽成功阈值 ———
BASE_ENV_ID = "gym_pusht/PushT-v0"
# 拆掉原有 TimeLimit，拿到“裸”环境
raw = gym.make(BASE_ENV_ID, render_mode="human").unwrapped
# 放宽成功阈值到 70%
raw.success_threshold = 0.70

MAX_STEPS = 5_000    # 想无限长就改成 None
# 只包一层 TimeLimit：6000 步
env = TimeLimit(raw, max_episode_steps=MAX_STEPS)

obs, info = env.reset()

# ——— 2. 初始化 pygame 计时器/状态 ———
pygame.init()
clock = pygame.time.Clock()
running = True
target = None        # 当前鼠标目标，None 表示不推

# ——— 3. 主循环 ———
while running:
    # 3.1 处理退出和鼠标事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 3.2 检测鼠标是否按住左键；若按住就取当前位置为目标
    if pygame.mouse.get_pressed()[0]:  # 按住左键
        mx, my = pygame.mouse.get_pos()
        target = np.array([mx, my], dtype=np.float32)
    else:
        target = None  # 松开键则不推

    # 3.3 计算动作并步进环境
    action = target if target is not None else np.array([0.0, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    # 3.4 渲染并维持 60 FPS
    env.render()
    clock.tick(60)

    # 3.5 回合结束：打印 success 并自动重置
    if terminated or truncated:
        print(
            f"回合结束: terminated={terminated}, truncated={truncated}, "
            f"coverage={info.get('coverage'):.3f}, is_success={info.get('is_success')}"
        )
        obs, info = env.reset()

# ——— 4. 退出清理 ———
env.close()
pygame.quit()
print("Bye!")
