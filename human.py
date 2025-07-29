import pygame
import gymnasium as gym
import gym_pusht
from gymnasium.utils.play import play

# 创建环境（human 模式会弹 pygame 窗口）
env = gym.make("gym_pusht/PushT-v0", render_mode="human")

# 定义键盘→动作 的映射
# 这里 action_space 是 Box([-1,-1], [1,1])，对应推力向量 (x, y)
keys_to_action = {
    (pygame.K_LEFT,):  (-1.0,  0.0),
    (pygame.K_RIGHT,): ( 1.0,  0.0),
    (pygame.K_UP,):    ( 0.0,  1.0),
    (pygame.K_DOWN,):  ( 0.0, -1.0),
}

# 打开人机交互界面
play(env, keys_to_action=keys_to_action, zoom=4)
