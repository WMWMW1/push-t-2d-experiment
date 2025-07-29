import pygame, numpy as np
import gymnasium as gym
import gym_pusht

pygame.init()
env = gym.make("gym_pusht/PushT-v0", render_mode="human")
obs, info = env.reset()

running = True
while running:
    action = np.zeros(2, dtype=np.float32)
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_LEFT:  action[0] = -1
            if e.key == pygame.K_RIGHT: action[0] = +1
            if e.key == pygame.K_UP:    action[1] = +1
            if e.key == pygame.K_DOWN:  action[1] = -1

    obs, r, done, trunc, info = env.step(action)
    env.render()

    if done or trunc:
        obs, info = env.reset()

env.close()
pygame.quit()
