# test_hybrid.py
import pygame
import torch
import numpy as np
from gd_env import GeometryEnv
from utils_visual import FrameStacker, render_env
from ModelHybrid import HybridAgent

device = torch.device("cpu")
checkpoint_path = "geometry_dqn_hybrid.pth"

env = GeometryEnv()
fs = FrameStacker(k=4, out_size=(84,84), grayscale=True)
agent = HybridAgent(state_size=len(env.get_state()), in_channels=4, action_size=2)
agent.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
agent.model.eval()

pygame.init()
WIDTH, HEIGHT = 600, 200
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Geometry Dash Hybrid DQN")
WHITE, GRAY, DARK_GRAY, RED, BLUE, BLACK = (255,255,255),(150,150,150),(80,80,80),(255,50,50),(0,150,255),(0,0,0)
clock = pygame.time.Clock()

state = env.reset()
frame = render_env(env)
visual = fs.reset(frame)
score, running = 0, True
font = pygame.font.SysFont("Arial", 20)

while running:
    clock.tick(30)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    with torch.no_grad():
        action = agent.act(visual, state, explore=False)

    next_state, reward, done, _ = env.step(action)
    frame_next = render_env(env)
    visual = fs.step(frame_next)
    state = next_state
    score += reward

    win.fill(WHITE)
    pygame.draw.rect(win, BLUE, (env.player_x, env.player_y, env.player_size, env.player_size))
    for ox, oy, tipo, ow, oh in env.obstacles:
        if tipo == "bloque":
            pygame.draw.rect(win, GRAY, (ox, oy, ow, oh))
        elif tipo == "bloqueXL":
            pygame.draw.rect(win, DARK_GRAY, (ox, oy, ow, oh))
        else:
            pygame.draw.polygon(win, RED, [(ox, oy+oh), (ox+ow//2, oy), (ox+ow, oy+oh)])
    pygame.draw.rect(win, BLACK, (0, HEIGHT-20, WIDTH, 20))
    text = font.render(f"Score: {int(score)}", True, BLACK)
    win.blit(text, (10, 10))
    pygame.display.update()

    if done:
        print("Game Over | Score:", int(score))
        running = False

pygame.quit()
