import pygame
import torch
import numpy as np
from gd_env import GeometryEnv
from model import DQN

# === CARGA DEL MODELO ENTRENADO ===
checkpoint_path = "Deep-gd/geometry_dqn.pth"
device = torch.device("cpu")

# Crear entorno para conocer tamaño del estado
env = GeometryEnv()
state_dim = len(env.get_state())
n_actions = 2

# Crear modelo y cargar pesos
policy = DQN(state_dim, n_actions)
policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
policy.eval()

print("✅ Modelo cargado correctamente.\n")

# === CONFIGURACIÓN DE PYGAME ===
pygame.init()
WIDTH, HEIGHT = 600, 200
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Geometry Dash DQN")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 150, 255)
RED = (255, 50, 50)
GRAY = (100, 100, 100)
DARK_GRAY = (60, 60, 60)

clock = pygame.time.Clock()
FPS = 30

# === FUNCIÓN PARA ELEGIR ACCIÓN ===
def get_action(state):
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        qvals = policy(state_tensor)
        return int(torch.argmax(qvals).item())

# === LOOP DEL JUEGO ===
state = env.reset()
score = 0
font = pygame.font.SysFont("Arial", 20)
running = True

while running:
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action = get_action(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    score += reward

    # === DIBUJAR ESCENA ===
    win.fill(WHITE)

    # Jugador
    pygame.draw.rect(win, BLUE, (env.player_x, env.player_y, env.player_size, env.player_size))

    # Obstáculos
    for ox, oy, tipo, ow, oh in env.obstacles:
        if tipo == "bloque":
            pygame.draw.rect(win, GRAY, (ox, oy, ow, oh))
        elif tipo == "bloqueXL":
            pygame.draw.rect(win, DARK_GRAY, (ox, oy, ow, oh))
        else:
            points = [(ox, oy + oh), (ox + ow // 2, oy), (ox + ow, oy + oh)]
            pygame.draw.polygon(win, RED, points)

    # Piso
    pygame.draw.rect(win, BLACK, (0, HEIGHT - 20, WIDTH, 20))

    # Score
    text = font.render("Score: " + str(int(score)), True, (0, 0, 0))
    win.blit(text, (10, 10))

    pygame.display.update()

    # Fin del episodio
    if done:
        print("Game Over | Score final:", int(score))
        running = False

pygame.quit()
