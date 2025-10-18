import pygame
import sys
import numpy as np
from gd_env import GeometryEnv  # importa tu entorno real

# FUll CHATGPT no tenia ganas de pensar en como hacerlo

pygame.init()
WIDTH, HEIGHT = 600, 200
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Geometry Dash - Manual Test 🕹️")

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 150, 255)
RED = (255, 50, 50)
GRAY = (100, 100, 100)
DARK_GRAY = (60, 60, 60)

clock = pygame.time.Clock()
FPS = 60

# ===============================
# INICIALIZAR ENTORNO
# ===============================
env = GeometryEnv()
state = env.reset()
score = 0
font = pygame.font.SysFont("Arial", 20)
running = True

# ===============================
# LOOP PRINCIPAL (MANUAL)
# ===============================
while running:
    clock.tick(FPS)
    action = 0  # 0 = no saltar, 1 = saltar

    # Eventos de teclado / salida
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:  # salto manual
                action = 1
            elif event.key == pygame.K_r:  # reinicio manual
                state = env.reset()
                score = 0

    # También puedes hacer salto continuo al mantener presionado espacio:
    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE]:
        action = 1

    # Paso del entorno
    next_state, reward, done, _ = env.step(action)
    state = next_state
    score += reward

    # -------------------------------
    # Renderizar
    # -------------------------------
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
            # Pincho (triángulo)
            points = [(ox, oy + oh), (ox + ow // 2, oy), (ox + ow, oy + oh)]
            pygame.draw.polygon(win, RED, points)

    # Piso
    pygame.draw.rect(win, BLACK, (0, HEIGHT - 20, WIDTH, 20))

    # Score e info
    text = font.render(f"Score: {int(score)}", True, BLACK)
    win.blit(text, (10, 10))

    debug = font.render(f"State: {np.round(state, 2)}", True, (80, 80, 80))
    win.blit(debug, (10, 30))

    pygame.display.update()

    # Si muere
    if done:
        print("☠️ Game Over")
        pygame.time.wait(1000)  # pausa de 1 seg
        state = env.reset()
        score = 0

pygame.quit()
sys.exit()
