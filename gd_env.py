import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# --------------------------
# ENTORNO CON PATRONES REALES
# --------------------------
class GeometryEnv:
    def __init__(self):
        self.WIDTH, self.HEIGHT = 600, 200
        self.player_x = 50
        self.player_size = 20
        self.gravity = 1
        self.vel_y = 0
        self.jump_force = -11
        self.ground_y = self.HEIGHT - self.player_size - 20
        self.is_jumping = False
        self.obstacle_speed = 6
        self.obstacles = []
        # Patrones de obstáculos
        self.patterns = [
            ["bloque"],
            ["pincho"],
            ["bloque", "bloque", "bloque"],
            ["pincho", "bloque", "pincho"],
            ["pincho", "pincho", "pincho"],
            ["bloque", "pincho", "pincho"],
            ["bloque","pincho", "pincho", "pincho", "bloqueXL"],
            ["bloque", "pincho", "pincho", "pincho", "bloqueXL", "pincho", "pincho", "pincho", "pincho"], 
            ["bloque", "pincho", "pincho", "pincho", "bloqueXL", "pincho", "pincho", "pincho", "pincho", "bloque", "pincho", "pincho", "pincho", "bloque"]
        ]
        self.reset()

    # --------------------------
    # Reiniciar entorno
    # --------------------------
    def reset(self):
        self.player_y = self.ground_y
        self.vel_y = 0
        self.is_jumping = False
        self.obstacles = self._generate_pattern(self.WIDTH)
        self.next_x = self.WIDTH + 300
        self.ticks_alive = 0
        return self.get_state()

    # --------------------------
    # Generar patrón
    # --------------------------
    def _generate_pattern(self, start_x):
        pattern = random.choice(self.patterns)
        new_obs = []
        current_x = start_x

        for tipo in pattern:
            if tipo == "bloqueXL":
                width, height = 30, 60
            elif tipo == "bloque":
                width, height = 20, 30
            else:  # pincho
                width, height = 20, 30

            x = current_x
            y = self.HEIGHT - height - 20
            new_obs.append([x, y, tipo, width, height])
            current_x += width + 10
        return new_obs

    # --------------------------
    # Estado
    # --------------------------
     # --------------------------
    # Estado (mejorado)
    # --------------------------
    def get_state(self):
        # Obtener los dos obstáculos más cercanos delante del jugador
        obs = [o for o in self.obstacles if o[0] + o[3] >= self.player_x]
        obs_sorted = sorted(obs, key=lambda o: o[0])

        # Si no hay obstáculos, devolver vector vacío
        if not obs_sorted:
            return np.zeros(9, dtype=np.float32)

        # Primer obstáculo
        ox, oy, tipo, ow, oh = obs_sorted[0]
        dist = (ox - (self.player_x + self.player_size)) / self.WIDTH
        vy = self.vel_y / 15.0
        on_ground = 1.0 if self.player_y >= self.ground_y else 0.0
        tipo_idx = {"pincho": 0.0, "bloque": 0.5, "bloqueXL": 1.0}[tipo]
        height_ratio = oh / 60.0

        # 🔹 Nuevo: tiempo estimado hasta el obstáculo
        time_to_obstacle = (ox - (self.player_x + self.player_size)) / self.obstacle_speed
        time_to_obstacle = np.clip(time_to_obstacle / 100.0, 0.0, 1.0)

        # 🔹 Nuevo: predicción de caída si salta ahora
        predicted_y = self.player_y
        vel = self.jump_force
        for t in range(15):
            predicted_y += vel
            vel += self.gravity
            if predicted_y >= self.ground_y:
                predicted_y = self.ground_y
                break
        landing_y_diff = (self.ground_y - predicted_y) / self.HEIGHT

        # 🔹 Segundo obstáculo (visión a futuro)
        if len(obs_sorted) > 1:
            ox2, oy2, tipo2, ow2, oh2 = obs_sorted[1]
            dist2 = (ox2 - (self.player_x + self.player_size)) / self.WIDTH
            tipo2_idx = {"pincho": 0.0, "bloque": 0.5, "bloqueXL": 1.0}[tipo2]
        else:
            dist2, tipo2_idx = 1.0, 0.0

        # 🔹 Estado final (9 valores)
        return np.array([
            dist, vy, on_ground, tipo_idx, height_ratio,
            time_to_obstacle, landing_y_diff,
            dist2, tipo2_idx
        ], dtype=np.float32)

    # --------------------------
    # Paso del entorno
    # --------------------------
    def step(self, action):
        reward, done = 0.0, False

        # Acción: 1 = saltar
        if action == 1 and not self.is_jumping:
            self.vel_y = self.jump_force
            self.is_jumping = True

        # Física del jugador
        self.player_y += self.vel_y
        self.vel_y += self.gravity
        if self.player_y >= self.ground_y:
            self.player_y = self.ground_y
            self.is_jumping = False

        # Mover obstáculos
        for obs in self.obstacles:
            obs[0] -= self.obstacle_speed

        # 🔹 NUEVO: detectar pinchos superados
        passed_spikes = 0
        for ox, oy, tipo, ow, oh in self.obstacles:
            if tipo == "pincho" and ox + ow < self.player_x and not hasattr(self, "passed_ids"):
                self.passed_ids = set()
            if tipo == "pincho" and ox + ow < self.player_x:
                # Crear ID único del obstáculo
                obs_id = id(obs)
                if obs_id not in self.passed_ids:
                    self.passed_ids.add(obs_id)
                    passed_spikes += 1

        if passed_spikes > 0:
            reward += passed_spikes * 3.0  # recompensa extra por pincho superado

        # Generar nuevos patrones
        if self.obstacles and self.obstacles[-1][0] < self.WIDTH - 250:
            self.obstacles.extend(self._generate_pattern(self.next_x))
            self.next_x += random.randint(200, 400)

        # Eliminar viejos
        self.obstacles = [o for o in self.obstacles if o[0] > -60]

        # Colisiones
        for ox, oy, tipo, ow, oh in self.obstacles:
            if tipo == "pincho":
                px_center = self.player_x + self.player_size / 2
                py_bottom = self.player_y + self.player_size
                if (ox < px_center < ox + ow) and (py_bottom > oy):
                    reward = -5
                    done = True
                    break

            elif tipo in ["bloque", "bloqueXL"]:
                if (self.player_x + self.player_size > ox and self.player_x < ox + ow):
                    if (self.player_y + self.player_size > oy and
                        self.player_y + self.player_size - self.vel_y <= oy):
                        self.player_y = oy - self.player_size
                        self.vel_y = 0
                        self.is_jumping = False
                        reward += 2.0
                    elif self.player_y + self.player_size > oy:
                        reward = -5
                        done = True
                        break

        if not done:
            reward += 0  # sobrevivir

        self.ticks_alive += 1
        return self.get_state(), reward, done, {}
