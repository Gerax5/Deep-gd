import numpy as np
from PIL import Image
from collections import deque
import cv2

def preprocess_frame(frame, out_size=(84, 84), grayscale=True):
    img = Image.fromarray(frame)
    if grayscale:
        img = img.convert("L")  # 1 canal
    img = img.resize(out_size, Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr  # (H,W) o (H,W,3)

class FrameStacker:
    """Mantiene un stack de k frames preprocesados, salida (k,H,W)."""
    def __init__(self, k=4, out_size=(84, 84), grayscale=True):
        self.k = k
        self.out_size = out_size
        self.grayscale = grayscale
        self.buffer = deque(maxlen=k)

    def reset(self, first_frame):
        f = preprocess_frame(first_frame, self.out_size, self.grayscale)
        self.buffer.clear()
        for _ in range(self.k):
            self.buffer.append(f)
        return np.stack(self.buffer, axis=0)  # (k,H,W)

    def step(self, frame):
        f = preprocess_frame(frame, self.out_size, self.grayscale)
        self.buffer.append(f)
        return np.stack(self.buffer, axis=0)  # (k,H,W)




def render_env(env):
    img = np.ones((env.HEIGHT, env.WIDTH, 3), dtype=np.uint8) * 255

    # jugador (azul)
    cv2.rectangle(
        img,
        (env.player_x, int(env.player_y)),
        (env.player_x + env.player_size, int(env.player_y + env.player_size)),
        (0, 0, 255),
        -1
    )

    # obstáculos (negros)
    for ox, oy, tipo, ow, oh in env.obstacles:
        color = (0, 0, 0) if tipo == "bloque" else (255, 0, 0)
        cv2.rectangle(img, (int(ox), int(oy)), (int(ox + ow), int(oy + oh)), color, -1)

    return img