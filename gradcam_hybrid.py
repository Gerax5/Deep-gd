import torch
import numpy as np
import cv2
from torchvision.transforms.functional import to_tensor
from torchcam.methods import GradCAM
import imageio
from tqdm import tqdm

from gd_env import GeometryEnv
from utils_visual import FrameStacker, render_env
from ModelHybrid import HybridAgent


OUTPUT_VIDEO = "gradcam_hybrid.gif"
FRAME_SIZE = (600, 200)
EPISODE_STEPS = 300
SAVE_VIDEO = True

env = GeometryEnv()
fs = FrameStacker(k=4, out_size=(84, 84), grayscale=True)
agent = HybridAgent(state_size=len(env.get_state()), in_channels=4, action_size=2)
agent.model.load_state_dict(torch.load("geometry_dqn_hybrid.pth", map_location="cpu"))
agent.model.eval()

cam = GradCAM(agent.model.cnn, target_layer="2")  


frames = []
env.reset()
first_frame = render_env(env)
visual = fs.reset(first_frame)
state_vec = env.get_state()

for t in tqdm(range(EPISODE_STEPS), desc="Simulando episodio con Grad-CAM híbrido"):
    v = torch.from_numpy(visual).unsqueeze(0).float()
    s = torch.from_numpy(state_vec).unsqueeze(0).float()

    out = agent.model(v, s)
    action = int(torch.argmax(out, dim=1))
    idx = action

    maps = cam(idx, out)
    cam_map = maps[0].squeeze().detach().numpy()
    cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)
    heatmap = cv2.resize(cam_map, FRAME_SIZE)
    colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    frame = render_env(env)
    overlay = cv2.addWeighted(frame, 0.6, colored, 0.4, 0)

    txt = "SALTAR 🟩" if action == 1 else "NO SALTAR 🟥"
    cv2.putText(overlay, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    frames.append(overlay[..., ::-1])

    next_state, reward, done, _ = env.step(action)
    frame_next = render_env(env)
    visual = fs.step(frame_next)
    state_vec = next_state

    if done:
        break

if SAVE_VIDEO:
    imageio.mimsave(OUTPUT_VIDEO, frames)
    print(f"\n✅ Video guardado como: {OUTPUT_VIDEO}")
else:
    import matplotlib.pyplot as plt
    for i, f in enumerate(frames):
        plt.imshow(f)
        plt.axis("off")
        plt.title(f"Frame {i}")
        plt.pause(0.05)
    plt.show()
