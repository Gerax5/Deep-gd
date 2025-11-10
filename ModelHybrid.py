# ModelHybrid.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class HybridDQN(nn.Module):
    """
    Red híbrida: combina estado tabular + representación visual (frames).
    Entrada:
      - vector numérico (9,)
      - stack de imágenes (C,84,84)
    """
    def __init__(self, state_size, in_channels, action_size):
        super().__init__()

        # Bloque convolucional
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.ReLU()
        )

        # calcular tamaño plano
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            n_flat = self.cnn(dummy).view(1, -1).size(1)

        # Bloque MLP (vector)
        self.mlp_state = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Fusión
        self.fc = nn.Sequential(
            nn.Linear(n_flat + 64, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

    def forward(self, visual, state):
        v = self.cnn(visual)
        v = v.view(v.size(0), -1)
        s = self.mlp_state(state)
        x = torch.cat([v, s], dim=1)
        return self.fc(x)


class HybridAgent:
    def __init__(self, state_size, in_channels, action_size,
                 lr=1e-4, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.995):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.action_size = action_size
        self.memory = deque(maxlen=100_000)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = HybridDQN(state_size, in_channels, action_size).to(self.device)
        self.target_model = HybridDQN(state_size, in_channels, action_size).to(self.device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def act(self, visual, state_vec, explore=True):
        if explore and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            v = torch.from_numpy(visual).unsqueeze(0).float().to(self.device)
            s = torch.from_numpy(state_vec).unsqueeze(0).float().to(self.device)
            q = self.model(v, s)
            return int(torch.argmax(q).item())

    def remember(self, visual, state_vec, action, reward, next_visual, next_state, done):
        self.memory.append((visual, state_vec, action, reward, next_visual, next_state, done))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        v = torch.from_numpy(np.stack([b[0] for b in batch])).float().to(self.device)
        s = torch.from_numpy(np.stack([b[1] for b in batch])).float().to(self.device)
        a = torch.tensor([b[2] for b in batch], dtype=torch.long, device=self.device)
        r = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=self.device)
        nv = torch.from_numpy(np.stack([b[4] for b in batch])).float().to(self.device)
        ns = torch.from_numpy(np.stack([b[5] for b in batch])).float().to(self.device)
        d = torch.tensor([b[6] for b in batch], dtype=torch.float32, device=self.device)

        q = self.model(v, s).gather(1, a.view(-1,1)).squeeze(1)
        with torch.no_grad():
            nq = self.target_model(nv, ns).max(1)[0]
            target = r + (1 - d) * self.gamma * nq

        loss = self.loss_fn(q, target)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
