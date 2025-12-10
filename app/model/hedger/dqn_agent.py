from collections import deque
import random
from typing import Tuple

import torch


class QNetwork(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 20_000,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.buffer: deque[Tuple] = deque(maxlen=buffer_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.q_net = QNetwork(state_dim, action_dim).to(self.device).float()
        self.target_net = QNetwork(state_dim, action_dim).to(self.device).float()
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()

    def store(self, s, a, r, ns, d) -> None:
        self.buffer.append((s, a, r, ns, d))

    def act(self, state, eps: float) -> int:
        if random.random() < eps:
            return random.randrange(self.action_dim)
        st_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.q_net(st_t)
        return int(q.argmax(dim=1).item())

    def train_step(self, batch_size: int = 64, tau: float = 0.01) -> None:
        if len(self.buffer) < batch_size:
            return
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        s_t = torch.tensor(s, dtype=torch.float32, device=self.device)
        a_t = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(-1)
        r_t = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(-1)
        ns_t = torch.tensor(ns, dtype=torch.float32, device=self.device)
        d_t = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(-1)

        qv = self.q_net(s_t).gather(1, a_t)
        with torch.no_grad():
            nq = self.target_net(ns_t).max(dim=1, keepdim=True)[0]
            tgt = r_t + self.gamma * (1 - d_t) * nq
        loss = self.loss_fn(qv, tgt)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)
