import os
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

warnings.simplefilter('ignore')
os.environ['PYTHONHASHSEED'] = '0'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hu=24):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hu)
        self.fc2 = nn.Linear(hu, hu)
        self.fc3 = nn.Linear(hu, action_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQLAgent:
    def __init__(self, symbol, feature, n_features, env, hu=24, lr=0.001):
        self.epsilon = 1.0
        self.epsilon_decay = 0.9975
        self.epsilon_min = 0.1
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        self.gamma = 0.5
        self.trewards = []
        self.max_treward = -np.inf
        self.n_features = n_features
        self.env = env
        self.episodes = 0
        # Q-Network and optimizer
        self.model = QNetwork(self.n_features, self.env.action_space.n, hu).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.symbol = symbol
        self.feature = feature
        self.n_features = n_features

        # Si env est un entier, nous devons l'adapter
        if isinstance(env, int):
            # Créer une structure simple pour simuler l'environnement
            class SimpleEnv:
                def __init__(self, n_actions):
                    self.action_space = type('obj', (object,), {'n': n_actions})

            self.env = SimpleEnv(env)  # env représente le nombre d'actions
        else:
            self.env = env  # env est déjà un environnement approprié

        self.hu = hu
        self.lr = lr

        # Reste du code d'initialisation...
        self.episodes = 0
        # Q-Network and optimizer
        self.model = QNetwork(self.n_features, self.env.action_space.n, hu).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()



    def _reshape(self, state):
        state = state.flatten()
        return np.reshape(state, [1, len(state)])

    def act(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        state_tensor = torch.FloatTensor(state).to(device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return int(torch.argmax(q_values[0]).item())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states = np.vstack([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        next_states = np.vstack([e[2] for e in batch])
        rewards = np.array([e[3] for e in batch], dtype=np.float32)
        dones = np.array([e[4] for e in batch], dtype=bool)

        states_tensor = torch.FloatTensor(states).to(device)
        next_states_tensor = torch.FloatTensor(next_states).to(device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards_tensor = torch.FloatTensor(rewards).to(device)
        dones_tensor = torch.BoolTensor(dones).to(device)

        current_q = self.model(states_tensor).gather(1, actions_tensor).squeeze(1)
        next_q = self.model(next_states_tensor).max(1)[0]
        target_q = rewards_tensor + self.gamma * next_q * (~dones_tensor).float()

        loss = self.criterion(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self, episodes):
        for e in range(1, episodes + 1):
            self.episodes += 1
            state, _ = self.env.reset()
            state = self._reshape(state)
            treward = 0
            for f in range(1, 5000):
                self.f = f
                action = self.act(state)
                next_state, reward, done, trunc, _ = self.env.step(action)
                treward += reward
                next_state = self._reshape(next_state)
                self.memory.append((state, action, next_state, reward, done))
                state = next_state
                if done:
                    self.trewards.append(treward)
                    self.max_treward = max(self.max_treward, treward)
                    templ = f'episode={self.episodes:4d} | '
                    templ += f'treward={treward:7.3f} | max={self.max_treward:7.3f}'
                    print(templ, end='\r')
                    break
            if len(self.memory) > self.batch_size:
                self.replay()
            print()

    def test(self, episodes, min_accuracy=0.0, min_performance=0.0, verbose=True, full=True):
        # Backup and set environment thresholds
        ma = getattr(self.env, 'min_accuracy', None)
        if hasattr(self.env, 'min_accuracy'):
            self.env.min_accuracy = min_accuracy
        mp = None
        if hasattr(self.env, 'min_performance'):
            mp = self.env.min_performance
            self.env.min_performance = min_performance
            self.performances = []
        for e in range(1, episodes + 1):
            state, _ = self.env.reset()
            state = self._reshape(state)
            for f in range(1, 5001):
                action = self.act(state)
                state, reward, done, trunc, _ = self.env.step(action)
                state = self._reshape(state)
                if done:
                    templ = f'total reward={f:4d} | accuracy={self.env.accuracy:.3f}'
                    if hasattr(self.env, 'min_performance'):
                        self.performances.append(self.env.performance)
                        templ += f' | performance={self.env.performance:.3f}'
                    if verbose:
                        if full:
                            print(templ)
                        else:
                            print(templ, end='\r')
                    break
        # Restore environment thresholds
        if hasattr(self.env, 'min_accuracy') and ma is not None:
            self.env.min_accuracy = ma
        if mp is not None:
            self.env.min_performance = mp
        print()