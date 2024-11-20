import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

GRID_SIZE = 5
STATE_SIZE = GRID_SIZE * GRID_SIZE * 3
ACTION_SIZE = 4
HIDDEN_SIZE = 128
LEARNING_RATE = 0.001
GAMMA = 0.99
MEMORY_SIZE = 10000
BATCH_SIZE = 64
EPISODES = 5000
MAX_STEPS = 50
EXPLORATION_INTERVAL = 100
EXPLORATION_EPISODE_LENGTH = 20

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class GridEnv:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.reset()
    def reset(self):
        self.agent_pos = [random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)]
        self.food_pos = [random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)]
        while self.food_pos == self.agent_pos:
            self.food_pos = [random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)]
        self.done = False
        return self.get_state()
    def get_state(self):
        state = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        state[self.agent_pos[0]][self.agent_pos[1]][0] = 1.0
        state[self.food_pos[0]][self.food_pos[1]][1] = 1.0
        return state.flatten()
    def step(self, action):
        if action == 0 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.grid_size -1:
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.grid_size -1:
            self.agent_pos[1] += 1
        reward = -0.1
        if self.agent_pos == self.food_pos:
            reward = 1.0
            self.done = True
        return self.get_state(), reward, self.done

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class Agent:
    def __init__(self, state_size, action_size, hidden_size, lr, gamma, memory_size, batch_size):
        self.model = MLP(state_size, hidden_size, action_size).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = ReplayMemory(memory_size)
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def select_action(self, state, explore=False):
        if explore:
            return random.randint(0, ACTION_SIZE-1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        q_values = self.model(states).gather(1, actions)
        next_q_values = self.model(next_states).max(1)[0].unsqueeze(1)
        target = rewards + self.gamma * next_q_values * (1 - dones)
        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def main():
    env = GridEnv(GRID_SIZE)
    agent = Agent(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE, LEARNING_RATE, GAMMA, MEMORY_SIZE, BATCH_SIZE)
    exploration_count = 0
    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        if exploration_count < EPISODES // EXPLORATION_INTERVAL:
            explore = True
            exploration_count += 1
        else:
            explore = False
        for step in range(MAX_STEPS):
            if explore and step < EXPLORATION_EPISODE_LENGTH:
                action = agent.select_action(state, explore=True)
            else:
                action = agent.select_action(state, explore=False)
            next_state, reward, done = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.train_step()
            if done:
                break
        if (episode+1) % 100 == 0:
            print(f"Episode {episode+1}, Total Reward: {total_reward}")
    torch.save(agent.model.state_dict(), "mlp_agent.pth")

if __name__ == "__main__":
    main()

