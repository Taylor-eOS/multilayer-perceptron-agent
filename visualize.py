import torch
import torch.nn as nn
import random
import numpy as np
import time
import os

GRID_SIZE = 5
STATE_SIZE = GRID_SIZE * GRID_SIZE * 3
ACTION_SIZE = 4
HIDDEN_SIZE = 128
MODEL_PATH = "mlp_agent.pth"
SLEEP_TIME = 1

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
    def render(self):
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        grid[self.food_pos[0]][self.food_pos[1]] = 'F'
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        os.system('cls' if os.name == 'nt' else 'clear')
        for row in grid:
            print(' '.join(row))
        print()

class Agent:
    def __init__(self, state_size, action_size, hidden_size, model_path):
        self.model = MLP(state_size, hidden_size, action_size)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

def main():
    env = GridEnv(GRID_SIZE)
    agent = Agent(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE, MODEL_PATH)
    state = env.reset()
    env.render()
    done = False
    step_count = 0
    while not done and step_count < 50:
        action = agent.select_action(state)
        state, reward, done = env.step(action)
        env.render()
        time.sleep(SLEEP_TIME)
        step_count += 1
    if done:
        if env.agent_pos == env.food_pos:
            print("Agent found the food!")
        else:
            print("Agent did not find the food.")
    else:
        print("Agent did not find the food within the step limit.")

if __name__ == "__main__":
    main()

