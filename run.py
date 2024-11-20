# Training Script: train.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
GRID_SIZE=5
STATE_SIZE=GRID_SIZE*GRID_SIZE*3 +5*4 +5*2
ACTION_SIZE=4
HIDDEN_SIZE=128
LEARNING_RATE=0.001
GAMMA=0.99
MEMORY_SIZE=10000
BATCH_SIZE=64
EPISODES=5000
MAX_STEPS=50
EXPLORATION_INTERVAL=100
EXPLORATION_EPISODE_LENGTH=20
ACTION_HISTORY_LENGTH=5
POSITION_HISTORY_LENGTH=5
WALL_PENALTY=-1.0
STUCK_PENALTY=-0.5
STEP_LIMIT_PENALTY=-0.5
class MLP(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(MLP,self).__init__()
        self.fc1=nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(hidden_size,hidden_size)
        self.fc3=nn.Linear(hidden_size,output_size)
    def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x
class GridEnv:
    def __init__(self,grid_size):
        self.grid_size=grid_size
        self.reset()
    def reset(self):
        positions=[(i,j) for i in range(self.grid_size) for j in range(self.grid_size)]
        column=random.randint(0,self.grid_size-1)
        start_row=random.randint(0,self.grid_size-4)
        self.walls=[(start_row+k,column) for k in range(4)]
        for wall in self.walls:
            positions.remove(wall)
        self.agent_pos=list(random.choice(positions))
        positions.remove(tuple(self.agent_pos))
        self.food_pos=list(random.choice(positions))
        positions.remove(tuple(self.food_pos))
        self.done=False
        self.step_count=0
        self.action_history=deque(maxlen=ACTION_HISTORY_LENGTH)
        self.position_history=deque(maxlen=POSITION_HISTORY_LENGTH)
        self.position_history.append(tuple(self.agent_pos))
        return self.get_state()
    def get_state(self):
        state=np.zeros((self.grid_size,self.grid_size,3),dtype=np.float32)
        state[self.agent_pos[0]][self.agent_pos[1]][0]=1.0
        state[self.food_pos[0]][self.food_pos[1]][1]=1.0
        for wall in self.walls:
            state[wall[0]][wall[1]][2]=1.0
        action_history_encoded=np.zeros(ACTION_HISTORY_LENGTH * ACTION_SIZE,dtype=np.float32)
        for idx,action in enumerate(self.action_history):
            action_history_encoded[idx * ACTION_SIZE + action]=1.0
        position_history_encoded=np.zeros(POSITION_HISTORY_LENGTH *2,dtype=np.float32)
        for idx,pos in enumerate(self.position_history):
            position_history_encoded[idx *2]=pos[0]/(self.grid_size -1)
            position_history_encoded[idx *2 +1]=pos[1]/(self.grid_size -1)
        return np.concatenate([state.flatten(), action_history_encoded, position_history_encoded])
    def step(self,action):
        target_pos=self.agent_pos.copy()
        if action==0 and self.agent_pos[0]>0:
            target_pos[0]-=1
        elif action==1 and self.agent_pos[0]<self.grid_size -1:
            target_pos[0]+=1
        elif action==2 and self.agent_pos[1]>0:
            target_pos[1]-=1
        elif action==3 and self.agent_pos[1]<self.grid_size -1:
            target_pos[1]+=1
        if tuple(target_pos) in self.walls:
            reward=WALL_PENALTY
            self.done=True
        else:
            self.agent_pos=target_pos
            self.action_history.append(action)
            self.position_history.append(tuple(self.agent_pos))
            self.step_count+=1
            reward=-0.1
            if self.agent_pos==self.food_pos:
                reward=1.0
                self.done=True
            elif self.step_count >= MAX_STEPS:
                reward += STEP_LIMIT_PENALTY
                self.done=True
        if tuple(self.agent_pos) in self.walls and not self.done:
            reward += STUCK_PENALTY
        return self.get_state(), reward, self.done
    def render_grid(self):
        grid=[['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for poison in self.walls:
            grid[poison[0]][poison[1]]='P'
        grid[self.food_pos[0]][self.food_pos[1]]='F'
        grid[self.agent_pos[0]][self.agent_pos[1]]='A'
        return grid
class ReplayMemory:
    def __init__(self,capacity):
        self.memory=deque(maxlen=capacity)
    def push(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)
    def __len__(self):
        return len(self.memory)
class Agent:
    def __init__(self,state_size,action_size,hidden_size,lr,gamma,memory_size,batch_size):
        self.model=MLP(state_size,hidden_size,action_size).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.optimizer=optim.Adam(self.model.parameters(),lr=lr)
        self.criterion=nn.MSELoss()
        self.memory=ReplayMemory(memory_size)
        self.gamma=gamma
        self.batch_size=batch_size
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def select_action(self,state,explore=False):
        if explore:
            return random.randint(0,ACTION_SIZE-1)
        state=torch.tensor(state,dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values=self.model(state)
        return torch.argmax(q_values).item()
    def train_step(self):
        if len(self.memory)<self.batch_size:
            return
        batch=self.memory.sample(self.batch_size)
        states,actions,rewards,next_states,dones=zip(*batch)
        states=torch.tensor(states,dtype=torch.float32).to(self.device)
        actions=torch.tensor(actions,dtype=torch.long).unsqueeze(1).to(self.device)
        rewards=torch.tensor(rewards,dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states=torch.tensor(next_states,dtype=torch.float32).to(self.device)
        dones=torch.tensor(dones,dtype=torch.float32).unsqueeze(1).to(self.device)
        q_values=self.model(states).gather(1,actions)
        next_q_values=self.model(next_states).max(1)[0].unsqueeze(1)
        target=rewards + self.gamma * next_q_values * (1 - dones)
        loss=self.criterion(q_values,target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
def train():
    env=GridEnv(GRID_SIZE)
    agent=Agent(STATE_SIZE,ACTION_SIZE,HIDDEN_SIZE,LEARNING_RATE,GAMMA,MEMORY_SIZE,BATCH_SIZE)
    exploration_count=0
    total_rewards=0
    total_successes=0
    total_poisons=0
    total_step_limit_penalties=0
    total_stuck_penalties=0
    for episode in range(EPISODES):
        state=env.reset()
        episode_reward=0
        episode_success=0
        episode_poison=0
        episode_step_limit=0
        episode_stuck=0
        if exploration_count < EPISODES // EXPLORATION_INTERVAL:
            explore=True
            exploration_count +=1
        else:
            explore=False
        for step in range(MAX_STEPS):
            if explore and step < EXPLORATION_EPISODE_LENGTH:
                action=agent.select_action(state,explore=True)
            else:
                action=agent.select_action(state,explore=False)
            next_state,reward,done=env.step(action)
            agent.memory.push(state,action,reward,next_state,done)
            state=next_state
            episode_reward +=reward
            if reward ==1.0:
                episode_success=1
            elif reward ==WALL_PENALTY:
                episode_poison=1
            elif reward <= STEP_LIMIT_PENALTY:
                episode_step_limit=1
            elif reward <= STUCK_PENALTY:
                episode_stuck=1
            agent.train_step()
            if done:
                break
        total_rewards +=episode_reward
        total_successes +=episode_success
        total_poisons +=episode_poison
        total_step_limit_penalties +=episode_step_limit
        total_stuck_penalties +=episode_stuck
        if (episode+1) %100 ==0:
            average_reward=total_rewards /100
            average_success=total_successes
            average_poison=total_poisons
            average_step_limit=total_step_limit_penalties
            average_stuck=total_stuck_penalties
            print(f"Average Reward: {average_reward}, Successes: {average_success}, Poisons: {average_poison}, Limits: {average_step_limit}, Stuck: {average_stuck}")
            total_rewards=0
            total_successes=0
            total_poisons=0
            total_step_limit_penalties=0
            total_stuck_penalties=0
    torch.save(agent.model.state_dict(),"mlp_agent_with_poison_walls.pth")
if __name__=="__main__":
    train()

