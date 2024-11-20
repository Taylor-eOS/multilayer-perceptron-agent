# Visualization Script: visualize.py
import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque
import tkinter as tk
GRID_SIZE=5
STATE_SIZE=GRID_SIZE*GRID_SIZE*3 +5*4 +5*2
ACTION_SIZE=4
HIDDEN_SIZE=128
MODEL_PATH="mlp_agent_with_poison_walls.pth"
SLEEP_TIME=500
WALL_PENALTY=-1.0
STUCK_PENALTY=-0.5
STEP_LIMIT_PENALTY=-0.5
ACTION_HISTORY_LENGTH=5
POSITION_HISTORY_LENGTH=5
MAX_STEPS=50
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
class AgentModel:
    def __init__(self,state_size,action_size,hidden_size,model_path):
        self.model=MLP(state_size,hidden_size,action_size)
        self.model.load_state_dict(torch.load(model_path,map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),weights_only=True),strict=True)
        self.model.eval()
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    def select_action(self,state):
        state=torch.tensor(state,dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values=self.model(state)
        return torch.argmax(q_values).item()
def main():
    env=GridEnv(GRID_SIZE)
    agent=AgentModel(STATE_SIZE,ACTION_SIZE,HIDDEN_SIZE,MODEL_PATH)
    state=env.reset()
    root=tk.Tk()
    root.title("Agent Visualization with Poison Walls")
    cell_size=60
    canvas_height=GRID_SIZE * cell_size +30
    canvas=tk.Canvas(root,width=GRID_SIZE * cell_size,height=canvas_height)
    canvas.pack()
    def draw_grid(grid):
        canvas.delete("all")
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x0=j * cell_size
                y0=i * cell_size
                x1=x0 + cell_size
                y1=y0 + cell_size
                canvas.create_rectangle(x0,y0,x1,y1,fill="white",outline="black")
                if grid[i][j]=='A':
                    canvas.create_text(x0 + cell_size/2,y0 + cell_size/2,text='A',font=("Helvetica",20))
                elif grid[i][j]=='F':
                    canvas.create_text(x0 + cell_size/2,y0 + cell_size/2,text='F',font=("Helvetica",20),fill="green")
                elif grid[i][j]=='P':
                    canvas.create_text(x0 + cell_size/2,y0 + cell_size/2,text='P',font=("Helvetica",20),fill="red")
    status_text_id=None
    def step_action():
        nonlocal state, done, step_count, status_text_id
        if not done and step_count < MAX_STEPS:
            action = agent.select_action(state)
            state, reward, done = env.step(action)
            grid = env.render_grid()
            draw_grid(grid)
            if done:
                if env.agent_pos == env.food_pos:
                    status = "Agent found the food!"
                elif tuple(env.agent_pos) in env.walls:
                    status = "Agent hit the wall!"
                elif env.step_count >= MAX_STEPS:
                    status = "Agent reached step limit!"
                else:
                    status = "Agent ended the episode."
                canvas.create_text(GRID_SIZE * cell_size / 2, GRID_SIZE * cell_size + 15, text=status, font=("Helvetica", 16))
                state = env.reset()
                done = False  # Reset 'done' to False
                step_count = 0
                root.after(SLEEP_TIME, step_action)
            else:
                root.after(SLEEP_TIME, step_action)
            step_count += 1
    grid=env.render_grid()
    draw_grid(grid)
    step_count=0
    done=False
    root.after(SLEEP_TIME,step_action)
    root.mainloop()
if __name__=="__main__":
    main()

