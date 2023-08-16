import torch.nn as nn
from function import shuffle, newTurn, checkSolved
import numpy as np
import random
from collections import namedtuple
import torch.optim as optim
import torch.nn.functional as F
import math
import torch
from tqdm import tqdm

#Format of experiences to insert into our replaybuffer
Experience = namedtuple('Experience',['state','action','reward','next_state','done'])

#Stores and dispenses batches of experience to train our model on
class ReplayBuffer(object):
    def __init__(self,capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def addExperience(self,experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
    
    def sampleBatch(self,batch_size):
        batch = random.sample(self.buffer,batch_size)
        return batch
    
    def __len__(self):
        return len(self.buffer)


class Environment():
    def __init__(self):
        self.state = shuffle()

    def shuffle(self):
        self.state = shuffle()

    def encode(self):
        encode=[]
        for a in range(6):
            encode.append(self.state[a].flatten())
        return torch.tensor(np.concatenate(encode),dtype=torch.float32)
    
    def turn(self,action):
        reward=-1
        done=False
        self.state=newTurn(self.state,action)
        if checkSolved(self.state):
            done=True
            reward=300
        return reward, done

class DQN(nn.Module):
    def __init__(self,inp,out):
        super(DQN, self).__init__()
        self.feature_layers = nn.Sequential(
            nn.Linear(inp, 256),
            nn.ReLU(),
            nn.Linear(256, out),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.feature_layers(x)
        return x


def randomTurn():
    global cube, totalReward
    cubeState = cube.encode().to('cuda')
    action = random.randint(0,9)
    reward,done = cube.turn(action)
    totalReward+=reward
    newState = cube.encode().to('cuda')
    newExperience = Experience(cubeState,action,reward,newState,done)
    return newExperience

#Funny Osu! Reference
def makeAMove():
    global stepsDone, totalReward, done, cube
    sample=random.random()
    stepsDone+=1
    state = cube.encode().to('cuda')
    if sample > epsilon:
        #Exploitation
        with torch.no_grad():
            pns=policy_net(state)
            move=pns.argmax().item()
            cubeState = cube.encode().to('cuda')
            reward,done = cube.turn(move)
            newState = cube.encode().to('cuda')
            totalReward+=reward
            newExperience = Experience(cubeState,move,reward,newState,done)
            return newExperience
    else:
        #Exploration
        return randomTurn()

def update_policy_network(policy_net, target_net, optimizer, batch, gamma):
    states = torch.tensor([experience.state for experience in batch], dtype=torch.float32).to('cuda')
    actions = torch.tensor([experience.action for experience in batch], dtype=torch.long).to('cuda')
    rewards = torch.tensor([experience.reward for experience in batch], dtype=torch.float32).to('cuda')
    next_states = torch.tensor([experience.next_state for experience in batch], dtype=torch.float32).to('cuda')
    dones = torch.tensor([experience.done for experience in batch], dtype=torch.float32).to('cuda')

    # Compute Q-values for current states using the policy network
    q_values = policy_net(states).gather(1, actions.unsqueeze(1))

    # Compute target Q-values using the target network
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
        target_q_values = rewards + gamma * next_q_values * (1 - dones)

    # Calculate the loss and perform gradient descent
    loss = nn.functional.smooth_l1_loss(q_values, target_q_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__ == "__main__":
    print("here")
    #Training Phase
    cube = Environment()
    replay_buffer = ReplayBuffer(capacity=100)
    
    #Training Phase 2
    batchsize = 128
    gamma = 0.99
    epsilon=0.1
    TAU = 0.005
    LR = 1e-4
    episodeLength=300
    numEps=5000
    target_network_update_interval = 200
    numFinish = 0

    policy_net = DQN(54,10).to('cuda')
    policy_net.load_state_dict(torch.load('model.pth'))
    target_net = DQN(54,10).to('cuda')
    target_net.load_state_dict(torch.load('target.pth'))
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    episodeDuration = []
    pbar = tqdm(total=numEps, unit="episode", ncols=125)

    for episode in range(numEps):
        stepsDone=0
        totalReward=0
        cube.shuffle()
        done = False

        for step in range(episodeLength):
            ep=makeAMove()
            replay_buffer.addExperience(ep)

            if len(replay_buffer)>=batchsize:
                batch = replay_buffer.sampleBatch(batchsize)
                update_policy_network(policy_net,target_net,optimizer,batch,gamma)
            
            if done:
                numFinish+=1
                break

            if episode % target_network_update_interval == 0:
                target_net.load_state_dict(policy_net.state_dict())
                target_net.eval()
            
        pbar.set_description(f"Episode: {episode+1}/{numEps} | Total Reward: {totalReward} | Finish: {numFinish}")
        pbar.update()
    
    torch.save(policy_net.state_dict(), "model.pth")
    torch.save(target_net.state_dict(),"target.pth")
    print("Finished training and saved model")
