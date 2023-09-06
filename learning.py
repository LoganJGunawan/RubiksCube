import torch.nn as nn
from function import shuffle, newTurn, checkSolved, checkAlign
import numpy as np
import random
from collections import namedtuple
import torch.optim as optim
import torch.nn.functional as F
import math
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

#Format of experiences to insert into our replaybuffer
Experience = namedtuple('Experience',['state','action','reward','next_state','done'])

#Stores and dispenses batches of experience to train our model on
class ReplayBuffer(object):
    def __init__(self,capacity, inputSize):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.inputSize = inputSize
    
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
        self.oneHotEncoding = np.eye(6)

    def shuffle(self):
        self.state = shuffle()

    def encode(self):
        return torch.tensor(self.state)
    
    def checkGoal(self):
        global maxGoal
        h = checkAlign(self.state)
        if h > maxGoal:
            maxGoal = h
            print(f"{maxGoal} GOAL")
            return h

    def turn(self,action):
        reward=-1
        done=False
        self.state=newTurn(self.state,action)
        h = self.checkGoal()
        if h:
            reward+=(30*maxGoal)
        if h==6:
            done=True
            reward+=1000
        return reward, done

class DQN(nn.Module):
    def __init__(self,out,embedding_dim):
        super(DQN, self).__init__()
        self.embedding = nn.Embedding(6*9,embedding_dim)
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=32, kernel_size=3), 
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
        )

        self.feature_layers = nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.Linear(256, out),
            nn.ReLU()
        )

    def forward(self, x):
        x=x.long()
        embedded = self.embedding(x)
        if len(embedded.size())==4:
            embedded = embedded.view(x.size(0),9,6,-1)
            x = self.convolutional_layers(embedded)
            x = x.view(x.size(0),10,640)
            x = self.feature_layers(x)
            x = x.transpose(1,2).sum(dim=2)
        else:
            embedded = embedded.view(x.size(0),9,6,-1)
            x = self.convolutional_layers(embedded)
            x = x.view(x.size(0),-1)
            x = self.feature_layers(x)
            x = x.t().sum()
        return x

def randomTurn():
    global cube, totalReward, done
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
    states = torch.stack([experience.state for experience in batch]).to('cuda')
    actions = torch.tensor([experience.action for experience in batch], dtype=torch.long).to('cuda')
    rewards = torch.tensor([experience.reward for experience in batch], dtype=torch.float32).to('cuda')
    next_states = torch.stack([experience.next_state for experience in batch]).to('cuda')
    dones = torch.tensor([experience.done for experience in batch], dtype=torch.float32).to('cuda')

    # Compute Q-values for current states using the policy network
    q_values = policy_net(states)
    q_values = q_values.gather(1,actions.unsqueeze(1)).squeeze()

    # Compute target Q-values using the target network
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
        target_q_values = rewards + gamma * next_q_values * (1 - dones)

    loss = nn.functional.mse_loss(q_values,target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

if __name__ == "__main__":
    #Training Phase
    cube = Environment()
    replay_buffer = ReplayBuffer(capacity=2000,inputSize=(6,9))
    batchsize = 128
    gamma = 0.99
    epsilon=0.2
    TAU = 0.005
    LR = 1e-8
    episodeLength=150
    numEps=5000
    target_network_update_interval = 512
    numFinish = 0
    maxReward=-500

    policy_net = DQN(10,54).to('cuda')
    target_net = DQN(10,54).to('cuda')
    #policy_net.load_state_dict(torch.load('model.pth'))
    #target_net.load_state_dict(torch.load('target.pth'))
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    episodeDuration = []
    losses = []
    for z in range(1):
        numFinish = 0
        pbar = tqdm(total=numEps, unit="episode", ncols=125)
        maxReward = -500
        for episode in range(numEps):
            stepsDone=0
            totalReward=0
            cube.shuffle()
            done = False
            maxGoal = 0

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
            if totalReward > maxReward:
                maxReward = totalReward
            pbar.set_description(f"Episode: {episode+1}/{numEps} | Finish: {numFinish} | Best Run: {maxReward}")
            pbar.update()
        
        torch.save(policy_net.state_dict(), "model.pth")
        torch.save(target_net.state_dict(),"target.pth")
        print("Finished training and saved model")
        plt.plot(losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()
