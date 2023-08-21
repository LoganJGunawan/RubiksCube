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
        one_hot_encoded = np.empty(self.state.shape + (6,))

        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                    color_value = self.state[i,j]
                    one_hot_encoded[i,j] = self.oneHotEncoding[color_value]
        
        return torch.tensor(one_hot_encoded)
    
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
    def __init__(self,inp,out):
        super(DQN, self).__init__()
        
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(in_channels=inp, out_channels=32, kernel_size=3),  # Use padding to maintain size
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
        )

        self.feature_layers = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(),
            nn.Linear(256, out),
            nn.ReLU()
        )

    def forward(self, x):
        x=x.float()
        x = self.convolutional_layers(x)
        if(x.size(1)==5):
            x=torch.flatten(x,start_dim=1)
        else:
            x=torch.flatten(x,start_dim=2)
        x = self.feature_layers(x)
        print(x)
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
    states = torch.stack([experience.state for experience in batch]).to('cuda')
    actions = torch.tensor([experience.action for experience in batch], dtype=torch.long).to('cuda')
    rewards = torch.tensor([experience.reward for experience in batch], dtype=torch.float32).to('cuda')
    next_states = torch.stack([experience.next_state for experience in batch]).to('cuda')
    dones = torch.tensor([experience.done for experience in batch], dtype=torch.float32).to('cuda')

    # Compute Q-values for current states using the policy network
    q_values = policy_net(states)
    print(actions.size())

    # Compute target Q-values using the target network
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
        next_q_values = next_q_values.squeeze()
        rewards=rewards.unsqueeze(1)
        print(next_q_values)
        print("Shapes before multiplication:")
        print("rewards shape:", rewards.shape)
        print("next_q_values shape:", next_q_values.shape)
        target_q_values = rewards + gamma * next_q_values * (1 - dones)

    # Calculate the loss and perform gradient descent
    loss = nn.functional.smooth_l1_loss(q_values, target_q_values.unsqueeze(1))
    print(f"\n{loss.item()}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__ == "__main__":
    #Training Phase
    cube = Environment()
    replay_buffer = ReplayBuffer(capacity=2000,inputSize=(6,9))
    batchsize = 256
    gamma = 0.99
    epsilon=0.15
    TAU = 0.005
    LR = 1e-4
    episodeLength=150
    numEps=1000
    target_network_update_interval = 200
    numFinish = 0
    maxReward=-500

    policy_net = DQN(6,10).to('cuda')
    target_net = DQN(6,10).to('cuda')
    #policy_net.load_state_dict(torch.load('model.pth'))
    #target_net.load_state_dict(torch.load('target.pth'))
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    episodeDuration = []
    pbar = tqdm(total=numEps, unit="episode", ncols=125)

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
