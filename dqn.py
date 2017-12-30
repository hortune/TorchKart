import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from loader import TensorKartDataSet
import sys
import numpy as np

from memory import ReplayBuffer
from utils import process_frame, discrete_action, actionTransform
from model import Network
from IPython import embed
from pretrain import preTrain
import gym, gym_mupen64plus
from random import random, randrange, sample
from collections import deque

class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=10000)
        self.steps = 10000000
        self.GAMMA = 0.999
        self.batch_size = 64
         
        self.online_model = Network().cuda()
        self.online_model.load_state_dict(torch.load('pre-train.pt'))
        self.target_model = Network().cuda()
        self.target_model.load_state_dict(torch.load('pre-train.pt'))
        
        self.exploration = LinearSchedule(100000,0.01,1.0) 
        self.optimizer = torch.optim.RMSprop(self.online_model.parameters(), lr=1e-4)

    def update(self):
        if len(self.memory) < self.batch_size:
            return 0
        batch = sample(self.memory, self.batch_size)
        batch_state, batch_next, batch_action, batch_reward, batch_done = zip(*batch)
        
        batch_state = Variable(torch.stack(batch_state)).cuda().squeeze()
        batch_next = Variable(torch.stack(batch_next)).cuda().squeeze()
        batch_action = Variable(torch.stack(batch_action)).cuda()
        batch_reward = Variable(torch.stack(batch_reward)).cuda()
        batch_done = Variable(torch.stack(batch_done)).cuda()
        
        current_q = self.online_model(batch_state).gather(1, batch_action)
        next_q = batch_reward + (1 - batch_done) * self.GAMMA * self.target_model(batch_next).detach().max(-1)[0].unsqueeze(-1)
 
        self.optimizer.zero_grad()
        loss = F.mse_loss(current_q, next_q)
        loss.backward()
        self.optimizer.step()
    
    def train(self):
        loss = 0
        for e in range(1000000):
            total_reward = 0
            step = 0
            race = 0
            observation = process_frame(self.env.reset()).astype(np.float64) / 255
            done = False
            while not done:
                self.env.render()
                action = self.make_action(observation)
                next_observation, reward, done, _ = self.env.step(actionTransform(action))
                next_observation = process_frame(next_observation).astype(np.float64) / 255
                total_reward += reward
                step += 1
                if step % 5 == 0:
                    self.memory.append((
                        torch.FloatTensor([observation]),
                        torch.FloatTensor([next_observation]),
                        action.data,
                        torch.FloatTensor([reward]),
                        torch.FloatTensor([done])))
                    observation = next_observation
                if step >= 500 and step % 20 == 0:
                    self.update()
                if step % 5000 == 0:
                    self.target_model.load_state_dict(self.online_model.tate_dict())
                if reward > 0:
                    race += 1
                if race == 2:
                    break
            print('Episode:{} step:{} reward:{}'.format(e, step, total_reward))

    def make_action(self,observation,test=False):
        action = self.online_model(Variable(torch.FloatTensor(observation).unsqueeze(0),volatile=True).cuda()).squeeze()
        
        if test:
            eps_threshold = 0.05
        else:
            eps_threshold = 0.01# self.exploration.value(self.steps)

        if random() > eps_threshold:
            action = action.max(0)[1]
        else:
            action = Variable(torch.cuda.LongTensor([randrange(4)]))
        return action 

class LinearSchedule:
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p
    
    def value(self, t):
        return self.initial_p + min(float(t) / self.schedule_timesteps,1.0) * (self.final_p - self.initial_p)

