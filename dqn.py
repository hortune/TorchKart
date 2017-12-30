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
from random import random, randrange

class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = ReplayBuffer(10000)
        self.steps = 10000000
        self.GAMMA = 0.999
         
        self.online_model = Network().cuda()
        self.online_model.load_state_dict(torch.load('pre-train.pt'))
        self.target_model = Network().cuda()
        self.target_model.load_state_dict(torch.load('pre-train.pt'))
        
        self.exploration = LinearSchedule(1,1.0,0.05) 
    
    def train(self):
        total_reward = 0
        loss = 0
        while True:
            observation = process_frame(self.env.reset()).astype(np.float64) / 255
            done = False
            while not done:
                self.env.render()
                action = self.make_action(observation)
                continuous_action = actionTransform(action)
                print (continuous_action) 
                observation, reward, done, _ = self.env.step(continuous_action)
                observation = process_frame(observation).astype(np.float64) / 255
    def make_action(self,observation,test=False):
        action = self.online_model(Variable(torch.FloatTensor(observation).unsqueeze(0),volatile=True).cuda()).squeeze()
        
        if test:
            eps_threshold = 0.05
        else:
            eps_threshold = 0 # self.exploration.value(self.steps)

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

