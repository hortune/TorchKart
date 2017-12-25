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


class DQN:
    def __init__(self, env, model):
        self.env = env
        self.model = model
    
    def train(self):
        while True:
            observation = process_frame(self.env.reset()).astype(np.float64) / 255
            done = False
            while not done:
                action = self.model(Variable(torch.FloatTensor(observation).unsqueeze(0)).cuda()).squeeze().data.cpu().numpy()
                action = action.round()
                continuous_action = actionTransform(action)
                print(continuous_action)
                
                observation, reward, done, _ = self.env.step(continuous_action)
                observation = process_frame(observation).astype(np.float64) / 255
