import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from loader import TensorKartDataSet
import sys
import numpy as np



from dqn import DQN
from memory import ReplayBuffer
from utils import process_frame, discrete_action, actionTransform
from model import Network
from IPython import embed
from pretrain import preTrain


pretrain = True
if not pretrain:
    model = Network().cuda()
    epochs = 300
    batch_size = 128
    preTrain(model,epochs,batch_size)

import gym, gym_mupen64plus

env = gym.make('Mario-Kart-Luigi-Raceway-v0')
dqn = DQN(env)
dqn.train()
