import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from loader import TensorKartDataSet
import sys
import numpy as np


from utils import process_frame, discrete_action
from model import Network
from IPython import embed
from pretrain import preTrain

model = Network().cuda()
epochs = 200
batch_size = 128
pretrain = False

if not pretrain:
    preTrain(model,epochs,batch_size)
else:
    model.load_state_dict(torch.load('pre-train.pt'))

import gym, gym_mupen64plus

env = gym.make('Mario-Kart-Luigi-Raceway-v0')
observation = process_frame(env.reset()).astype(np.float64) / 255
done = False
while not done:
    env.render()
    action = model(Variable(torch.FloatTensor(observation).unsqueeze(0)).cuda()).squeeze().data.cpu().numpy()
    action = action.round()
    continuous_action = [0, 0, 0, 0, 0]
    if action[0] > 0:
        continuous_action[0] = 60
    if action[1] > 0:
        continuous_action[0] = -60
    if action[2] > 0:
        continuous_action[2] = 1
    print(continuous_action)
    observation, reward, done, _ = env.step(continuous_action)
    observation = process_frame(observation).astype(np.float64) / 255
