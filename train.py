import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from loader import TensorKartDataSet
import sys
import numpy as np

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2))
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1)) 
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(2, 2), stride=(1, 1))
        self.pool2 = nn.MaxPool2d(2)
        self.dense1 = nn.Linear(5 * 8 * 64, 512)
        self.dense2 = nn.Linear(512, 3)
    def forward(self, input):
        output = F.relu(self.conv1(input.transpose(3, 2).transpose(2, 1)))
        output = F.relu(self.conv2(output))
        output = self.pool1(output)
        output = F.relu(self.conv3(output))
        output = F.relu(self.conv4(output))
        output = self.pool2(output)
        output = F.relu(self.dense1(output.view(-1, 5 * 8 * 64)))
        output = self.dense2(output)
        return F.sigmoid(output)

def discrete_action(action):
    d_action = [0, 0, 0]
    if action[0] > 0:
        d_action[0] = 1
    if action[0] < 0:
        d_action[1] = 1
    if action[2] > 0:
        d_action[2] = 1
    return d_action

model = Network().cuda()
epochs = 100
batch_size = 128
x_train, y_train = [], []
for i in range(10):
    x_train.append(np.load('/tmp2/h_data/{}_x.npy'.format(i)))
    y_train.append(np.load('/tmp2/h_data/{}_y.npy'.format(i)))
x_train = np.vstack(x_train)
y_train = np.vstack(y_train)
y_train = np.array([discrete_action(a) for a in y_train])
print(x_train.shape)

train_loader = DataLoader(
        dataset=TensorKartDataSet(x_train,y_train),
        batch_size= batch_size,
        shuffle=True,
        num_workers=8)

optimizer = torch.optim.Adam(model.parameters())
for epoch in range(epochs):
    total_loss = 0
    for i, sample in enumerate(train_loader):
        x_batch, y_batch = Variable(sample["images"]).cuda(), Variable(sample["tags"]).cuda()
        predict = model(x_batch)
        loss = F.binary_cross_entropy(predict, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.data[0]
    print('Epoch:{}, Loss:{}'.format(epoch, total_loss))

print('Save model.')
torch.save(model.state_dict(), 'gg.pt')


model.load_state_dict(torch.load('model.pt'))
import gym, gym_mupen64plus

def process_frame(frame):
    #frame = 0.2126 * frame[:, :, 0] + 0.7152 * frame[:, :, 1] + 0.0722 * frame[:, :, 2]
    frame = frame.astype(np.uint8)[::2, ::2,:]
    return frame


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

