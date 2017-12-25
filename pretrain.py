import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from loader import TensorKartDataSet
import sys
import numpy as np


from utils import process_frame, discrete_action


def preTrain(model,epochs,batch_size):
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
    torch.save(model.state_dict(), 'pre-train.pt')
