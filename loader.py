import os
import io
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F



class TensorKartDataSet(Dataset):
    def __init__(self,images,tags):
        self.images = images
        self.tags = tags

    def __len__(self):
        return len(self.images)


    def __getitem__(self,index):
        image = self.images[index].astype(np.float32)/255
        tag = self.tags[index].astype(np.float32)
        
        return {"images" : image,
                "tags" : tag}
