# from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn as nn
from utils import load_pickle_file, save_pickle_file
import pickle
import sys



def noise_generator():
    # Gaussian noise generation
    e = sys.float_info.epsilon
    g_noise= torch.randn(1, 80, 80)

    n_min, n_max = g_noise.min(), g_noise.max()
    new_min, new_max = 0, 1
    min_max_g_noise = (g_noise - n_min)/(n_max - n_min)*(new_max - new_min) + new_min
    min_max_g_noise = np.log(min_max_g_noise+e) * 1e-8
    
    return min_max_g_noise


class MelDataset(Dataset):
    def __init__(self, root_x, root_y, transform=None):
        self.root_x = root_x
        self.root_y = root_y
        self.transform = transform

        self.x_mels = os.listdir(root_x)
        self.y_mels = os.listdir(root_y)
        
        self.length_dataset = max(len(self.x_mels), len(self.y_mels)) 
        self.x_len = len(self.x_mels)
        self.y_len = len(self.y_mels)

    def __len__(self):
        
        return self.length_dataset

    def __getitem__(self, index):
        x_mel = self.x_mels[index % self.x_len]
        y_mel = self.y_mels[index % self.y_len]

        x_path = os.path.join(self.root_x, x_mel)
        y_path = os.path.join(self.root_y, y_mel)

        x_mel = load_pickle_file(x_path)
        y_mel = load_pickle_file(y_path)

        x_mel = torch.tensor(x_mel) # may not be necessary
        y_mel = torch.tensor(y_mel) # may not be necessary
        
        x_mel = x_mel.reshape(1, 80, 80)
        y_mel = y_mel.reshape(1, 80, 80)
        
        # ### for adding gaussian noise to input mels
        # additional_channel_x_1 = x_mel + noise_generator()
        # additional_channel_x_2 = x_mel + noise_generator()
        # additional_channel_y_1 = y_mel + noise_generator()
        # additional_channel_y_2 = y_mel + noise_generator()
        
        # x_mel = torch.cat([x_mel, additional_channel_x_1], dim=0)
        # x_mel = torch.cat([x_mel, additional_channel_x_2], dim=0)

        # y_mel = torch.cat([y_mel, additional_channel_y_1], dim=0)
        # y_mel = torch.cat([y_mel, additional_channel_y_2], dim=0)

        return x_mel, y_mel


if __name__ == "__main__":
    pass