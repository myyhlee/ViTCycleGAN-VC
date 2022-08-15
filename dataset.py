# from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn as nn
from utils import load_pickle_file, save_pickle_file
import pickle


            #############################################################
            ############## Dataset function for fully concatnated data ################
            #############################################################


# class MelDataset(Dataset):
#     def __init__(self, root_x, root_y, transform=None):
#         self.root_x = root_x
#         self.root_y = root_y
#         self.transform = transform

#         self.x_mels = load_pickle_file(root_x)
#         self.y_mels = load_pickle_file(root_y)
        
#         self.length_dataset = max(len(self.x_mels), len(self.y_mels)) # 1000, 1500
#         self.x_len = len(self.x_mels)
#         self.y_len = len(self.y_mels)

#     def __len__(self):
#         return self.length_dataset


#     def __getitem__(self, index):

#         x_data = []
#         y_data = []
        
#         num1 = 0
#         num2 = 80
#         while num2 < self.x_mels[1].shape[0]:
#             x_data.append(self.x_mels[:, num1:num2])
#             num1 += 81
#             num2 += 81


#         num1 = 0
#         num2 = 80
#         while num2 < self.y_mels[1].shape[0]:
#             y_data.append(self.y_mels[:, num1:num2])
#             num1 += 81
#             num2 += 81

#         x_mel = x_data[index]
        
#         y_mel = y_data[index]

#         return x_mel.reshape(1, 80, 80), y_mel.reshape(1, 80, 80)



            #############################################################
            ############## Dataset function for sep. data ###############
            #############################################################


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


        return x_mel.reshape(1, 80, 80), y_mel.reshape(1, 80, 80)