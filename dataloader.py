from torchvision import transforms
import random
import torch
import json
import cv2
import os

from earth_env import *
from ViewBox import *


class Dataset():

    def __init__(self, 
                 world_size, 
                 num_workers,
                 imagery_dir, 
                 json_path, 
                 split,
                 gamma = 0.999,
                 eps_start = .9,
                 eps_end = 0.05,
                 eps_decay = 200,
                 target_update = 10):

        print("In dataloader!!")

        self.world_size = world_size
        self.num_workers = num_workers
        self.imagery_dir = imagery_dir
        self.split = split

        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update

        with open(json_path, "r") as f:
            self.ys = json.load(f)

        print("Migration JSON has been loaded!")

        self.data = []
        self.to_tens = transforms.ToTensor()
        self.load_data()
        print("Done loading data in DL!!")
        self.train_val_split()
        print("Done train splitting data in DL!!")


    def load_data(self):

        print("IN DATALOADER'S LOAD DATA FUNCTION!")

        # Load in and prep all of the data
        for impath in os.listdir(self.imagery_dir)[8:21]:

            self.data.append((os.path.join(self.imagery_dir, impath), 
                             self.ys[impath.replace(".png", "")], 
                             [3, 5, False, self.gamma, self.eps_start, self.eps_end, self.eps_decay, self.target_update]
                                      ))

    def train_val_split(self):

        self.batch_size = int(len(self.data) / (self.world_size - 1))

        print("LENGTH OF DATA: ", len(self.data))
        print("BATCH SIZE: ", self.batch_size)

        train_num = int(len(self.data) * self.split)
        train_indices = random.sample(range(len(self.data)), train_num)
        val_indices = [i for i in range(len(self.data)) if i not in train_indices]
        train_data = [self.data[i] for i in train_indices]
        val_data = [self.data[i] for i in val_indices]     
        self.train_data = [train_data[i:i + self.batch_size] for i in range(0, len(train_data), self.batch_size)]
        self.val_data = [val_data[i:i + self.batch_size] for i in range(0, len(val_data), self.batch_size)]

        # Correct for number of workers
        over = (len(self.train_data) - self.num_workers) + 1
        print("OVER: ", over)
        to_index = len(self.train_data) - over
        if over != 0:
            leftover = self.train_data[-over:]
            leftover = [item for sublist in leftover for item in sublist]
            self.train_data = self.train_data[0:to_index]
            self.train_data.append(leftover)



"""
A TENSOR HAS SHAPE B, C, H, W
"""