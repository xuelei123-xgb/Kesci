import warnings
warnings.filterwarnings("ignore")
import os
import gc
from tqdm import tqdm
import joblib
import pandas as pd
import numpy as np
from scipy.signal import resample
from collections import OrderedDict, namedtuple, defaultdict
from itertools import chain
from copy import copy,deepcopy
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics, preprocessing
# import keras
import  random
import os
import tensorflow.keras as keras
from imblearn.over_sampling import  RandomOverSampler,SMOTE


data_dir = 'E:/2020 xw_bank/'
train_data_path=data_dir+"sensor_train.csv"
test_data_path=data_dir+"sensor_test.csv"

mean=[ 8.03889039e-03, -6.41381949e-02,  2.37856977e-02,  8.64949391e-01,
       2.80964889e+00,  7.83041714e+00,  6.44853358e-01,  9.78580749e+00,]
std=[0.6120893,  0.53693888, 0.7116134,  3.22046385, 3.01195336, 2.61300056,0.87194132, 0.68427254,]



class DataGenerator(keras.utils.Sequence):
    def __init__(self,dataset, batch_size=32,sample_weight=False,
                classes=19, shuffle=True,  update_after_epoch=True,**kwargs):
        self.dataset=dataset
        self.batch_size = batch_size
        self.sample_weight = sample_weight
        self.dim = tuple(dataset.dim)
        if isinstance(classes, int):
            self.n_classes = classes
            self.classes = [i for i in range(classes)]
        elif isinstance(classes, list):
            self.n_classes = len(classes)
            self.classes = classes
        self.shuffle = shuffle
        self.update_after_epoch=update_after_epoch
        self.indexes = np.arange(len(self.dataset))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)
    def on_epoch_end(self):
        '''Applies augmentation and updates indexes after each epoch'''

        self.dataset.enchance_train_data() # wslsdx 20200724
        if self.update_after_epoch:
            self.indexes = np.arange(len(self.dataset))

        if self.shuffle is True:
            np.random.shuffle(self.indexes)
    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        '''Generate one batch of data'''
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X,Y = [],[]
        for k, index in enumerate(indexes):
            x,y=self.dataset[index]
            X.append(x)
            Y.append(y)
        return (np.array(X), np.array(Y))




    # xx,yy=train_data.get_valid_data(2)
    #
    # xx,yy=train_data.get_valid_data(3)
    #
    # xx,yy=train_data.get_valid_data(4)



