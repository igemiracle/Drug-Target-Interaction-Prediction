# -*- coding: utf-8 -*-
"""Copy of Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13JWJeU9nC0nd53gCoeijdGJoeJD3x5zP
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install git+https://github.com/bp-kelley/descriptastorus

# Commented out IPython magic to ensure Python compatibility.
# %pip install DeepPurpose

from DeepPurpose import DTI as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *

from google.colab import drive
drive.mount('/content/drive')

SAVE_PATH='/content/drive/MyDrive/CMU/02718/final_project/'
import os
if not os.path.exists(SAVE_PATH):
  os.makedirs(SAVE_PATH)

X_drug, X_target, y = load_process_DAVIS(SAVE_PATH, binary=False)

drug_encoding, target_encoding = 'CNN', 'CNN'

train,val,test=data_process(X_drug,X_target,y,drug_encoding,target_encoding,split_method='random',frac=[0.7,0.1,0.2],random_seed=1)

config=generate_config(drug_encoding,
                       target_encoding,
                       cls_hidden_dims=[1024,1024,512],
                       train_epoch=100,
                       LR=0.001,
                       batch_size=256,
                       cnn_drug_filters=[32,64,96],
                       cnn_drug_kernels=[4,8,12],
                       cnn_target_filters=[32,64,96],
                       cnn_target_kernels=[4,8,12])

net = models.model_initialize(**config)

net.train(train,val,test)