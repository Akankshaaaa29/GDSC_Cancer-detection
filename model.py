# source https://naomi-fridman.medium.com/multi-class-image-segmentation-a5cc671e647a
# import system libs
import os
import time
import random
import pathlib
import itertools
from glob import glob


# import data handling tools

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt



# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate

import keras
import keras.backend as K
from keras.callbacks import CSVLogger
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.layers.experimental import preprocessing

# # import system libs
import os
import time
import random
import pathlib
import itertools
from glob import glob


# import data handling tools

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt



# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate

import keras
import keras.backend as K
from keras.callbacks import CSVLogger
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.layers.experimental import preprocessing

import os
import glob

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.models import *
from keras.layers import *
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot as plt
import keras.backend as K

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model

def dense_block(x, n_layers, growth_rate, ker_init, dropout):
    for i in range(n_layers):
        bn = BatchNormalization()(x)
        act = Activation('relu')(bn)
        conv = Conv2D(growth_rate, 3, padding='same', kernel_initializer=ker_init)(act)
        if dropout:
            conv = Dropout(dropout)(conv)
        x = concatenate([x, conv], axis=-1)
    return x

def build_unet(inputs, ker_init, dropout):
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(inputs)
    dense1 = dense_block(conv1, n_layers=2, growth_rate=32, ker_init=ker_init, dropout=dropout)
    pool1 = MaxPooling2D(pool_size=(2, 2))(dense1)

    dense2 = dense_block(pool1, n_layers=2, growth_rate=64, ker_init=ker_init, dropout=dropout)
    pool2 = MaxPooling2D(pool_size=(2, 2))(dense2)

    dense3 = dense_block(pool2, n_layers=3, growth_rate=128, ker_init=ker_init, dropout=dropout)
    pool3 = MaxPooling2D(pool_size=(2, 2))(dense3)

    dense4 = dense_block(pool3, n_layers=3, growth_rate=256, ker_init=ker_init, dropout=dropout)
    pool4 = MaxPooling2D(pool_size=(2, 2))(dense4)

    dense5 = dense_block(pool4, n_layers=3, growth_rate=512, ker_init=ker_init, dropout=dropout)
    drop5 = Dropout(dropout)(dense5)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(drop5))
    merge7 = concatenate([dense4,up7], axis = 3)
    dense7 = dense_block(merge7, n_layers=3, growth_rate=256, ker_init=ker_init, dropout=dropout)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(dense7))
    merge8 = concatenate([dense3,up8], axis = 3)
    dense8 = dense_block(merge8, n_layers=2, growth_rate=128, ker_init=ker_init, dropout=dropout)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(dense8))
    merge9 = concatenate([dense2,up9], axis = 3)
    dense9 = dense_block(merge9, n_layers=2, growth_rate=64, ker_init=ker_init, dropout=dropout)

    up = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(dense9))
    merge = concatenate([dense1,up], axis = 3)
    dense = dense_block(merge, n_layers=2, growth_rate=32, ker_init=ker_init, dropout=dropout)

    outputs = Conv2D(2, 1, activation = 'sigmoid')(dense)
    model = Model(inputs=[inputs], outputs=[outputs])

    return model




#model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema ,dice_coef_enhancing] )


