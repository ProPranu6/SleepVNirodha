import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
#from torchsummary import summary


import numpy as np
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.preprocessing import OneHotEncoder as OH
from sklearn.utils import shuffle

from numpy.lib.nanfunctions import nanargmin
from math import log
import numpy as np
from math import e
import random
from tensorflow import keras
import pyedflib
#import pyedflibdata_paths_traindata_paths_train
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd
import re
import os


from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split

from keras.models import load_model
from keras.layers import Lambda
import tensorflow as tf


#Preprocessing does affect the channel amplitudes due to removal of baseline, re-referencing, and artifacts removal :https://www.biorxiv.org/content/10.1101/2020.01.20.913327v1.full

from collections import Counter as c
def EEGNet_tensorflow(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:
        
        1. Depthwise Convolutions to learn spatial filters within a 
        temporal convolution. The use of the depth_multiplier option maps 
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn 
        spatial filters within each filter in a filter-bank. This also limits 
        the number of free parameters to fit when compared to a fully-connected
        convolution. 
        
        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions. 
        
    
    While the original paper used Dropout, we found that SpatialDropout2D 
    sometimes produced slightly better results for classification of ERP 
    signals. However, SpatialDropout2D significantly reduced performance 
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.
        
    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the 
    kernel lengths for double the sampling rate, etc). Note that we haven't 
    tested the model performance with this rule so this may not work well. 
    
    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
	advised to do some model searching to get optimal performance on your
	particular dataset.
    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D. 
    Inputs:
        
      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.     
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
    """
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (Chans, Samples, 1))

    ##################################################################(Chans, Samples, 1)
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)  
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),              
                                   use_bias = False, padding = 'same')(block1)   
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)  
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)




class TrainingEEG():
  def __init__(self, model, model_params=None, data_paths=None, train_split_per_recording=0.8, shuffle_before_record_split=False, channels=64, scale=1, data_resample=0, data_time_steps=0, sever_model_top=False, out_classes=None):
    self.data_paths = data_paths
    self.scale = scale
    self.data_resample = data_resample
    self.data_time_steps = data_time_steps
    self.channels = channels
    self.train_split_per_recording = train_split_per_recording
    self.shuffle_before_record_split=shuffle_before_record_split
    self.X = None
    self.Y = None
    self.X_test = None
    self.Y_test = None
    self.X_test_in_rec_partitions = []
    self.Y_test_in_rec_partitions = []
    self.model = model
    self.sever_model_top = sever_model_top
    self.model_params = model_params
    self.out_classes = out_classes
    self.test_loaderz = (False, None)
    self.train_losses = []
    self.val_losses = []
    self.data_generators_info = None

  
  def organise_data(self):

    xtrain, xtest, ytrain, ytest = self.pullin()
    xtrain, ytrain = shuffle(xtrain, ytrain)
    xtest, ytest = shuffle(xtest, ytest)
    
    self.X, self.Y, self.X_test, self.Y_test = xtrain, ytrain, xtest, ytest


  def pullin(self):
    dataxtrain = []
    dataytrain = []
    dataxtest = []
    dataytest = []
    
    
    data_paths = self.data_paths
    
    classes = len(data_paths)//2 if len(data_paths)%2 == 0 else len(data_paths)
    self.out_classes = classes
    for clsid, path in enumerate(data_paths):
      raw = mne.io.read_raw_bdf(f'{path}', infer_types=True)
      if self.data_resample:
        raw = raw.resample(self.data_resample)
      x, t = raw[:self.channels, :300*self.data_resample]  #considering only 300 seconds of recorded data pruning the extra 1 inconsistent seconds
      x = x[:, :300*self.data_resample]
        
      if self.data_time_steps:
        ts = self.data_time_steps
      else:
        ts = raw.Info().sfreq
      
      x = x[:, :300*self.data_resample-(300*self.data_resample)%self.data_time_steps] #pruning extra data points that are left over after slicing time windows
      x = x.reshape(-1, self.channels,ts) 
      x *= self.scale
      y = [clsid%classes]*x.shape[0]
      
      recxtrain, recxtest, recytrain, recytest = train_test_split(x, y, train_size=self.train_split_per_recording, shuffle=self.shuffle_before_record_split)
      
      dataxtrain += [recxtrain]
      dataxtest += [recxtest]
      dataytrain += recytrain
      dataytest += recytest
      self.X_test_in_rec_partitions += [recxtest]
      self.Y_test_in_rec_partitions += [recytest]
    return np.concatenate(dataxtrain,axis=0, dtype=np.float32), np.concatenate(dataxtest,axis=0, dtype=np.float32), np.array(dataytrain), np.array(dataytest)

  

  def train(self, epochs, batch_size, progressive_data_load=False, generator_info={'train_generator':None, 'test_generator':None, 'steps_per_train_epoch':None, 'steps_per_validation_epoch':None}, lr=1e-3, class_weights=None, callbacks=None, ):
    model = self.model
    model.compile(Adam(learning_rate=lr), 'sparse_categorical_crossentropy', 'accuracy')
    if progressive_data_load:
        history = model.fit(generator_info['train_generator'], epochs=epochs, batch_size=batch_size, validation_data=generator_info['test_generator'], callbacks=callbacks, steps_per_epoch=generator_info['steps_per_train_epoch'], validation_steps = generator_info['steps_per_test_epoch'])
    else:    
        history = model.fit(self.X, self.Y, epochs=epochs, batch_size=batch_size, validation_data=(self.X_test, self.Y_test), callbacks=callbacks)
    self.model = model

    return history



      
    
  def __call__(self, train=True, epochs=100, batch_size=16, progressive_data_load = False, generator_info={'train_generator':None, 'test_generator':None, 'steps_per_train_epoch':None, 'steps_per_test_epoch':None}, class_weights=None, callbacks=None, lr=1e-3):

      K.clear_session()
      if self.model_params:
        self.model.load_weights(self.model_params)  

        print("="*20+"Prepping Pretrained Model"+"="*20)
        print("PROPERTIES TO REMAIN")
        print("Pretrained Data Channels : 60\t\t Pretrained Data Sampling Rate : 151Hz")
        print("Pretrained Data Time Steps : 151\t\t Pretrained Model Kernel Length : 32")
        print("Pretrained Model Optimizer  : Adam\t\t Pretrained Model Learning Rate : 0.001")
      
        if self.model_params and self.sever_model_top:
          self.model = self.manage_freezing(self.model)


        print("="*20+"Fine Tuned Model Architecture"+"="*20) 
      else:
        print("="*20+"Model Architecture"+"="*20) 

      print(self.model.summary())

      print("="*20+"Organising Data"+"="*20) 

      if type(self.X) == type(None) and not(progressive_data_load):
        self.organise_data()
        
      if not(progressive_data_load):
          print("Train X : ", self.X.shape)
          print("Train Y : ", self.Y.shape)
          print("Test  X : ", self.X_test.shape) 
          print("Test  Y : ", self.Y_test.shape)

      
      if train:
          print("="*20+"Training Model"+"="*20)
          
          self.data_generators_info = generator_info
          history = self.train(epochs, batch_size, progressive_data_load=progressive_data_load, generator_info=generator_info, class_weights=class_weights, callbacks=callbacks, lr=lr)
          

          print("="*20+"Training Stats"+"="*20) 
          
          plt.subplot(1,2,1)
          plt.plot(range(len(history.history['loss'])), history.history['loss'], color='black')
          plt.plot(range(len(history.history['val_loss'])), history.history['val_loss'], color='orange')
          plt.xlabel("Epoch")
          plt.ylabel("Loss")
          plt.legend(["training loss", "validation loss"])
          
          plt.subplot(1,2,2)
          plt.plot(history.history['accuracy'], color="black")
          plt.plot(history.history['val_accuracy'], color="orange")
          plt.title('model accuracy')
          plt.ylabel('Accuracy')
          plt.xlabel('Epoch')
          plt.legend(['training accuracy', 'validation accuracy'])
        
      return self.model
  
  def get_stats(self, mode='per_sample_acc'):
        """
        Parameters :
        mode : 'per_sample_acc'/'per_class_acc'/'per_recording_acc'/'per_recording_vot'/'every'
        per_sample_acc = (all the correct samples/total samples)
        per_class_acc = (all the correct samples per class/total samples in that class)
        per_recording_acc = (all the correct samples per recording file/total samples in that recording file)
        per_recording_vot = (votes for different classes per recording file)
        """
        if type(self.data_generators_info) != type(None):
            return "This method doesn't support generator input"
        
        if mode == 'per_sample_acc':
            _, acc = self.model.evaluate(self.X_test, self.Y_test) 
            return acc
        elif mode == "per_class_acc":
            accs = []
            for clasind in range(0, self.out_classes):
                x = np.concatenate([self.X_test_in_rec_partitions[xs] for xs in [clasind, clasind+self.out_classes] ], axis=0)
                yss = [] 
                for ys in [clasind, clasind+self.out_classes]:
                    yss += self.Y_test_in_rec_partitions[ys]
                y = np.array(yss)
                
                _, acc = self.model.evaluate(x, y)
                accs.append(acc)
            return accs
        elif mode == "per_rec_acc":
            accs = []
            for clasind in range(0, len(self.data_paths)):
                x = self.X_test_in_rec_partitions[clasind]
                y = np.array(self.Y_test_in_rec_partitions[clasind])
                
                _, acc = self.model.evaluate(x, y)
                accs.append(acc)
            return accs
        else:
            vots = []
            for clasind in range(0, len(self.data_paths)):
                x = self.X_test_in_rec_partitions[clasind]
                y = np.array(self.Y_test_in_rec_partitions[clasind])
                
                vot = np.argmax(self.model.predict(x), axis=-1)
                vots.append(c(vot))
            return vots
            
                
                
            
  
  def manage_freezing(self, prmodel):
    total_layers = len(prmodel.layers)
    xinp = prmodel.input
    x = xinp
    for l in range(1, total_layers-2):
        x = prmodel.layers[l](x)
    xout = Dense(self.out_classes, activation='softmax')(x)
    ftmodel = Model(inputs=xinp, outputs=xout)
    for layer in range(len(ftmodel.layers)-1):
        ftmodel.layers[layer].trainable = False
      
    return ftmodel
  
  

class TrainingSleepVNirodha(TrainingEEG):
    def __init__(self, limit_data_seconds=300, *args, **kwargs):
        self.limit_data_seconds = limit_data_seconds
        super().__init__(*args, **kwargs)
    
    def pullin(self):
        dataxtrain = []
        dataytrain = []
        dataxtest = []
        dataytest = []


        data_paths = self.data_paths

        classes = len(data_paths)
        self.out_classes = classes
        for clsid, path in enumerate(data_paths):
          raw = mne.io.read_raw_eeglab(f'{path}') # mne.io.read_raw_bdf(f'{path}', infer_types=True)
          if self.data_resample:
            raw = raw.resample(self.data_resample)
          
          if self.data_time_steps:
            ts = self.data_time_steps
          else:
            ts = raw.Info().sfreq
          
          limited_time_steps = min(self.limit_data_seconds*self.data_resample, raw.n_times)
          x, t = raw[:self.channels, :limited_time_steps]  #(raw.n_times//self.data_time_steps) #considering only n seconds of recorded data that can be split into specified time window size 
          x = x[:, :limited_time_steps]

          

          x = x[:, :limited_time_steps-(limited_time_steps)%self.data_time_steps] #pruning extra data points that are left over after slicing time windows
          x = x.reshape(-1, self.channels,ts) 
          x *= self.scale
          y = [clsid%classes]*x.shape[0]

          recxtrain, recxtest, recytrain, recytest = train_test_split(x, y, train_size=self.train_split_per_recording, shuffle=self.shuffle_before_record_split)

          dataxtrain += [recxtrain]
          dataxtest += [recxtest]
          dataytrain += recytrain
          dataytest += recytest
          self.X_test_in_rec_partitions += [recxtest]
          self.Y_test_in_rec_partitions += [recytest]
        return np.concatenate(dataxtrain,axis=0, dtype=np.float32), np.concatenate(dataxtest,axis=0, dtype=np.float32), np.array(dataytrain), np.array(dataytest)
    
    def get_stats(self, mode='per_sample_acc'):
        """
        Parameters :
        mode : 'per_sample_acc'/'per_class_acc'/'per_recording_acc'/'per_recording_vot'/'every'
        per_sample_acc = (all the correct samples/total samples)
        per_class_acc = (all the correct samples per class/total samples in that class)
        per_recording_acc = (all the correct samples per recording file/total samples in that recording file)
        per_recording_vot = (votes for different classes per recording file)
        """
        if mode == 'per_sample_acc':
           _, acc = self.model.evaluate(self.X_test, self.Y_test) 
           return acc
        elif mode == "per_class_acc":
            accs = []
            for clasind in range(0, self.out_classes):
                x = self.X_test_in_rec_partitions[clasind]
                y = np.array(self.Y_test_in_rec_partitions[clasind])
                
                _, acc = self.model.evaluate(x, y)
                accs.append(acc)
            return accs
        elif mode == "per_rec_acc":
            accs = []
            for clasind in range(0, len(self.data_paths)):
                x = self.X_test_in_rec_partitions[clasind]
                y = np.array(self.Y_test_in_rec_partitions[clasind])
                
                _, acc = self.model.evaluate(x, y)
                accs.append(acc)
            return accs
        else:
            vots = []
            for clasind in range(0, len(self.data_paths)):
                x = self.X_test_in_rec_partitions[clasind]
                y = np.array(self.Y_test_in_rec_partitions[clasind])
                
                vot = np.argmax(self.model.predict(x), axis=-1)
                vots.append(c(vot))
            return vots
      

class EEGAsDatasetCreator:
    def __init__(self, dataset_dir, limit_data_seconds=None, data_resample=None, channels=None, data_time_steps=None, data_paths=None, train_split_per_recording=0.8, shuffle_before_record_split=False, scale=1e9):
        self.data_resample = data_resample
        self.channels = channels
        self.scale = scale
        self.data_time_steps = data_time_steps
        self.train_split_per_recording = train_split_per_recording
        self.data_paths = data_paths
        self.limit_data_seconds = limit_data_seconds
        self.shuffle_before_record_split = shuffle_before_record_split
        self.pool_data_in_directories(dataset_dir)
        
        
    def pool_data_in_directories(self, dataset_dir):
        
        try:
            os.mkdir(dataset_dir)
        except e:
            print(e)
        
        train_dir = dataset_dir + '/train'
        test_dir = dataset_dir + '/test'
        
        
        try:    
            os.mkdir(train_dir)
            os.mkdir(test_dir)
        except e:
            print(e)
        
        
        dataxtrain = []
        dataytrain = []
        dataxtest = []
        dataytest = []


        data_paths = self.data_paths

        classes = len(data_paths)
        for clsid, path in enumerate(data_paths):
          raw = mne.io.read_raw_eeglab(f'{path}') # mne.io.read_raw_bdf(f'{path}', infer_types=True)
          if self.data_resample:
            raw = raw.resample(self.data_resample)
          
          if self.data_time_steps:
            ts = self.data_time_steps
          else:
            ts = raw.Info().sfreq
          
          limited_time_steps = min(self.limit_data_seconds*self.data_resample, raw.n_times)
          x, t = raw[:self.channels, :limited_time_steps]  #(raw.n_times//self.data_time_steps) #considering only n seconds of recorded data that can be split into specified time window size 
          x = x[:, :limited_time_steps]

          
          x = x[:, :limited_time_steps-(limited_time_steps)%self.data_time_steps] #pruning extra data points that are left over after slicing time windows
          x = x.reshape(-1, self.channels,ts) 
          x *= self.scale
          y = np.array([clsid%classes]*x.shape[0])

          recxtrain, recxtest, _, _ = train_test_split(x, y, train_size=self.train_split_per_recording, shuffle=self.shuffle_before_record_split)
          for train_instance in range(recxtrain.shape[0]):
            with open(train_dir + f'/{clsid}_{train_instance}.npy', 'wb') as train_file:
                np.save(train_file, recxtrain[train_instance], allow_pickle=False)
          for test_instance in range(recxtest.shape[0]):  
            with open(test_dir + f'/{clsid}_{test_instance}.npy', 'wb') as test_file:
                np.save(test_file, recxtest[test_instance],  allow_pickle=False)
          
          del recxtrain
          del recxtest
          del raw
          del x
          del t
          del y
        
        return dataset_dir
    
    @staticmethod
    def _batch_generator(fldr, batch_size = 10):
        batch=[]
        ids = np.arange(len(os.listdir(fldr)))
        np.random.shuffle(ids) 
        while True:
                for i in ids:
                    batch.append(i)
                    if len(batch)==batch_size:
                        yield EEGAsDatasetCreator._load_data(fldr, batch)
                        batch=[]
    
    @staticmethod
    def _load_data(fldr, ids):
       X = []
       Y = []

       for i in ids:
         file_name = os.listdir(fldr)[i]
         x = np.load(fldr + f'/{file_name}')
         y = int(re.search("([0-9])_",file_name).group(1))

         X.append(x)
         Y.append(y)

       return np.array(X), np.array(Y)

    def __call__(self, fldr, batch_size=32):
        data_generator = EEGAsDatasetCreator._batch_generator(fldr, batch_size)
        return data_generator
