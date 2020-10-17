#!/usr/bin/env python
# coding: utf-8

# In[]:

import gc
import os
from pathlib import Path
import random
import sys

from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import scipy as sp


import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import display, HTML

# --- plotly ---
from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
pio.templates.default = "plotly_dark"

# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# --- setup ---
pd.set_option('max_columns', 50)


# In[]:


import zarr

import l5kit
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset

from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR
from l5kit.geometry import transform_points
from tqdm import tqdm
from collections import Counter
from l5kit.data import PERCEPTION_LABELS
from prettytable import PrettyTable

from matplotlib import animation, rc
from IPython.display import HTML

rc('animation', html='jshtml')
print("l5kit version:", l5kit.__version__)


# In[]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error


# In[]:


import time
from datetime import datetime


# In[]:


os.environ["L5KIT_DATA_FOLDER"] = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"


# In[]:


dm = LocalDataManager()
dataset_path = dm.require('scenes/sample.zarr')
zarr_dataset = ChunkedDataset(dataset_path)
zarr_dataset.open()
print(zarr_dataset)


# In[]:


print(zarr_dataset.agents)
print(zarr_dataset.agents.shape)
n = zarr_dataset.agents.shape


# In[]:


# helper to convert a timedelta to a string (dropping milliseconds)
def deltaToString(delta):
    timeObj = time.gmtime(delta.total_seconds())
    return time.strftime('%H:%M:%S', timeObj)

class ProgressBar:
    
    # constructor
    #   maxIterations: maximum number of iterations
    def __init__(self, maxIterations):
        self.maxIterations = maxIterations
        self.granularity = 100 # 1 whole percent
    
    # start the timer
    def start(self):
        self.start = datetime.now()
    
    # check the progress of the current iteration
    #   # currentIteration: the current iteration we are on
    def check(self, currentIteration, chunked=False):
        if currentIteration % round(self.maxIterations / self.granularity) == 0 or chunked:
            
            percentage = round(currentIteration / (self.maxIterations - self.maxIterations / self.granularity) * 100)
            
            current = datetime.now()
            
            # time calculations
            timeElapsed = (current - self.start)
            timePerStep = timeElapsed / (currentIteration + 1)
            totalEstimatedTime = timePerStep * self.maxIterations
            timeRemaining = totalEstimatedTime - timeElapsed
            
            # string formatting
            percentageStr = "{:>3}%  ".format(percentage)
            remainingStr = "Remaining: {}  ".format(deltaToString(timeRemaining))
            elapsedStr = "Elapsed: {}  ".format(deltaToString(timeElapsed))
            totalStr = "Total: {}\r".format(deltaToString(totalEstimatedTime))
            
            print(percentageStr + remainingStr + elapsedStr + totalStr, end="")

    def end(self):
        print()


# In[]:


def getAgentsChunked(dataset, subsetPercent=1, chunks=10):

    datasetLength = round(len(dataset) * subsetPercent)
    chunkSize = round(datasetLength / chunks)
    
    pb = ProgressBar(datasetLength)
    pb.start()

    agents = []
    for i in range(0, datasetLength, chunkSize):

        agentsSubset = dataset[i:i+chunkSize]
        for j in range(0,len(agentsSubset)):

            agent = agentsSubset[j]
            track_id = agent[4]

            if track_id >= len(agents):
                agents.append([])

            data = []
            centroid = agent[0]
            yaw = agent[2]
            velocity = agent[3]
            data.append(centroid[0])
            data.append(centroid[1])
            data.append(yaw)
            data.append(velocity[0])
            data.append(velocity[1])
            agents[int(track_id)-1].append(data)
        pb.check(i, True)

    return agents


# In[]:


print(zarr_dataset.agents, "\n")
print(type(zarr_dataset.agents[0][0][0]))
print(type(zarr_dataset.agents[0][0]))
print(type(zarr_dataset.agents[0]))
print(type(zarr_dataset.agents))
agents = []
print(type(agents))


# In[]:


subsetPercent = 1 #1*10**-2
print(subsetPercent)
agents = getAgentsChunked(zarr_dataset.agents, subsetPercent, 100)


# In[]:


def plotAgents(agents):
    r = lambda: random.randint(0,255)
    pb = ProgressBar(len(agents))
    pb.start()
    for i in range(0, len(agents)):
        agent = agents[i]
        centroid_x = []
        centroid_y = []
        for centroid in agent:
            centroid_x.append(centroid[0])
            centroid_y.append(centroid[1])
        plt.plot(centroid_x, centroid_y, 'o', color='#%02X%02X%02X' % (r(),r(),r()))
        pb.check(i)


# In[]:


plotAgents(agents)


# In[]:


def normalizeAgents(agents):
    dataForNormalization = []
    pb = ProgressBar(len(agents))
    pb.start()
    for agent in agents:
        pb.check(0, True)
        for data in agent:
            for i in range(0, len(data)):
                feature = data[i]
                if i >= len(dataForNormalization):
                    dataForNormalization.append([])
                dataForNormalization[i].append(feature)
        
    
    first = True
    normalizedAgents = []
    pb = ProgressBar(len(dataForNormalization) * len(agents))
    pb.start()
    for i in range(0, len(dataForNormalization)):
        pb.end()
        data = dataForNormalization[i]
        mean = np.mean(data)
        std = np.std(data)
        print("max[{}]".format(i),np.max(data))
        print("min[{}]".format(i),np.min(data))
        print("mean[{}]".format(i),mean)
        print("std[{}]".format(i),std, "\n")
        
        for j in range(0, len(agents)):
            pb.check(i * j)
            if j >= len(normalizedAgents):
                normalizedAgents.append([])
                
            agent = agents[j]
            normalizedAgent = normalizedAgents[j]
            
            for k in range(0, len(agent)):
                if k >= len(normalizedAgent):
                    normalizedAgent.append([])
                data = agent[k]
                normalizedData = normalizedAgent[k]
                
                feature = data[i]
                normalizedFeature = (feature - mean) / std
                if i == 0 and first:
                    print(feature)
                    print(normalizedFeature)
                    first = False
                
                if i >= len(normalizedData):
                    normalizedData.append(0)
                normalizedData[i] = normalizedFeature
    return normalizedAgents


# In[]:


import copy


# In[]:


normalizedAgents = normalizeAgents(agents)


# In[]:


print(len(agents))
print(len(normalizedAgents),"\n")

print(agents[0][0][0])
print(normalizedAgents[0][0][0],"\n")


# In[]:


def printAgentsInfo(agents, limit):
    print("len(agents)", len(agents), "\n")

    agentCentroidLengths = []
    agentsOverLimit = []
    for agent in agents:
        agentCentroidLengths.append(len(agent))
        if len(agent) > limit:
            agentsOverLimit.append(agent)

    print("len(agentCentroidLengths)",len(agentCentroidLengths), "\n")

    print("max",np.max(agentCentroidLengths))
    print("min",np.min(agentCentroidLengths))
    print("mean",np.mean(agentCentroidLengths))
    print("std",np.std(agentCentroidLengths), "\n")

    print("agents with {}+ history".format(limit),len(agentsOverLimit))
    return agentsOverLimit


# In[]:


limit = 10
agentsOverLimit = printAgentsInfo(normalizedAgents, limit)


# In[]:


def getTrainingSets(agents, limit):
    allTrainingSets = []
    totalNumberOfTrainingSets = 0
    
    pb = ProgressBar(len(agentsOverLimit))
    pb.start()
    for i in range(0, len(agentsOverLimit)):
        agent = agentsOverLimit[i]
        agentTrainingSets = []
        for i in range(limit, len(agent)-1):
            agentTrainingSet = []

            start = i - limit
            end = i
            output = i + 1

            agentTrainingSet.append(agent[start:end])
            agentTrainingSet.append(agent[output])
            agentTrainingSets.append(agentTrainingSet)

            totalNumberOfTrainingSets = totalNumberOfTrainingSets + 1

        allTrainingSets.append(agentTrainingSets)
        pb.check(i)

    print("len(allTrainingSets)", len(allTrainingSets))
    print("len(allTrainingSets[0])",len(allTrainingSets[0]), "\n")

    print("len(agentsOverLimit)",len(agentsOverLimit))
    print("len(agentsOverLimit[0]) - limit - 1",len(agentsOverLimit[0]) - limit - 1, "\n")

    print("totalNumberOfTrainingSets",totalNumberOfTrainingSets)
    return allTrainingSets


# In[]:


allTrainingSets = getTrainingSets(agentsOverLimit, limit)


# In[]:


def flattenTrainingSets(allTrainingSets):
    allTrainingSetsFlattened_X = []
    allTrainingSetsFlattened_Y = []
    for allTrainingSet in allTrainingSets:
        for trainingSet in allTrainingSet:
            allTrainingSetsFlattened_X.append(trainingSet[0])
            allTrainingSetsFlattened_Y.append(trainingSet[1])
    print("len(allTrainingSetsFlattened_X)", len(allTrainingSetsFlattened_X))
    return allTrainingSetsFlattened_X, allTrainingSetsFlattened_Y


# In[]:


allTrainingSetsFlattened_X, allTrainingSetsFlattened_Y = flattenTrainingSets(allTrainingSets)


# In[]:


def reshapeFlattenedTrainingSets(allTrainingSetsFlattened_X, allTrainingSetsFlattened_Y):
    length = len(allTrainingSetsFlattened_X)
    depth = len(allTrainingSetsFlattened_X[0])
    channels = len(allTrainingSetsFlattened_X[0][0])

    print("length", length)
    print("depth", depth)
    print("channels",channels)
    print("length*depth*channels",length*depth*channels)

    allTrainingSetsFlattened_Input = np.reshape(allTrainingSetsFlattened_X, (length,depth,channels))
    allTrainingSetsFlattened_Output = np.reshape(allTrainingSetsFlattened_Y, (length,1,channels))

    print(allTrainingSetsFlattened_Input.shape[1])
    print(allTrainingSetsFlattened_Input.shape[2])
    
    return allTrainingSetsFlattened_Input, allTrainingSetsFlattened_Output


# In[]:


allTrainingSetsFlattened_Input = allTrainingSetsFlattened_X
allTrainingSetsFlattened_Output = allTrainingSetsFlattened_Y


# In[]:


allTrainingSetsFlattened_Input, allTrainingSetsFlattened_Output = reshapeFlattenedTrainingSets(allTrainingSetsFlattened_X, allTrainingSetsFlattened_Y)


# In[]:


# The LSTM architecture
regressor = Sequential()
# First LSTM layer with Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(allTrainingSetsFlattened_Input.shape[1],allTrainingSetsFlattened_Input.shape[2])))
regressor.add(Dropout(0.2))
# Second LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Third LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Fourth LSTM layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
# The output layer
regressor.add(Dense(units=allTrainingSetsFlattened_Input.shape[2]))

# Compiling the RNN
regressor.compile(optimizer='rmsprop',loss='mean_squared_error')


# In[]:


from tensorflow import keras


# In[]:


# Fitting to the training set

class CustomCallback(keras.callbacks.Callback):
    
    def __init__(self):
        self.epoch = 0
        
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Epoch: {}             loss: {}\n".format(self.epoch, logs['loss']), end="")
        self.epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        if batch % 100 == 0:
            print("Epoch: {} batchs: {}% loss: {}\r".format(self.epoch, round(batch / self.params['steps'] * 100), logs['loss']), end="")

regressor.fit(allTrainingSetsFlattened_Input,allTrainingSetsFlattened_Output,epochs=2,batch_size=128,verbose=0,callbacks=[CustomCallback()])


# In[]:


dataset_path_test = dm.require('scenes/test.zarr')
zarr_dataset_test = ChunkedDataset(dataset_path_test)
zarr_dataset_test.open()
print(zarr_dataset_test)


# In[]:


print(len(zarr_dataset_test.agents))


# In[]:


subsetPercent = 1*10**-3
print(subsetPercent)
agentsTest = getAgentsChunked(zarr_dataset_test.agents, subsetPercent, 1000)


# In[]:


plotAgents(agents)


# In[]:


normalizedAgentsTest = normalizeAgents(agentsTest)


# In[]:


agentsTestOverLimit = printAgentsInfo(normalizedAgentsTest, limit)


# In[]:


allTestingSets = getTrainingSets(agentsTestOverLimit, limit)


# In[]:


allTestingSetsFlattened_X, allTestingSetsFlattened_Y = flattenTrainingSets(allTestingSets)


# In[]:


allTestingSetsFlattened_Input, allTestingSetsFlattened_Output = reshapeFlattenedTrainingSets(allTestingSetsFlattened_X, allTestingSetsFlattened_Y)


# In[]:


max = len(allTestingSetsFlattened_Input)
print(max)
chunkSize = 1000
pb = ProgressBar(max)
pb.start()
predictedTestAgentCentroid = np.empty((1,5))
for i in range(0, max-chunkSize, chunkSize):#len(zarr_dataset.agents)):
    newPredictions = regressor.predict(allTestingSetsFlattened_Input[i:i+chunkSize])
    predictedTestAgentCentroid = np.concatenate((predictedTestAgentCentroid, newPredictions))
    pb.check(i, True)


# In[]:


print(len(predictedTestAgentCentroid))
predictedTestAgentCentroid = predictedTestAgentCentroid[1:len(predictedTestAgentCentroid)]
print(len(predictedTestAgentCentroid))


# In[]:


randomSamples = 10
for i in range(0, len(predictedTestAgentCentroid), round(len(predictedTestAgentCentroid) / randomSamples)):
    testSet = allTestingSetsFlattened_Input[i]
    print(testSet[0][0])
    print(predictedTestAgentCentroid[i][0],"\n")


# In[]:




