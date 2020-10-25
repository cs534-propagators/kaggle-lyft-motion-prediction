#>
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

#>
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

#>
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

#>
import time
from datetime import datetime

#>
os.environ["L5KIT_DATA_FOLDER"] = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"

#>
dm = LocalDataManager()
dataset_path = dm.require('scenes/sample.zarr')
zarr_dataset = ChunkedDataset(dataset_path)
zarr_dataset.open()
print(zarr_dataset)

#>
print(zarr_dataset.agents)
print(zarr_dataset.agents.shape)
n = zarr_dataset.agents.shape

#>
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

#>
def getAgentsChunked(dataset, subsetPercent=1, chunks=10):

    datasetLength = round(len(dataset) * subsetPercent)
    chunkSize = round(datasetLength / chunks)
    print("datasetLength", datasetLength)
    print("chunkSize", chunkSize)
    
    pb = ProgressBar(datasetLength)
    pb.start()

    track_id1_indexes = []
    for i in range(0, datasetLength, chunkSize):

        agentsSubset = dataset[i:i+chunkSize]
        for j in range(0,len(agentsSubset)):

            agent = agentsSubset[j]
            track_id = agent[4]
            if track_id == 1:
                track_id1_indexes.append(i+j)
        pb.check(i, True)
    pb.end()
    return agents, track_id1_indexes

#>
subsetPercent = 1 #1*10**-1
print(subsetPercent)
agents, track_id1_indexes = getAgentsChunked(zarr_dataset.agents, subsetPercent, 100)

#>
hertz = 10 # frames per second
secondsPerMinute = 60

print("frames", len(track_id1_indexes))
print("seconds", len(track_id1_indexes)/hertz)
print("minutes", len(track_id1_indexes)/hertz/secondsPerMinute)

print(track_id1_indexes[0:10])
print(track_id1_indexes[len(track_id1_indexes)-10:len(track_id1_indexes)])

#>
frameIntervalIndex = 0
agentsIntervalIndex = 1
print(zarr_dataset.scenes[0][frameIntervalIndex])
print(zarr_dataset.frames[248][agentsIntervalIndex])

#>
subsetPercent = 21347 / len(zarr_dataset.agents)
print(subsetPercent)
print(subsetPercent*len(zarr_dataset.agents))
agents, track_id1_indexes = getAgentsChunked(zarr_dataset.agents, subsetPercent, 100)

#>
print(len(track_id1_indexes))
print(track_id1_indexes)

#>
track_id1_indexes_pointer = 0
framesFound = []
for i in range(0, 249):
    frame = zarr_dataset.frames[i]
    agentsInterval = frame[agentsIntervalIndex]
    start = agentsInterval[0]
    end = agentsInterval[1]
    
    track_id1_index = track_id1_indexes[track_id1_indexes_pointer]
    print("start", start, "track_id1_index", track_id1_index, "end", end)
    if start <= track_id1_index and track_id1_index < end:
        framesFound.append(i)
        track_id1_indexes_pointer += 1
framesFound

#>
frame = zarr_dataset.frames[248]
agentsInterval = frame[agentsIntervalIndex]
print(agentsInterval)

#>


#>
dataset_path_test = dm.require('scenes/test.zarr')
zarr_dataset_test = ChunkedDataset(dataset_path_test)
zarr_dataset_test.open()
print(zarr_dataset_test)

#>
test_mask = np.load('../input/lyft-motion-prediction-autonomous-vehicles/scenes/mask.npz')
for k in test_mask.files:
    print("key:",k)
test_mask = test_mask["arr_0"]
print("test_mask", test_mask)
print("test_mask.shape", test_mask.shape)
print("test_mask[0]", test_mask[0])

#>
subsetPercent = 1*10**-1
subsetLength = round(len(test_mask) * subsetPercent)
print("subsetLength", subsetLength)
count = 0
pb = ProgressBar(subsetLength)
pb.start()
chunkSize = 100
mask_copy = []
mask_indexes = []
for i in range(0, subsetLength, chunkSize):
    chunkedTestMask = test_mask[i: i + chunkSize]
    for j in range(0, len(chunkedTestMask)):
        mask = chunkedTestMask[j]
        mask_copy.append(mask)
        if mask:
            mask_indexes.append(i + j)
            count = count + 1
    pb.check(i)
pb.end()
print("count", count)

#>
print(len(mask_indexes))
print(len(mask_indexes)/subsetPercent)
print(len(test_mask))
print(len(mask_copy)/subsetPercent)
print(len(mask_copy))

#>
track_id_indexes = {}
pb = ProgressBar(len(mask_indexes))
pb.start()
for i in range(0, len(mask_indexes)):
    mask_index = mask_indexes[i]
    agent = zarr_dataset_test.agents[mask_index]
    track_id = agent[4]
    if track_id not in track_id_indexes:
        track_id_indexes[track_id] = []
    track_id_indexes[track_id].append(mask_index)
    pb.check(i)

#>
for key in track_id_indexes:
    track_id_index = track_id_indexes[key]
    print(len(track_id_index))

#>
print(len(track_id2_indexes))
print(track_id2_indexes[0])
print(zarr_dataset_test.frames[99][1])
print(track_id2_indexes[319])
print(zarr_dataset_test.frames[110399][1])

#>

